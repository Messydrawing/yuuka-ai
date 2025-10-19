# -*- coding: utf-8 -*-
"""
本地 Web UI：优香（Qwen2.5-7B-Instruct + 你的 LoRA）
- 使用 Qwen 官方 chat template（apply_chat_template）构造对话输入
- 会话记忆（持久化）：保留最近对话并自动分层摘要
- 关键术语提示：persona 常载 + memory.jsonl 动态命中（5 轮衰减）
- 多样化输出：候选采样K（1~3）并选取与最近回复相似度最低者
- “继续说”与“重新说”：继续隐式续写或回放上一条回复
- 推理：默认 4-bFit NF4（BitsAndBytes）；可选 lashAttention2（若环境支持）
"""

import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["no_proxy"] = "127.0.0.1,localhost,::1"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

import json
import time
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Set, Optional

import gradio as gr

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ================== 路径/目录 ==================
OFFLOAD_DIR = Path("offload")
OFFLOAD_DIR.mkdir(exist_ok=True, parents=True)

# ✅ 改为 Qwen2.5-Instruct（本地目录或 Hub 名）
BASE_MODEL = "models/Qwen2.5-7B-Instruct"      # 例如 "Qwen/Qwen2.5-7B-Instruct"
LORA_DIR   = "models/yuuka_qwen-lora8"         # 你的 LoRA 输出目录

MEM_DIR = Path("memory")
MEM_DIR.mkdir(exist_ok=True, parents=True)
DEFAULT_MEM_ID = "default"
PERSONA_FILE = MEM_DIR / "persona_yuuka.txt"
KB_DIR = Path("kb")
KB_DIR.mkdir(exist_ok=True, parents=True)
KB_MEMORY_FILE = KB_DIR / "hard_memory.jsonl"

# 生成超参
MAX_NEW_TOKENS = 96
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 0.9
DEFAULT_MIN_NEW_TOKENS = 48
DEFAULT_LENGTH_PENALTY = 1.05
MIN_REPLY_CHARS = 35
TARGET_REPLY_CHARS = 90
AUTO_CONTINUE_MIN_CHARS = 48
AUTO_CONTINUE_MAX_ROUNDS = 2

# 记忆分层参数
RAW_KEEP_TURNS = 100
BLOCK_RANGE_TURNS = 100
BLOCK_SIZE = 10
SPARSE_STRIDE = 5
SUPER_SUMMARY_EVERY = 50

# =============== Persona 设定与关键词 ===============
PERSONA_STYLE_KEYWORDS = ["老师", "预算", "盘点", "纠偏", "唔", "哼"]
ACCOUNTING_KEYWORDS = {"收据", "发票", "预算", "金额", "价", "购物", "报销", "支出", "欠款", "超支", "账单"}
ACCOUNTING_ALERT = "（注意：当前出现财务关键词，请从专业的角度给出分析。）"
SMALL_TALK_KEYWORDS = {
    "你好", "您好", "嗨", "哈喽", "hello", "hi", "在吗", "在么", "晚安", "早安",
    "早上好", "上午好", "中午好", "下午好", "晚上好", "最近好吗", "最近怎么样", "吃饭了吗",
    "干嘛呢", "聊聊", "聊聊天", "待会见", "回来了", "拜拜", "再见", "辛苦了", "辛苦啦",
}

# =============== 分词 / 匹配辅助 ===============
def _normalize_for_small_talk(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z一-鿿]+", "", t)
    return t


def is_small_talk(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    normalized = _normalize_for_small_talk(stripped)
    if not normalized:
        return False
    if len(normalized) <= 6 and any(kw in normalized for kw in SMALL_TALK_KEYWORDS):
        return True
    if len(normalized) <= 10:
        for kw in SMALL_TALK_KEYWORDS:
            if normalized.endswith(kw) or normalized.startswith(kw):
                return True
    return False


def build_persona_block() -> str:
    if PERSONA_FILE.exists():
        text = PERSONA_FILE.read_text(encoding="utf-8").strip()
        return text or "（未设置人物设定）"
    return "（未设置人物设定）"

# =============== 系统提示模板与关键术语记忆 ===============
FRAME_PROMPT_TEMPLATE = (
    "你是早濑优香，请你基于以下内容给出对用户的回答："
    "1.你的人物设定：{persona}"
    "2.你的历史记忆：{history}"
    "3.关键术语：{keywords}；用户（老师）输入：{user_input}"
)


def _extract_chinese_chars(text: str) -> Set[str]:
    return {ch for ch in (text or "") if "\u4e00" <= ch <= "\u9fff"}


class KeywordMemory:
    def __init__(self, path: Path):
        self.path = path
        self.entries: List[Dict[str, Any]] = []
        self.load()

    def load(self):
        self.entries.clear()
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entity = (obj.get("entity") or "").strip()
                text = (obj.get("text") or "").strip()
                if not entity or not text:
                    continue
                entity_chars = _extract_chinese_chars(entity)
                if len(entity_chars) < 2:
                    continue
                self.entries.append({
                    "entity": entity,
                    "text": text,
                    "chars": entity_chars,
                })

    def match(self, query: str) -> List[Dict[str, Any]]:
        user_chars = _extract_chinese_chars(query)
        if len(user_chars) < 2:
            return []
        matches: List[Dict[str, Any]] = []
        for entry in self.entries:
            if len(user_chars & entry["chars"]) >= 2:
                matches.append(entry)
        return matches


def new_dynamic_state() -> Dict[str, Any]:
    return {"turn": 0, "active": {}}


def ensure_dynamic_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return new_dynamic_state()
    if "turn" not in state or "active" not in state:
        return new_dynamic_state()
    if not isinstance(state["active"], dict):
        state["active"] = {}
    return state


def _prune_dynamic_state(state: Dict[str, Any]):
    threshold = int(state.get("turn", 0)) - 5
    remove_keys = [ent for ent, rec in state["active"].items() if int(rec.get("last_turn", -10**6)) <= threshold]
    for ent in remove_keys:
        state["active"].pop(ent, None)


def collect_dynamic_keywords(state: Dict[str, Any]) -> List[str]:
    state = ensure_dynamic_state(state)
    ordered = sorted(
        state["active"].items(),
        key=lambda kv: (-int(kv[1].get("last_turn", 0)), kv[0])
    )
    texts: List[str] = []
    for _, rec in ordered:
        for txt in rec.get("texts", []):
            if txt not in texts:
                texts.append(txt)
    return texts


def advance_dynamic_state(state: Dict[str, Any], user_text: str, kb: KeywordMemory, enabled: bool) -> Tuple[Dict[str, Any], List[str]]:
    state = ensure_dynamic_state(state)
    state["turn"] = int(state.get("turn", 0)) + 1
    if enabled:
        matches = kb.match(user_text)
        for entry in matches:
            rec = state["active"].setdefault(entry["entity"], {"texts": [], "last_turn": state["turn"]})
            rec["last_turn"] = state["turn"]
            if entry["text"] not in rec["texts"]:
                rec["texts"].append(entry["text"])
    _prune_dynamic_state(state)
    keywords = collect_dynamic_keywords(state) if enabled else []
    return state, keywords


# =============== 会话记忆持久化 ===============
def mem_path(mem_id: str) -> Path:
    safe = "".join(c for c in mem_id if c.isalnum() or c in ("-", "_"))
    return MEM_DIR / f"{safe or 'default'}.json"

def new_memory(mem_id: str) -> Dict:
    return {
        "id": mem_id,
        "created_at": int(time.time()),
        "turns": [],  # [{u, a}]
        "block_summaries": [],
        "sparse": [],
        "super_summary": ""
    }

def load_memory(mem_id: str) -> Dict:
    p = mem_path(mem_id)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return new_memory(mem_id)

def save_memory(mem: Dict):
    with open(mem_path(mem["id"]), "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)

def ensure_teacher_prefix(s: str) -> str:
    if s.startswith("[") and "]" in s.split(" ", 1)[0]:
        return s
    return f"[老师] {s}"

# =============== 文本后处理（去开场白） ===============
_BAD_STARTS = ["好的", "好吧", "行吧", "行的", "明白", "了解", "那我", "那就", "嗯", "啊这", "好的，我", "好的，那"]
BANNED_PHRASES = [
    "作为AI", "作为 AI", "作为一个AI", "作为大语言模型", "语言模型", "我只是一个AI",
    "我不能", "无法访问网络", "抱歉", "sourceMappingURL", "Teacher", "teacher", "sensei"
]

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_TAG_RE = re.compile(r"(?:^|\s)#\S+")


def extract_first_sentence(text: str) -> str:
    if not text:
        return text
    stripped = text.strip()
    if not stripped:
        return stripped
    for idx, ch in enumerate(stripped):
        if ch in "。？！?!":
            return stripped[:idx + 1].strip()
    first_line = stripped.splitlines()[0].strip()
    return first_line or stripped


def _sanitize(text: str) -> str:
    if not text:
        return text
    t = _URL_RE.sub("", text)
    t = _TAG_RE.sub("", t)
    t = t.replace("Teacher", "老师").replace("teacher", "老师").replace("sensei", "老师")
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def _limit_sentences(text: str, max_sents: int = 2, max_questions: int = 1) -> str:
    import re as _re

    if not text:
        return text

    sents = _re.split(r"(?<=[。！？!?])\s*", text)
    sents = [s.strip() for s in sents if s.strip()]
    kept = []
    question_count = 0
    for s in sents:
        has_question = ('?' in s) or ('？' in s)
        if has_question and question_count >= max_questions:
            continue
        question_count += 1 if has_question else 0
        kept.append(s)
        if len(kept) >= max_sents:
            break
    if not kept and sents:
        kept.append(sents[0])
    return " ".join(kept) if kept else text

def _strip_bland_openers(text: str) -> str:
    t = text.lstrip()
    for _ in range(3):
        matched = False
        for p in _BAD_STARTS:
            if t.startswith(p):
                cut = None
                for sep in ['\n', '。', '！', '？', '，', '.', '!', '?', ':', '：']:
                    pos = t.find(sep)
                    if pos != -1:
                        cut = pos + 1
                        break
                t = (t[cut:].lstrip() if cut else t[len(p):].lstrip())
                matched = True
                break
        if not matched:
            break
    return t or text

def violates_rules(text: str) -> bool:
    return any(bad in text for bad in BANNED_PHRASES)


def refine_short(text: str, tokenizer, model) -> str:
    if not text:
        return text

    sys = "把回答改写为 1-2 句，逻辑连贯，不新增任何设定，最多一个澄清问句。"
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": text},
    ]
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            ids,
            do_sample=True,
            temperature=0.2,
            typical_p=0.95,
            max_new_tokens=96,
            use_cache=True,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    gen = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    refined = _limit_sentences(_sanitize(gen), 2, 1)
    return refined or text

def should_auto_continue(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if len(stripped) < AUTO_CONTINUE_MIN_CHARS:
        return True
    return stripped.endswith(("…", "......", "...", "……", "，", "、"))

def jaccard_sim(a: str, b: str) -> float:
    sa = set(a.split()); sb = set(b.split())
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def score_candidate(text: str, recent_ref: Optional[str] = None) -> float:
    stripped = (text or "").strip()
    if not stripped:
        return -1e9
    length = len(stripped)
    if length < MIN_REPLY_CHARS:
        return -200 + length
    length_score = min(length / TARGET_REPLY_CHARS, 1.4)
    if length > TARGET_REPLY_CHARS * 1.6:
        length_score -= 0.4
    persona_hits = sum(1 for kw in PERSONA_STYLE_KEYWORDS if kw in stripped)
    persona_score = persona_hits * 0.35
    penalty = 0.0
    if stripped.endswith(("…", "......", "...", "……")):
        penalty -= 0.2
    if recent_ref:
        penalty -= jaccard_sim(stripped, recent_ref.strip()) * 0.8
    return length_score + persona_score + penalty

# =============== 模型加载（Qwen2.5 推理路径） ===============
def load_model():
    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, local_files_only=True
    )
    # 仅在缺省时设置 pad 为 eos
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # 默认 4-bit（NF4）；显存充足可改为 bf16/fp16（去掉 quantization_config）
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 可选：FlashAttention2（若环境不支持会回退）
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
            quantization_config=bnb_cfg,
            attn_implementation="flash_attention_2",  # 若报错将走 except 回退
            low_cpu_mem_usage=True,
            offload_folder=str(OFFLOAD_DIR),
            local_files_only=True,
        )
    except Exception:
        mdl = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
            quantization_config=bnb_cfg,
            low_cpu_mem_usage=True,
            offload_folder=str(OFFLOAD_DIR),
            local_files_only=True,
        )

    # 加载 LoRA 适配
    mdl = PeftModel.from_pretrained(
        mdl,
        LORA_DIR,
        device_map="auto",
        offload_folder=str(OFFLOAD_DIR),
        local_files_only=True,
    )
    mdl.eval()

    # 生成更友好：保持 use_cache=True
    try:
        mdl.config.use_cache = True
        if hasattr(mdl, "generation_config") and mdl.generation_config is not None:
            mdl.generation_config.use_cache = True
    except Exception:
        pass

    # 对齐 pad/eos
    if tok.pad_token_id is not None:
        mdl.config.pad_token_id = tok.pad_token_id
        if hasattr(mdl, "generation_config") and mdl.generation_config is not None:
            mdl.generation_config.pad_token_id = tok.pad_token_id
    if tok.eos_token_id is not None:
        mdl.config.eos_token_id = tok.eos_token_id

    return tok, mdl

# ====== 初始化 ======
tokenizer, model = load_model()

BAN_STRINGS = ["http", "https", "www.", "sourceMappingURL", "#", "Teacher", "teacher", "sensei"]
BAD_WORDS_IDS = [ids for s in BAN_STRINGS if (ids := tokenizer.encode(s, add_special_tokens=False))]
keyword_memory = KeywordMemory(KB_MEMORY_FILE)

# =============== Chat 模板与编码（Qwen 官方方式） ===============
def _apply_chat_template(msgs: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
    """
    使用 tokenizer.apply_chat_template 生成单条文本，再一次性分词为张量。
    """
    text = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=getattr(tokenizer, "chat_template", None)
    )
    model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    return {k: v.to(model.device) for k, v in model_inputs.items()}

# =============== 记忆压缩（摘要亦走同一生成通道） ===============
def summarize_turns(turns: List[Dict], max_tokens=160) -> str:
    text_lines = []
    for t in turns:
        text_lines.append(f"[老师]{t['u']}")
        text_lines.append(f"[优香]{t['a']}")
    convo = "\n".join(text_lines[-60:])
    sys = "你是总结助手。请把以下对话概括为2-4行要点，保留人名/事件/态度，不新增信息，简体中文。"
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": convo}]
    inputs = _apply_chat_template(msgs)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=False,
            use_cache=True,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    summary = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return summary

def refresh_memory_layers(mem: Dict):
    n = len(mem["turns"])
    if n <= RAW_KEEP_TURNS:
        return
    start_block = max(0, n - RAW_KEEP_TURNS - BLOCK_RANGE_TURNS)
    end_block = max(0, n - RAW_KEEP_TURNS)
    if end_block - start_block >= BLOCK_SIZE:
        done_until = 0
        if mem["block_summaries"]:
            done_until = max(b["end"] for b in mem["block_summaries"]) + 1
        i = max(done_until, start_block)
        while i + BLOCK_SIZE <= end_block:
            block = mem["turns"][i:i+BLOCK_SIZE]
            summ = summarize_turns(block)
            mem["block_summaries"].append({"start": i, "end": i+BLOCK_SIZE-1, "summary": summ})
            i += BLOCK_SIZE

    sparse_start = 0
    if mem["sparse"]:
        sparse_start = max(s["idx"] for s in mem["sparse"]) + 1
    sparse_end = max(0, n - RAW_KEEP_TURNS - BLOCK_RANGE_TURNS)
    for idx in range(sparse_start, sparse_end, SPARSE_STRIDE):
        if idx < len(mem["turns"]):
            t = mem["turns"][idx]
            mem["sparse"].append({"idx": idx, "u": t["u"], "a": t["a"]})

    if n % SUPER_SUMMARY_EVERY == 0:
        early = mem["turns"][:max(0, n - RAW_KEEP_TURNS - BLOCK_RANGE_TURNS)]
        if early:
            sample = early[::SPARSE_STRIDE][:50]
            mem["super_summary"] = summarize_turns(sample, max_tokens=120)

def build_memory_context(mem: Dict) -> str:
    lines = []
    recent = mem["turns"][-6:]
    if recent:
        lines.append("【近期对话要点】")
        for t in recent:
            u = t["u"].strip(); a = t["a"].strip()
            if len(u) > 60: u = u[:60] + "…"
            if len(a) > 60: a = a[:60] + "…"
            lines.append(f"- [老师]{u}")
            lines.append(f"- [优香]{a}")
    if mem["block_summaries"]:
        lines.append("【较早摘要】")
        for b in mem["block_summaries"][-2:]:
            s = b["summary"].strip()
            lines.append(f"- {s if len(s)<=200 else s[:200]+'…'}")
    if mem.get("super_summary"):
        lines.append("【更早的整体回顾】")
        ss = mem["super_summary"].strip()
        lines.append(f"- {ss if len(ss)<=220 else ss[:220]+'…'}")
    return "\n".join(lines)

# =============== 生成 & 多样化 ===============
def generate_one(
    msgs: List[Dict[str,str]],
    max_new_tokens,
    temperature,
    top_p,
    repetition_penalty,
    no_repeat_ngram_size,
    typical_p=None,
    length_penalty=DEFAULT_LENGTH_PENALTY,
    min_new_tokens=0,
    seed=None
) -> str:
    if seed is not None:
        torch.manual_seed(seed); random.seed(seed)

    inputs = _apply_chat_template(msgs)

    min_new = int(min_new_tokens) if min_new_tokens else 0
    if max_new_tokens and min_new >= max_new_tokens:
        min_new = max(1, max_new_tokens - 1)

    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            min_new_tokens=max(1, int(min_new)) if max_new_tokens else None,
            do_sample=True,
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            length_penalty=float(length_penalty),
            bad_words_ids=BAD_WORDS_IDS,
            use_cache=True,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        if typical_p is not None and float(typical_p) < 0.999:
            gen_kwargs["typical_p"] = float(typical_p)
        else:
            gen_kwargs["top_p"] = float(top_p)

        out = model.generate(
            **inputs,
            **gen_kwargs,
        )
    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    text = _strip_bland_openers(text)
    text = _sanitize(text)
    return _limit_sentences(text, max_sents=2, max_questions=1)

def generate_diverse(msgs, k_candidates, history_ui_recent, **gen_kwargs):
    recent_ref = ""
    if history_ui_recent:
        for i in range(len(history_ui_recent)-1, -1, -1):
            if history_ui_recent[i][1]:
                recent_ref = history_ui_recent[i][1]
                break
    cands = []
    for _ in range(k_candidates):
        seed = random.randint(1, 10**9)
        text = generate_one(msgs, seed=seed, **gen_kwargs)
        cands.append(text)
    filtered = [c for c in cands if c and not violates_rules(c)]
    if not filtered:
        filtered = [c for c in cands if c]
    if not filtered:
        return ""
    filtered.sort(key=lambda txt: score_candidate(txt, recent_ref), reverse=True)
    return filtered[0]

# =============== 构造消息 ===============
def format_system_prompt(persona_text: str, history_text: str, keyword_texts: List[str], user_input: str) -> str:
    persona = persona_text.strip() if persona_text else "（未设置人物设定）"
    history = history_text.strip() if history_text else "（暂无历史记忆）"
    keywords = "；".join(keyword_texts) if keyword_texts else "（暂无关键术语）"
    user = user_input.strip() if user_input else "（空输入）"
    return FRAME_PROMPT_TEMPLATE.format(
        persona=persona,
        history=history,
        keywords=keywords,
        user_input=user,
    )


def build_messages(history_msgs: List[Dict[str, str]], user_text: str,
                   persona_text: str, history_text: str,
                   keyword_texts: List[str]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    sys_content = format_system_prompt(persona_text, history_text, keyword_texts, user_text)
    msgs.append({"role": "system", "content": sys_content})
    for m in history_msgs:
        msgs.append(m)
    msgs.append({"role": "user", "content": ensure_teacher_prefix(user_text)})
    return msgs


def build_messages_for_continue(history_msgs: List[Dict[str, str]],
                                ui_hist: List[Tuple[str, str]],
                                persona_text: str, history_text: str,
                                keyword_texts: List[str]) -> Tuple[List[Dict[str, str]], str]:
    """不把“继续”显示到UI；仅构造临时 msgs 以续写"""
    frag_pairs = ui_hist[-3:] if ui_hist else []
    lines = []
    for u, a in frag_pairs:
        if u:
            lines.append(f"[老师]{u}")
        if a:
            lines.append(f"[优香]{a}")
    recent_frag = "\n".join(lines[-12:])
    hidden_user = "继续（不要重复开场白，直接承接你上一条的语气与内容展开；如需列点，从上次编号往下接）。\n【最近片段】\n" + recent_frag

    sys_content = format_system_prompt(persona_text, history_text, keyword_texts, hidden_user)
    msgs: List[Dict[str, str]] = []
    msgs.append({"role": "system", "content": sys_content})
    for m in history_msgs:
        msgs.append(m)
    msgs.append({"role": "user", "content": hidden_user})
    return msgs, hidden_user


# =============== Gradio UI ===============
with gr.Blocks(title="yuuka-ai") as demo:
    gr.Markdown("### 优香（Qwen2.5 LoRA） · 会话记忆 + 关键术语提示\n_系统设定隐藏；支持加载/保存记忆、多样化输出与“继续说/重新说”。_")

    with gr.Row():
        use_rag = gr.Checkbox(label="启用关键术语提示", value=True)
        mem_enabled = gr.Checkbox(label="启用会话记忆（持久化）", value=True)
        mem_id = gr.Textbox(label="记忆ID（文件名）", value=DEFAULT_MEM_ID, scale=2)
        btn_load_mem = gr.Button("加载记忆", variant="secondary")
        btn_save_mem = gr.Button("保存记忆", variant="secondary")
        btn_clear_mem = gr.Button("清空记忆", variant="stop")

    with gr.Row():
        temperature = gr.Slider(0.2, 1.2, value=DEFAULT_TEMPERATURE, step=0.05, label="温度")
        top_p = gr.Slider(0.5, 1.0, value=DEFAULT_TOP_P, step=0.01, label="Top-p")
        typical_slider = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="typical_p（与 Top-p 二选一）")
        repetition_penalty = gr.Slider(1.0, 1.3, value=1.2, step=0.01, label="重复惩罚")
        no_repeat_ngram = gr.Slider(0, 8, value=6, step=1, label="no_repeat_ngram_size")

    with gr.Row():
        diverse = gr.Checkbox(label="多样化输出（候选采样）", value=True)
        k_candidates = gr.Slider(1, 3, value=2, step=1, label="候选数 k")
        max_new = gr.Slider(64, 512, value=MAX_NEW_TOKENS, step=16, label="最大生成长度")
        refine_enabled = gr.Checkbox(label="启用二阶段精修", value=False)

    chat = gr.Chatbot(height=520, label="与优香对话", type="tuples")
    txt = gr.Textbox(placeholder="和优香聊点什么吧…（直接输入即可）", show_label=False)
    with gr.Row():
        send = gr.Button("发送", variant="primary")
        btn_continue = gr.Button("继续说 ▶", variant="secondary")
        btn_repeat = gr.Button("重新说 ⟳", variant="secondary")
        clear = gr.Button("清空对话", variant="secondary")

    state_msgs = gr.State([{"role":"system","content":"(hidden)"}])
    state_mem = gr.State(new_memory(DEFAULT_MEM_ID))
    state_dynamic = gr.State(new_dynamic_state())

    # === 记忆按钮 ===
    def do_load_mem(mem_id_text):
        mem = load_memory(mem_id_text or DEFAULT_MEM_ID)
        ui = [[t["u"], t["a"]] for t in mem["turns"][-10:]]
        gr.Info(f"已加载记忆：{mem_id_text or DEFAULT_MEM_ID}")
        return mem, ui
    btn_load_mem.click(do_load_mem, inputs=[mem_id], outputs=[state_mem, chat])

    def do_save_mem(mem_obj, mem_id_text):
        mem_obj["id"] = mem_id_text or DEFAULT_MEM_ID
        save_memory(mem_obj)
        gr.Info(f"已保存记忆：{mem_obj['id']}")
        return None
    btn_save_mem.click(do_save_mem, inputs=[state_mem, mem_id], outputs=[])

    def do_clear_mem(mem_id_text):
        m = new_memory(mem_id_text or DEFAULT_MEM_ID)
        save_memory(m)
        gr.Info("记忆与对话已清空")
        return m, [], [{"role": "system", "content": "(hidden)"}], new_dynamic_state()
    btn_clear_mem.click(do_clear_mem, inputs=[mem_id], outputs=[state_mem, chat, state_msgs, state_dynamic])

    # === 发消息 ===
    def _send(user_input, ui_hist, msg_hist, mem_obj, dyn_state,
              use_rag_flag, mem_en_flag,
              temp, topp, typical_val, rep_pen, ngram, diverse_flag, k_cand, max_tokens, refine_flag):

        if not user_input.strip():
            return ui_hist, msg_hist, mem_obj, ensure_dynamic_state(dyn_state)

        mem_obj = mem_obj or new_memory(DEFAULT_MEM_ID)
        dyn_state = ensure_dynamic_state(dyn_state)

        persona_text = build_persona_block()
        history_text = build_memory_context(mem_obj) if mem_obj.get("turns") else ""

        dyn_state, keyword_texts = advance_dynamic_state(dyn_state, user_input, keyword_memory, bool(use_rag_flag))
        keywords_for_prompt = list(keyword_texts)
        if any(kw in user_input for kw in ACCOUNTING_KEYWORDS):
            keywords_for_prompt = [ACCOUNTING_ALERT] + keywords_for_prompt

        msgs = build_messages(msg_hist or [{"role": "system", "content": "(hidden)"}],
                              user_input, persona_text, history_text, keywords_for_prompt)
        max_new = int(max_tokens)
        min_new = min(DEFAULT_MIN_NEW_TOKENS, max(1, max_new - 1)) if max_new > 1 else 1
        gen_kwargs = dict(
            max_new_tokens=max_new,
            temperature=float(temp),
            top_p=float(topp),
            typical_p=float(typical_val),
            repetition_penalty=float(rep_pen),
            no_repeat_ngram_size=int(ngram),
            length_penalty=DEFAULT_LENGTH_PENALTY,
            min_new_tokens=min_new,
        )

        def _sample_reply(msgs_for_gen, history_for_score):
            if diverse_flag and int(k_cand) > 1:
                result = generate_diverse(msgs_for_gen, int(k_cand), history_for_score or [], **gen_kwargs)
                if result:
                    return result
            return generate_one(msgs_for_gen, **gen_kwargs)

        reply = _sample_reply(msgs, ui_hist or []) or ""
        if refine_flag:
            reply = refine_short(reply, tokenizer, model)

        ui_hist_next = list(ui_hist or [])
        ui_hist_next.append([user_input, reply])
        msg_hist_next = list(msg_hist or [{"role": "system", "content": "(hidden)"}])
        msg_hist_next.extend([
            {"role": "user", "content": ensure_teacher_prefix(user_input)},
            {"role": "assistant", "content": reply},
        ])

        temp_ui = []
        for pair in ui_hist_next:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                temp_ui.append([pair[0], pair[1]])
            else:
                temp_ui.append([pair, ""])
        temp_msgs = list(msg_hist_next)
        current_reply = reply
        if not refine_flag:
            for _ in range(AUTO_CONTINUE_MAX_ROUNDS):
                if not should_auto_continue(current_reply):
                    break
                msgs_continue, hidden_user = build_messages_for_continue(
                    temp_msgs,
                    temp_ui,
                    persona_text,
                    history_text,
                    keywords_for_prompt,
                )
                more = _sample_reply(msgs_continue, temp_ui)
                if not more or violates_rules(more) or not more.strip():
                    break
                if len(more.strip()) < 6 and len(current_reply.strip()) > MIN_REPLY_CHARS:
                    break
                if not current_reply.endswith("\n") and current_reply:
                    current_reply = current_reply + "\n"
                current_reply = current_reply + more
                temp_ui[-1][1] = current_reply
                temp_msgs.append({"role": "assistant", "content": more})

        full_reply = temp_ui[-1][1]
        reply = extract_first_sentence(full_reply)
        temp_ui[-1][1] = reply

        last_user_idx = None
        for idx in range(len(temp_msgs) - 1, -1, -1):
            if temp_msgs[idx].get("role") == "user":
                last_user_idx = idx
                break
        if last_user_idx is not None and last_user_idx + 1 < len(temp_msgs):
            temp_msgs = temp_msgs[:last_user_idx + 2]
            temp_msgs[-1]["content"] = reply

        ui_hist = temp_ui
        msg_hist = temp_msgs
        if len(msg_hist) > 13:
            msg_hist = msg_hist[:1] + msg_hist[-12:]

        if mem_en_flag:
            mem_obj["turns"].append({"u": user_input, "a": reply})
            refresh_memory_layers(mem_obj)
            save_memory(mem_obj)

        return ui_hist, msg_hist, mem_obj, dyn_state

    send.click(
        _send,
        inputs=[txt, chat, state_msgs, state_mem, state_dynamic,
                use_rag, mem_enabled,
                temperature, top_p, typical_slider, repetition_penalty, no_repeat_ngram,
                diverse, k_candidates, max_new, refine_enabled],
        outputs=[chat, state_msgs, state_mem, state_dynamic]
    ).then(lambda: "", None, txt)

    txt.submit(
        _send,
        inputs=[txt, chat, state_msgs, state_mem, state_dynamic,
                use_rag, mem_enabled,
                temperature, top_p, typical_slider, repetition_penalty, no_repeat_ngram,
                diverse, k_candidates, max_new, refine_enabled],
        outputs=[chat, state_msgs, state_mem, state_dynamic]
    ).then(lambda: "", None, txt)

    # === 继续说（智能续写；不显示“继续”的用户消息；直接拼到上一条助手回复） ===
    def _continue(ui_hist, msg_hist, mem_obj, dyn_state,
                  use_rag_flag, mem_en_flag,
                  temp, topp, typical_val, rep_pen, ngram, diverse_flag, k_cand, max_tokens, refine_flag):

        if not ui_hist or not isinstance(ui_hist[-1], (list, tuple)) or not ui_hist[-1][1]:
            return ui_hist, msg_hist, mem_obj, ensure_dynamic_state(dyn_state)

        mem_obj = mem_obj or new_memory(DEFAULT_MEM_ID)
        dyn_state = ensure_dynamic_state(dyn_state)

        persona_text = build_persona_block()
        history_text = build_memory_context(mem_obj) if mem_obj.get("turns") else ""
        keywords_for_prompt = collect_dynamic_keywords(dyn_state) if use_rag_flag else []

        last_user = ""
        for m in reversed(msg_hist or []):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break
        if any(kw in (last_user or "") for kw in ACCOUNTING_KEYWORDS):
            keywords_for_prompt = [ACCOUNTING_ALERT] + keywords_for_prompt

        msgs, hidden_user = build_messages_for_continue(
            msg_hist or [{"role": "system", "content": "(hidden)"}],
            ui_hist or [],
            persona_text,
            history_text,
            keywords_for_prompt,
        )
        max_new = int(max_tokens)
        min_new = min(DEFAULT_MIN_NEW_TOKENS, max(1, max_new - 1)) if max_new > 1 else 1
        gen_kwargs = dict(
            max_new_tokens=max_new,
            temperature=float(temp),
            top_p=float(topp),
            typical_p=float(typical_val),
            repetition_penalty=float(rep_pen),
            no_repeat_ngram_size=int(ngram),
            length_penalty=DEFAULT_LENGTH_PENALTY,
            min_new_tokens=min_new,
        )
        if diverse_flag and int(k_cand) > 1:
            more = generate_diverse(msgs, int(k_cand), ui_hist or [], **gen_kwargs)
            if not more:
                more = generate_one(msgs, **gen_kwargs)
        else:
            more = generate_one(msgs, **gen_kwargs)

        if refine_flag:
            more = refine_short(more, tokenizer, model)

        # 1) UI：把新内容接到上一条助手消息上
        prev_u, prev_a = ui_hist[-1]
        combined = prev_a + ("\n" if prev_a and not prev_a.endswith("\n") else "") + more
        truncated = extract_first_sentence(combined)
        ui_hist[-1] = [prev_u, truncated]

        # 2) 历史消息：仅保留截断后的助手消息
        msg_hist = msg_hist or [{"role": "system", "content": "(hidden)"}]
        if msg_hist and msg_hist[-1].get("role") == "assistant":
            msg_hist[-1]["content"] = truncated
        if len(msg_hist) > 13:
            msg_hist = msg_hist[:1] + msg_hist[-12:]

        # 3) 会话记忆：保持仅存第一句
        if mem_en_flag:
            mem_obj = mem_obj or new_memory(DEFAULT_MEM_ID)
            if mem_obj.get("turns"):
                existing = mem_obj["turns"][-1]["a"] or ""
                combined_mem = existing + ("\n" if existing else "") + more
                mem_obj["turns"][-1]["a"] = extract_first_sentence(combined_mem)
            else:
                mem_obj["turns"].append({"u": "", "a": extract_first_sentence(more)})
            refresh_memory_layers(mem_obj)
            save_memory(mem_obj)

        return ui_hist, msg_hist, mem_obj, dyn_state

    btn_continue.click(
        _continue,
        inputs=[chat, state_msgs, state_mem, state_dynamic,
                use_rag, mem_enabled,
                temperature, top_p, typical_slider, repetition_penalty, no_repeat_ngram,
                diverse, k_candidates, max_new, refine_enabled],
        outputs=[chat, state_msgs, state_mem, state_dynamic]
    )

    def _repeat(ui_hist, msg_hist, mem_obj, dyn_state):
        dyn_state = ensure_dynamic_state(dyn_state)
        if not msg_hist:
            return ui_hist, msg_hist, mem_obj, dyn_state
        last_assistant = None
        for m in reversed(msg_hist):
            if m.get("role") == "assistant":
                last_assistant = m.get("content", "")
                break
        if not last_assistant:
            return ui_hist, msg_hist, mem_obj, dyn_state

        ui_hist_next = list(ui_hist or [])
        ui_hist_next.append(["（重新说）", last_assistant])

        msg_hist_next = list(msg_hist or [{"role": "system", "content": "(hidden)"}])
        msg_hist_next.append({"role": "assistant", "content": last_assistant})
        if len(msg_hist_next) > 13:
            msg_hist_next = msg_hist_next[:1] + msg_hist_next[-12:]

        return ui_hist_next, msg_hist_next, mem_obj, dyn_state

    btn_repeat.click(
        _repeat,
        inputs=[chat, state_msgs, state_mem, state_dynamic],
        outputs=[chat, state_msgs, state_mem, state_dynamic]
    )

    # === 重载关键术语记忆 ===
    def reload_hard():
        keyword_memory.load()
        gr.Info("关键术语记忆已重载")
        return None
    gr.Button("重载关键术语", variant="secondary").click(reload_hard, outputs=[])

if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=False,
        show_error=True
    )
