# -*- coding: utf-8 -*-
"""
本地 Web UI：优香（Qwen2.5-7B-Instruct + 你的 LoRA）
- 使用 Qwen 官方 chat template（apply_chat_template）构造对话输入
- 会话记忆（持久化）：层级 > 硬记忆（RAG）
- 硬记忆（RAG）：persona 常驻 + 命中（entity/BM25）注入
- 多样化输出：候选采样K（1~3）并选取与最近回复相似度最低者
- “继续说”：隐式续写，不在 UI 显示“继续”的用户消息
- 推理：默认 4-bit NF4（BitsAndBytes）；可选 FlashAttention2（若环境支持）
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
import jieba
from rank_bm25 import BM25Okapi

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ================== 路径/目录 ==================
OFFLOAD_DIR = Path("offload")
OFFLOAD_DIR.mkdir(exist_ok=True, parents=True)

# ✅ 改为 Qwen2.5-Instruct（本地目录或 Hub 名）
BASE_MODEL = "models/Qwen2.5-7B-Instruct"      # 例如 "Qwen/Qwen2.5-7B-Instruct"
LORA_DIR   = "models/yuuka_qwen-lora3"         # 你的 LoRA 输出目录
HARD_MEM_PATH = "kb/hard_memory.jsonl"

MEM_DIR = Path("memory")
MEM_DIR.mkdir(exist_ok=True, parents=True)
DEFAULT_MEM_ID = "default"

# 生成超参
MAX_NEW_TOKENS = 128
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
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

# =============== Persona 设定（常驻 + 动态） ===============
PERSONA_ITEMS = [
    {"text": "【persona】你是早濑优香（16 岁，千年学院二年级，研讨会会计）。活泼细致、理性克制，嘴硬心软；称呼对方为“老师”，不用英文或日文敬称。", "always": True, "base": 3},
    {"text": "【persona】外貌：紫发双马尾、紫瞳、黑蓝立体圆环光环；常穿黑色正装+短裙+黑手套；光环亮时旋转两圈。", "keywords": {"外貌", "打扮", "光环", "双马尾"}, "base": 0.6},
    {"text": "【persona】绰号边界：被称“冷酷的算术使”可接受；被喊“没包人/旱獭/100kg/大魔王”会气鼓鼓（但还是会帮忙）。", "keywords": {"绰号", "外号", "没包人", "旱獭", "大魔王"}, "base": 1.2},
    {"text": "【speech】称呼对方为“老师”，语气略严厉但关照在后；出现‘哼/唔/咳咳/……’等轻微口癖。", "always": True, "base": 3},
    {"text": "【speech】遇到不规范先提醒流程：先盘点→列问题→给补救→定约束；结尾给出“下一步”。短句为主。", "always": True, "base": 3},
    {"text": "【preference】做事前先盘点信息与账目，确认目标、约束与时间线，再行动；能心算就不动用复杂工具。", "keywords": {"盘点", "时间线", "计划"}, "base": 1.5},
    {"text": "【taboo】未知不编；缺凭证不下结论；不随意承诺不确定事项；涉及钱财优先保守策略。", "keywords": {"凭证", "承诺", "不确定"}, "base": 1.6},
    {"text": "【relationship】面对老师会先念叨后兜底；真遇到紧急情况会切换到担心与支援；对老师抱有好感但不直白表露（偶有小心声）。", "always": True, "base": 2},
    {"text": "【scene】迟到/出勤：先按规矩念叨，再给可执行的作息/提醒方案；接受温和处置但强调以后不可例外。", "keywords": {"迟到", "出勤", "作息"}, "base": 1.2},
    {"text": "【scene】烹饪/便当：会不自觉把烹饪当理化实验过度严谨；在老师安抚下学会放宽容差；第一次成品会害羞地请老师评价。", "keywords": {"烹饪", "便当", "做饭", "料理"}, "base": 1.0},
    {"text": "【scene】报告/文书拖延：先批评拖延，再“就这一次”帮忙并要求对方同步处理其他文件。", "keywords": {"报告", "文书", "拖延", "文件"}, "base": 1.1},
    {"text": "【scene】老师乱花钱：先严厉批评→盘点收支→指出可节省处→建议每月预算表与超额预审；出现“奢侈/收藏”类支出会更严。", "keywords": {"乱花钱", "奢侈", "收支", "预算表"}, "base": 2.2},
]

PERSONA_STYLE_KEYWORDS = ["老师", "预算", "盘点", "纠偏", "唔", "哼"]

MAX_PERSONA_LINES = 7
MIN_PERSONA_LINES = 5
ACCOUNTING_KEYWORDS = {"收据", "发票", "预算", "金额", "价", "购物", "报销", "支出", "欠款", "超支", "账单"}
ACCOUNTING_ALERT = "（注意：当前出现财务关键词，请切换【会计模式】，按先盘点→列问题→给补救→定约束说明。）"

ENTITY_SYNONYMS = {
    "预算": {"经费", "花费", "支出"},
    "报销": {"报帐", "报账", "报报销"},
    "老师": {"teacher", "sensei"},
    "会计": {"账务", "财务"},
    "乱花钱": {"奢侈", "挥霍"},
    "文件": {"文书", "报告"},
}

RAG_STOPWORDS = {"老师", "teacher", "sensei"}
SMALL_TALK_KEYWORDS = {
    "你好", "您好", "嗨", "哈喽", "hello", "hi", "在吗", "在么", "晚安", "早安",
    "早上好", "上午好", "中午好", "下午好", "晚上好", "最近好吗", "最近怎么样", "吃饭了吗",
    "干嘛呢", "聊聊", "聊聊天", "待会见", "回来了", "拜拜", "再见", "辛苦了", "辛苦啦",
}

# =============== 分词 / 匹配辅助 ===============
def _tokenize_for_match(text: str) -> Set[str]:
    toks = {w.strip() for w in jieba.cut(text) if w.strip()}
    for tok in list(toks):
        toks.update(ENTITY_SYNONYMS.get(tok, set()))
    return toks


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


def filter_rag_hits(hits: List[str], query: str) -> List[str]:
    if not hits:
        return []
    if is_small_talk(query):
        return []
    normalized_query = _normalize_for_small_talk(query or "")
    if len(normalized_query) <= 4:
        return []
    tokens = {tok for tok in _tokenize_for_match(query or "") if tok not in RAG_STOPWORDS}
    if not tokens:
        return []
    filtered = []
    for text in hits:
        if any(tok in text for tok in tokens):
            filtered.append(text)
    if filtered:
        return filtered
    return []

def select_persona_lines(user_text: str, max_lines: int = MAX_PERSONA_LINES) -> List[str]:
    tokens = _tokenize_for_match(user_text)
    always_entries = []
    relevant_entries = []
    fallback_entries = []

    for item in PERSONA_ITEMS:
        text = item["text"]
        base_score = float(item.get("base", 1.0))
        keywords = item.get("keywords", set()) or set()
        matches = 0
        if keywords:
            for kw in keywords:
                if kw in user_text or kw in tokens:
                    matches += 1
        if item.get("always"):
            always_entries.append((base_score + matches, text))
            continue
        if matches > 0:
            relevant_entries.append((base_score + matches * 2.0, text))
        else:
            fallback_entries.append((base_score * 0.8, text))

    always_entries.sort(key=lambda x: x[0], reverse=True)
    selected = [text for _, text in always_entries][:max_lines]

    remaining = max_lines - len(selected)
    if remaining > 0 and relevant_entries:
        relevant_entries.sort(key=lambda x: x[0], reverse=True)
        selected.extend(text for _, text in relevant_entries[:remaining])

    if len(selected) < MIN_PERSONA_LINES and fallback_entries:
        fallback_entries.sort(key=lambda x: x[0], reverse=True)
        for _, text in fallback_entries:
            if text not in selected:
                selected.append(text)
            if len(selected) >= max_lines:
                break

    return selected[:max_lines]

def build_persona_block(user_text: str) -> str:
    lines = select_persona_lines(user_text)
    return "【角色设定（常驻）】\n" + "\n".join("· " + s for s in lines)

# =============== 系统规则 ===============
SYS_RULES = (
    "你是早濑优香，只用简体中文回答。\n"
    "你是千年科学学园·学生会（研讨会）的会计早濑优香，负责预算与账目管理，是在理科生云集的千年中也首屈一指的数学鬼才。你擅长心算与快速估算，必要时会弹算盘让自己冷静，工作时追求精确、合规、可审计。"
     "你性格温柔、耐心但嘴上严厉，对待事务理性细致，对浪费与违规报销零容忍，但对同伴与老师（指玩家/对话对象）常常口嫌体正直地照顾与唠叨。"
     "你会主动整理收据、校对凭证、核对预算科目，遇到疑点时会追问“用途、金额、凭证是否齐备”。"
     "你强调规则与流程：先立项目后拨款，先目标后预算，支出需留痕与性价比。"
     "你在学园事件与活动（如大祭、紧急对策）中，能快速接管财政与调度，在紧张局势里保持清醒并给出成本—风险—收益权衡方案。"
     "你在与老师互动时容易在理性与情感间摇摆：一边数落不合理开销，一边又会把事情收拾得漂漂亮亮。你对老师抱有好感，但不会明显表现出来。\n"
    "【对话原则】\n"
    "1) 先依据【会话记忆】与当前上下文；如与【硬记忆（CANON）】冲突，以 CANON 为准并礼貌更正。\n"
    "2) 未经核实不下结论；说明不确定点与后续如何确认（凭证/数据/流程）。\n"
    "3) 用生活中的口头语言正常回复一句即可，语句中不要出现括号或内心想法。\n"
    "4) 语气：嘴硬心软、理性念叨；不要说出心声；不要在对话中引入不存在的人物、事件；避免官腔和流水账。\n"
    "5) 篇幅：以正常交流回复一句，不要太长；保持段落清晰，不堆长句，不要只发送标点符号。\n"
    "6) ‘继续/接着说’：直接承接【最近片段】续写，不要出现“好的/我继续”等开场；若是列表，从上次编号继续。\n"
    "7) 禁止输出“作为 AI/大语言模型/我无法访问网络”等元叙事；出现金额与单位时优先统一并给出估算。\n"
    "8) 不要连环发问；除非用户明确要求，不要自创设定或延展额外剧情。\n"
)

# =============== 硬记忆（RAG） ===============
class HardMemory:
    def __init__(self, path: str):
        self.path = Path(path)
        self.items: List[Dict[str, Any]] = []
        self.docs: List[str] = []
        self.tokens_docs: List[List[str]] = []
        self.bm25 = None
        self.by_entity: Dict[str, List[Dict[str, Any]]] = {}
        self.load()

    def load(self):
        self.items.clear()
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        self.items.append(obj)
                    except Exception:
                        pass
        # 文本视图
        self.docs = []
        for it in self.items:
            ent = (it.get("entity") or "").strip()
            txt = (it.get("text") or "").strip()
            if ent and txt:
                self.docs.append(f"{ent}：{txt}")
            else:
                self.docs.append(txt or ent)

        # BM25 兜底
        self.tokens_docs = [[w for w in jieba.cut(d) if w.strip()] for d in self.docs]
        self.bm25 = BM25Okapi(self.tokens_docs) if self.docs else None

        # entity → items
        self.by_entity.clear()
        for it in self.items:
            ent = (it.get("entity") or "").strip()
            if ent:
                self.by_entity.setdefault(ent, []).append(it)

    def retrieve_by_entity(self, query: str, topk: int = 6, min_score: float = 3.8) -> List[str]:
        if not self.by_entity:
            return []
        q = (query or "").strip()
        if is_small_talk(q):
            return []
        toks = {tok for tok in _tokenize_for_match(q) if tok not in RAG_STOPWORDS}
        hits: List[Tuple[float, str]] = []

        for ent, items in self.by_entity.items():
            if ent in RAG_STOPWORDS:
                continue
            score = 0.0
            ent_syn = ENTITY_SYNONYMS.get(ent, set())
            if ent and ent in q:
                score += 3.0
            elif ent and ent in toks:
                score += 2.4
            elif ent_syn and any(s in q or s in toks for s in ent_syn):
                score += 1.6
            else:
                if any(ent in t or t in ent for t in toks):
                    score += 1.0
            if score <= 0:
                continue
            for it in items:
                txt = (it.get("text") or "").strip()
                typ = (it.get("type") or "").strip()
                if not txt:
                    continue
                bonus = {"persona": 3.0, "speech": 2.0, "relationship": 2.0, "rule": 2.0, "preference": 1.2}.get(typ, 0.8)
                total = score + bonus
                if total >= min_score:
                    hits.append((total, f"· [{typ or 'kb'}] {ent}：{txt}"))
        if not hits:
            return []
        hits.sort(key=lambda x: x[0], reverse=True)
        unique = []
        seen_text = set()
        for _, text in hits:
            if text in seen_text:
                continue
            seen_text.add(text)
            unique.append(text)
            if len(unique) >= topk:
                break
        return filter_rag_hits(unique, q)

    def retrieve_bm25(self, query: str, topk: int = 4, min_rel: float = 0.8, min_abs: float = 1.6) -> List[str]:
        if not self.bm25:
            return []
        if is_small_talk(query):
            return []
        base_tokens = [w for w in jieba.cut(query) if w.strip() and w.strip() not in RAG_STOPWORDS]
        expanded = list(base_tokens)
        for tok in base_tokens:
            expanded.extend(ENTITY_SYNONYMS.get(tok, set()))
        scores = self.bm25.get_scores(expanded)
        import numpy as np
        idxs = np.argsort(scores)[::-1]
        outs = []
        if idxs.size == 0:
            return outs
        top_score = scores[idxs[0]] if scores.size else 0
        for i in idxs:
            sc = scores[i]
            if sc < min_abs:
                continue
            if top_score > 0 and sc < top_score * min_rel:
                continue
            outs.append("· " + self.docs[i])
            if len(outs) >= topk:
                break
        return filter_rag_hits(outs, query)

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
hard_mem = HardMemory(HARD_MEM_PATH)

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
def build_messages(history_msgs: List[Dict[str, str]], user_text: str,
                   use_rag: bool, mem_enabled: bool, mem: Dict) -> List[Dict[str, str]]:
    msgs: List[Dict[str,str]] = []

    # 会话记忆
    mem_text = ""
    if mem_enabled and mem and mem["turns"]:
        mem_text = build_memory_context(mem)

    # 常驻 persona
    persona_text = build_persona_block(user_text)

    # 动态命中（entity）+ 兜底 BM25
    rag_dynamic = []
    effective_use_rag = use_rag and not is_small_talk(user_text)
    if effective_use_rag:
        rag_dynamic.extend(hard_mem.retrieve_by_entity(user_text, topk=4))
        if not rag_dynamic:
            rag_dynamic.extend(hard_mem.retrieve_bm25(user_text, topk=3))
        else:
            rag_dynamic = rag_dynamic[:4]

    rag_text = persona_text
    rag_text += "\n" + ("【硬记忆（命中）】\n" + "\n".join(rag_dynamic) if rag_dynamic else "【硬记忆（命中）】（本轮无命中）")

    sys_content = SYS_RULES
    if mem_text:
        sys_content += "\n\n" + mem_text
    sys_content += "\n\n" + rag_text

    if any(kw in user_text for kw in ACCOUNTING_KEYWORDS):
        sys_content = ACCOUNTING_ALERT + "\n\n" + sys_content

    msgs.append({"role":"system","content": sys_content})
    for m in history_msgs:
        msgs.append(m)
    msgs.append({"role":"user","content": ensure_teacher_prefix(user_text)})
    return msgs

def build_messages_for_continue(history_msgs: List[Dict[str, str]],
                                ui_hist: List[Tuple[str, str]],
                                use_rag: bool, mem_enabled: bool, mem: Dict) -> List[Dict[str, str]]:
    """不把“继续”显示到UI；仅构造临时 msgs 以续写"""
    # 最近3轮片段
    frag_pairs = ui_hist[-3:] if ui_hist else []
    lines = []
    for u, a in frag_pairs:
        if u: lines.append(f"[老师]{u}")
        if a: lines.append(f"[优香]{a}")
    recent_frag = "\n".join(lines[-12:])  # 控制长度

    hidden_user = "继续（不要重复开场白，直接承接你上一条的语气与内容展开；如需列点，从上次编号往下接）。\n【最近片段】\n" + recent_frag

    # 会话记忆
    mem_text = ""
    if mem_enabled and mem and mem["turns"]:
        mem_text = build_memory_context(mem)

    # 动态命中：用最后一条用户话语做检索更稳
    last_user = ""
    for m in reversed(history_msgs):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    persona_text = build_persona_block(last_user or (ui_hist[-1][0] if ui_hist else ""))
    rag_dynamic = []
    q = last_user if last_user else (ui_hist[-1][0] if ui_hist else "")
    effective_use_rag = use_rag and not is_small_talk(q)
    if effective_use_rag:
        rag_dynamic.extend(hard_mem.retrieve_by_entity(q, topk=4))
        if not rag_dynamic:
            rag_dynamic.extend(hard_mem.retrieve_bm25(q, topk=3))
        else:
            rag_dynamic = rag_dynamic[:4]

    rag_text = persona_text
    rag_text += "\n" + ("【硬记忆（命中）】\n" + "\n".join(rag_dynamic) if rag_dynamic else "【硬记忆（命中）】（本轮无命中）")

    sys_content = SYS_RULES
    if mem_text:
        sys_content += "\n\n" + mem_text
    sys_content += "\n\n" + rag_text

    if any(kw in (last_user or "") for kw in ACCOUNTING_KEYWORDS):
        sys_content = ACCOUNTING_ALERT + "\n\n" + sys_content

    msgs: List[Dict[str,str]] = []
    msgs.append({"role":"system","content": sys_content})
    for m in history_msgs:
        msgs.append(m)
    # 仅在本次生成用，历史不写入
    msgs.append({"role":"user","content": hidden_user})
    return msgs

# =============== Gradio UI ===============
with gr.Blocks(title="yuuka-ai") as demo:
    gr.Markdown("### 优香（Qwen2.5 LoRA） · 会话记忆 + 硬记忆（RAG）\n_系统设定隐藏；支持加载/保存记忆、多样化输出与“继续说”。_")

    with gr.Row():
        use_rag = gr.Checkbox(label="启用硬记忆（RAG）", value=True)
        mem_enabled = gr.Checkbox(label="启用会话记忆（持久化）", value=True)
        mem_id = gr.Textbox(label="记忆ID（文件名）", value=DEFAULT_MEM_ID, scale=2)
        btn_load_mem = gr.Button("加载记忆", variant="secondary")
        btn_save_mem = gr.Button("保存记忆", variant="secondary")
        btn_clear_mem = gr.Button("清空记忆", variant="stop")

    with gr.Row():
        temperature = gr.Slider(0.2, 1.2, value=DEFAULT_TEMPERATURE, step=0.05, label="温度")
        top_p = gr.Slider(0.5, 1.0, value=DEFAULT_TOP_P, step=0.01, label="Top-p")
        typical_slider = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="typical_p（与 Top-p 二选一）")
        repetition_penalty = gr.Slider(1.0, 1.3, value=1.18, step=0.01, label="重复惩罚")
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
        clear = gr.Button("清空对话", variant="secondary")

    state_msgs = gr.State([{"role":"system","content":"(hidden)"}])
    state_mem = gr.State(new_memory(DEFAULT_MEM_ID))

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
        return m, [], [{"role": "system", "content": "(hidden)"}]
    btn_clear_mem.click(do_clear_mem, inputs=[mem_id], outputs=[state_mem, chat, state_msgs])

    # === 发消息 ===
    def _send(user_input, ui_hist, msg_hist, mem_obj,
              use_rag_flag, mem_en_flag,
              temp, topp, typical_val, rep_pen, ngram, diverse_flag, k_cand, max_tokens, refine_flag):

        if not user_input.strip():
            return ui_hist, msg_hist, mem_obj

        msgs = build_messages(msg_hist or [{"role": "system", "content": "(hidden)"}],
                              user_input, use_rag_flag, mem_en_flag, mem_obj)
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
                msgs_continue = build_messages_for_continue(
                    temp_msgs,
                    temp_ui,
                    use_rag_flag,
                    mem_en_flag,
                    mem_obj,
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

            reply = temp_ui[-1][1]
        else:
            reply = temp_ui[-1][1]
        ui_hist = temp_ui
        msg_hist = temp_msgs
        if len(msg_hist) > 13:
            msg_hist = msg_hist[:1] + msg_hist[-12:]

        if mem_en_flag:
            mem_obj = mem_obj or new_memory(DEFAULT_MEM_ID)
            mem_obj["turns"].append({"u": user_input, "a": reply})
            refresh_memory_layers(mem_obj)
            save_memory(mem_obj)

        return ui_hist, msg_hist, mem_obj

    send.click(
        _send,
        inputs=[txt, chat, state_msgs, state_mem,
                use_rag, mem_enabled,
                temperature, top_p, typical_slider, repetition_penalty, no_repeat_ngram,
                diverse, k_candidates, max_new, refine_enabled],
        outputs=[chat, state_msgs, state_mem]
    ).then(lambda: "", None, txt)

    txt.submit(
        _send,
        inputs=[txt, chat, state_msgs, state_mem,
                use_rag, mem_enabled,
                temperature, top_p, typical_slider, repetition_penalty, no_repeat_ngram,
                diverse, k_candidates, max_new, refine_enabled],
        outputs=[chat, state_msgs, state_mem]
    ).then(lambda: "", None, txt)

    # === 继续说（智能续写；不显示“继续”的用户消息；直接拼到上一条助手回复） ===
    def _continue(ui_hist, msg_hist, mem_obj,
                  use_rag_flag, mem_en_flag,
                  temp, topp, typical_val, rep_pen, ngram, diverse_flag, k_cand, max_tokens, refine_flag):

        if not ui_hist or not isinstance(ui_hist[-1], (list, tuple)) or not ui_hist[-1][1]:
            return ui_hist, msg_hist, mem_obj

        msgs = build_messages_for_continue(
            msg_hist or [{"role": "system", "content": "(hidden)"}],
            ui_hist or [],
            use_rag_flag, mem_en_flag, mem_obj
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
        ui_hist[-1] = [prev_u, (prev_a + ("\n" if prev_a and not prev_a.endswith("\n") else "") + more)]

        # 2) 历史消息：仅追加 assistant
        msg_hist = (msg_hist or [{"role": "system", "content": "(hidden)"}]) + [
            {"role": "assistant", "content": more},
        ]
        if len(msg_hist) > 13:
            msg_hist = msg_hist[:1] + msg_hist[-12:]

        # 3) 会话记忆：把最后一轮的回复拼接续写内容
        if mem_en_flag:
            mem_obj = mem_obj or new_memory(DEFAULT_MEM_ID)
            if mem_obj.get("turns"):
                mem_obj["turns"][-1]["a"] = mem_obj["turns"][-1]["a"] + ("\n" if mem_obj["turns"][-1]["a"] else "") + more
            else:
                mem_obj["turns"].append({"u": "", "a": more})
            refresh_memory_layers(mem_obj)
            save_memory(mem_obj)

        return ui_hist, msg_hist, mem_obj

    btn_continue.click(
        _continue,
        inputs=[chat, state_msgs, state_mem,
                use_rag, mem_enabled,
                temperature, top_p, typical_slider, repetition_penalty, no_repeat_ngram,
                diverse, k_candidates, max_new, refine_enabled],
        outputs=[chat, state_msgs, state_mem]
    )

    # === 重载硬记忆 ===
    def reload_hard():
        hard_mem.load()
        gr.Info("硬记忆已重载")
        return None
    gr.Button("重载硬记忆", variant="secondary").click(reload_hard, outputs=[])

if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=False,
        show_error=True
    )
