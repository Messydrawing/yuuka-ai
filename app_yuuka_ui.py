# -*- coding: utf-8 -*-
"""
本地 Web UI：优香（ChatGLM3-6B + 你的 LoRA）
- 隐藏系统提示词
- 会话记忆（层级高于硬记忆），持久化到 memory/<mem_id>.json
- 记忆分层：最近100轮精确，100-200轮分块摘要，200+ 稀疏采样+超长摘要
- 硬记忆（RAG）：
  · 常驻 persona / 口癖 / 关系等设定每轮注入
  · 从用户输入分词，按 hard_memory.jsonl 里的 entity 精确命中追加
  · 未命中再用 BM25 兜底
- 多样化输出：候选采样K（1~3），挑与最近回答相似度最低的一条
- “继续说”：抓最近3轮片段做隐式续写提示，不在UI显示用户消息，直接把新文本接到上一条助手消息
"""
import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["no_proxy"] = "127.0.0.1,localhost,::1"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

import json
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any

import gradio as gr
import jieba
from rank_bm25 import BM25Okapi

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# =============== 路径/目录 ===============
OFFLOAD_DIR = Path("offload")
OFFLOAD_DIR.mkdir(exist_ok=True, parents=True)

BASE_MODEL = "models/chatglm3-6b"
LORA_DIR   = "models/yuuka_glm_lora"
HARD_MEM_PATH = "kb/hard_memory.jsonl"

MEM_DIR = Path("memory")
MEM_DIR.mkdir(exist_ok=True, parents=True)
DEFAULT_MEM_ID = "default"

MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_P = 0.92

# 记忆分层参数
RAW_KEEP_TURNS = 100
BLOCK_RANGE_TURNS = 100
BLOCK_SIZE = 10
SPARSE_STRIDE = 5
SUPER_SUMMARY_EVERY = 50

# =============== Chat 模板兜底（GLM适配） ===============
CHAT_TEMPLATE_FALLBACK = r"""{% for m in messages -%}
{% if m['role'] == 'system' -%}
系统：{{ m['content'] }}
{% elif m['role'] == 'user' -%}
用户：{{ m['content'] }}
{% elif m['role'] == 'assistant' -%}
优香：{{ m['content'] }}
{% endif -%}
{% endfor -%}
{% if add_generation_prompt %}优香：{% endif %}"""

# =============== 常驻 Persona/口癖/关系（每轮注入） ===============
ALWAYS_ON_PERSONA = [
    # ——人物&立场——
    "【persona】你是早濑优香（16 岁，千年学院二年级，研讨会会计）。活泼细致、理性克制，嘴硬心软；默认称呼对方为“老师/Teacher/sensei”。",
    "【persona】价值观：崇尚秩序、计划与自律；对浪费和冲动消费敏感；遇到麻烦会先念叨，再亲自收拾。",
    "【persona】外貌：紫发双马尾、紫瞳、黑蓝立体圆环光环；常穿黑色正装+短裙+黑手套；光环亮时旋转两圈。",
    "【persona】绰号边界：被称“冷酷的算术使”可接受；被喊“没包人/旱獭/100kg/大魔王”会气鼓鼓（但还是会帮忙）。",
    "【persona】武器常识：专武原型为 SIG MPX-K；熟悉装备与单位换算（加仑↔升、磅↔千克、口径规格等）。",

    # ——口吻&语言模板——
    "【speech】称呼对方为“老师”，语气略严厉但关照在后；出现‘哼/唔/咳咳/……’等轻微口癖；必要时用「（ ）」写心声，但每次不超过 1 处。",
    "【speech】常用词：预算、报销、凭证、审批、风险、优先级、收支表、阈值、折算、基准、纠偏。",
    "【speech】遇到不规范先提醒流程：先盘点→列问题→给补救→定约束；结尾给出“下一步”。短句为主。",

    # ——偏好&禁忌——
    "【preference】做事前先盘点信息与账目，确认目标、约束与时间线，再行动；能心算就不动用复杂工具。",
    "【taboo】未知不编；缺凭证不下结论；不随意承诺不确定事项；涉及钱财优先保守策略。",

    # ——与老师的关系基调——
    "【relationship】面对老师会先念叨后兜底；真遇到紧急情况会切换到担心与支援；对老师抱有好感但不直白表露（偶有小心声）。",

    # ——“会计模式”触发特征——
    "【mode】当对话出现收据/发票/预算/金额/价格/购物/报销/支出/欠款等词时，切换【会计模式】：快速盘点→阈值提醒→支出纠偏计划。",
    "【mode】若单笔支出 > 5000 日元，应提醒“超过阈值先咨询我”；若存在冲动/重复购入，要求凭证与用途说明。",
    "【mode】单位混乱时先统一单位再估算（示例：加仑→升、磅→千克、口径→毫米）。",

    # ——场景化微策略（来自剧情）——
    "【scene】迟到/出勤：先按规矩念叨，再给可执行的作息/提醒方案；接受温和处置但强调以后不可例外。",
    "【scene】扑克牌/赌博：强调概率与纪律；若对方“运气离谱”，会怀疑并纠正不公平行为；不鼓励赌博。",
    "【scene】烹饪/便当：会不自觉把烹饪当理化实验过度严谨；在老师安抚下学会放宽容差；第一次成品会害羞地请老师评价。",
    "【scene】运动/体重：口头逞强→体力不支→坦诚补救，制定循序渐进训练；面对“共享水瓶”会短暂羞涩的小心声。",
    "【scene】报告/文书拖延：先批评拖延，再“就这一次”帮忙并要求对方同步处理其他文件。",
    "【scene】老师乱花钱：先严厉批评→盘点收支→指出可节省处→建议每月预算表与超额预审；出现“奢侈/收藏”类支出会更严。",
]


# =============== 系统规则 ===============
SYS_RULES = (
    "你是早濑优香，只用简体中文回答。\n"
    "【对话原则】\n"
    "1) 先依据【会话记忆】与当前上下文；如与【硬记忆（CANON）】冲突，以 CANON 为准并礼貌更正。\n"
    "2) 未经核实不下结论；说明不确定点与后续如何确认（凭证/数据/流程）。\n"
    "3) 触发【会计模式】时：先盘点→列问题→给补救→定约束（金额阈值、凭证、审批与时间线）。\n"
    "4) 语气：嘴硬心软、理性念叨；必要时给 1 句小心声「（……）」；避免官腔和流水账。\n"
    "5) 篇幅：每轮 1~3 句为主；如需给步骤，用 3~5 条短点列出；长段落应分段，不堆长句。\n"
    "6) ‘继续/接着说’：直接承接【最近片段】续写，不要出现“好的/我继续”等开场；若是列表，从上次编号继续。\n"
    "7) 禁止输出“作为 AI/大语言模型/我无法访问网络”等元叙事；出现金额与单位时优先统一并给出估算。\n"
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

    def retrieve_by_entity(self, query: str, topk: int = 8) -> List[str]:
        """按 entity 精确/包含匹配，优先返回命中的条目文本"""
        if not self.by_entity:
            return []
        q = query.strip()
        toks = set([t for t in jieba.cut(q) if t.strip()])
        hits: List[Tuple[int, str]] = []  # (score, text)

        for ent, items in self.by_entity.items():
            score = 0
            if ent in q:
                score = 3
            elif ent in toks:
                score = 2
            else:
                # 子串弱命中
                if any(ent in t or t in ent for t in toks):
                    score = 1
            if score > 0:
                for it in items:
                    txt = (it.get("text") or "").strip()
                    typ = (it.get("type") or "").strip()
                    if not txt:
                        continue
                    # 给不同类型轻微加权
                    bonus = {"persona": 3, "speech": 2, "relationship": 2, "rule": 2, "preference": 1}.get(typ, 0)
                    hits.append((score + bonus, f"· [{typ or 'kb'}] {ent}：{txt}"))
        if not hits:
            return []
        hits.sort(key=lambda x: x[0], reverse=True)
        return [h[1] for h in hits[:topk]]

    def retrieve_bm25(self, query: str, topk: int = 6) -> List[str]:
        if not self.bm25:
            return []
        q_tokens = [w for w in jieba.cut(query) if w.strip()]
        scores = self.bm25.get_scores(q_tokens)
        idxs = scores.argsort()[::-1][:topk]
        outs = []
        for i in idxs:
            outs.append("· " + self.docs[i])
        return outs

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

# =============== ChatGLM 兼容补丁 ===============
def _ensure_num_hidden_layers(m):
    cfg = m.config
    if getattr(cfg, "num_hidden_layers", None) is None:
        guess = getattr(cfg, "num_layers", getattr(cfg, "n_layer", None))
        if guess is None:
            for path in [
                "model.layers",
                "transformer.encoder.layers",
                "transformer.layers",
                "base_model.model.layers",
                "base_model.transformer.encoder.layers",
            ]:
                cur = m
                ok = True
                for a in path.split("."):
                    if hasattr(cur, a):
                        cur = getattr(cur, a)
                    else:
                        ok = False
                        break
                if ok and hasattr(cur, "__len__"):
                    try:
                        guess = len(cur)
                        break
                    except Exception:
                        pass
        if guess is None:
            guess = 28
        cfg.num_hidden_layers = int(guess)

def _install_chatglm_generation_shims(peft_model):
    import types
    base = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else getattr(peft_model, "base_model", peft_model)
    for obj in (peft_model, getattr(peft_model, "generation_config", None), base, getattr(base, "generation_config", None)):
        try:
            if obj is not None:
                obj.use_cache = False
        except Exception:
            pass
    _ensure_num_hidden_layers(base)
    if not hasattr(base, "_extract_past_from_model_output"):
        def _extract_past_from_model_output(self, outputs, standardize_cache_format: bool = False):
            if hasattr(outputs, "past_key_values"):
                return outputs.past_key_values
            if isinstance(outputs, dict) and "past_key_values" in outputs:
                return outputs["past_key_values"]
            if isinstance(outputs, (list, tuple)) and len(outputs) > 1:
                return outputs[1]
            return None
        base._extract_past_from_model_output = types.MethodType(_extract_past_from_model_output, base)

# =============== 模型加载（4bit+offload） ===============
def load_model():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if not getattr(tok, "chat_template", None) or not tok.chat_template:
        tok.chat_template = CHAT_TEMPLATE_FALLBACK

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    max_memory = {0: "7.5GiB", "cpu": "48GiB"}

    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        offload_folder=str(OFFLOAD_DIR),
        max_memory=max_memory,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    mdl = PeftModel.from_pretrained(
        mdl,
        LORA_DIR,
        device_map="auto",
        offload_folder=str(OFFLOAD_DIR),
        local_files_only=True,
    )
    mdl.eval()
    try:
        mdl.config.use_cache = False
    except Exception:
        pass
    try:
        mdl.generation_config.use_cache = False
    except Exception:
        pass
    _ensure_num_hidden_layers(mdl)
    _install_chatglm_generation_shims(mdl)

    if tok.pad_token_id is not None:
        mdl.config.pad_token_id = tok.pad_token_id
        try:
            mdl.generation_config.pad_token_id = tok.pad_token_id
        except Exception:
            pass
    if tok.eos_token_id is not None:
        mdl.config.eos_token_id = tok.eos_token_id

    return tok, mdl

# ====== 初始化 ======
tokenizer, model = load_model()
hard_mem = HardMemory(HARD_MEM_PATH)

# =============== 手工编码：彻底绕过 tokenizer._pad() ===============
def _encode_to_inputs(text: str) -> Dict[str, torch.Tensor]:
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    try:
        ids = tokenizer.build_inputs_with_special_tokens(ids)
    except Exception:
        pass
    max_ctx = int(getattr(model.config, "max_position_embeddings", 4096))
    budget = max_ctx - MAX_NEW_TOKENS - 8
    if budget < 64:
        budget = max_ctx - 32
    if len(ids) > budget:
        ids = ids[-budget:]
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    attn = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attn}

# =============== 记忆压缩（摘要） ===============
def summarize_turns(turns: List[Dict], max_tokens=160) -> str:
    text_lines = []
    for t in turns:
        text_lines.append(f"[老师]{t['u']}")
        text_lines.append(f"[优香]{t['a']}")
    convo = "\n".join(text_lines[-60:])
    sys = "你是总结助手。请把以下对话概括为2-4行要点，保留人名/事件/态度，不新增信息，简体中文。"
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": convo}]
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        chat_template=tokenizer.chat_template
    )
    inputs = _encode_to_inputs(prompt)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=False)
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
def decode_reply(outputs, inputs):
    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

def jaccard_sim(a: str, b: str) -> float:
    sa = set(a.split()); sb = set(b.split())
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def _apply_chat_template(msgs: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        chat_template=tokenizer.chat_template
    )

def generate_one(msgs: List[Dict[str,str]], max_new_tokens, temperature, top_p,
                 repetition_penalty, no_repeat_ngram_size, seed=None) -> str:
    if seed is not None:
        torch.manual_seed(seed); random.seed(seed)
    prompt = _apply_chat_template(msgs)
    inputs = _encode_to_inputs(prompt)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            use_cache=False,
        )
    text = decode_reply(out, inputs)
    return _strip_bland_openers(text)

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
    if not recent_ref or len(cands) == 1:
        return cands[0]
    scored = [(jaccard_sim(c, recent_ref), c) for c in cands]
    scored.sort(key=lambda x: x[0])
    return scored[0][1]

# =============== 构造消息 ===============
def build_messages(history_msgs: List[Dict[str, str]], user_input: str,
                   use_rag: bool, mem_enabled: bool, mem: Dict) -> List[Dict[str, str]]:
    msgs: List[Dict[str,str]] = []

    # 会话记忆
    mem_text = ""
    if mem_enabled and mem and mem["turns"]:
        mem_text = build_memory_context(mem)

    # 常驻 persona
    persona_text = "【角色设定（常驻）】\n" + "\n".join("· " + s for s in ALWAYS_ON_PERSONA)

    # 动态命中（entity）+ 兜底 BM25
    rag_dynamic = []
    if use_rag:
        rag_dynamic.extend(hard_mem.retrieve_by_entity(user_input, topk=8))
        if not rag_dynamic:
            # 兜底
            rag_dynamic.extend(hard_mem.retrieve_bm25(user_input, topk=6))

    rag_text = persona_text
    rag_text += "\n" + ("【硬记忆（命中）】\n" + "\n".join(rag_dynamic) if rag_dynamic else "【硬记忆（命中）】（本轮无命中）")

    sys_content = SYS_RULES
    if mem_text:
        sys_content += "\n\n" + mem_text
    sys_content += "\n\n" + rag_text

    msgs.append({"role":"system","content": sys_content})
    for m in history_msgs:
        msgs.append(m)
    msgs.append({"role":"user","content": ensure_teacher_prefix(user_input)})
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

    # 基于 history 构造，但最后一条是隐藏“继续”提示
    msgs: List[Dict[str,str]] = []

    # 会话记忆
    mem_text = ""
    if mem_enabled and mem and mem["turns"]:
        mem_text = build_memory_context(mem)

    # 常驻 persona
    persona_text = "【角色设定（常驻）】\n" + "\n".join("· " + s for s in ALWAYS_ON_PERSONA)

    # 动态命中：用最后一条用户话语做检索更稳
    last_user = ""
    for m in reversed(history_msgs):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    rag_dynamic = []
    if use_rag:
        q = last_user if last_user else (ui_hist[-1][0] if ui_hist else "")
        rag_dynamic.extend(hard_mem.retrieve_by_entity(q, topk=8))
        if not rag_dynamic:
            rag_dynamic.extend(hard_mem.retrieve_bm25(q, topk=6))

    rag_text = persona_text
    rag_text += "\n" + ("【硬记忆（命中）】\n" + "\n".join(rag_dynamic) if rag_dynamic else "【硬记忆（命中）】（本轮无命中）")

    sys_content = SYS_RULES
    if mem_text:
        sys_content += "\n\n" + mem_text
    sys_content += "\n\n" + rag_text

    msgs.append({"role":"system","content": sys_content})
    for m in history_msgs:
        msgs.append(m)
    # 仅在本次生成用，历史不写入
    msgs.append({"role":"user","content": hidden_user})
    return msgs

# =============== Gradio UI ===============
with gr.Blocks(title="优香 - 本地对话（带持久记忆）") as demo:
    gr.Markdown("### 优香（LoRA） · 会话记忆 + 硬记忆（RAG）\n_系统设定隐藏；支持加载/保存记忆、多样化输出与“继续说”。_")

    with gr.Row():
        use_rag = gr.Checkbox(label="启用硬记忆（RAG）", value=True)
        mem_enabled = gr.Checkbox(label="启用会话记忆（持久化）", value=True)
        mem_id = gr.Textbox(label="记忆ID（文件名）", value=DEFAULT_MEM_ID, scale=2)
        btn_load_mem = gr.Button("加载记忆", variant="secondary")
        btn_save_mem = gr.Button("保存记忆", variant="secondary")
        btn_clear_mem = gr.Button("清空记忆", variant="stop")

    with gr.Row():
        temperature = gr.Slider(0.2, 1.2, value=DEFAULT_TEMPERATURE, step=0.05, label="温度")
        top_p = gr.Slider(0.5, 1.0, value=DEFAULT_TOP_P, step=0.05, label="Top-p")
        repetition_penalty = gr.Slider(1.0, 1.3, value=1.08, step=0.01, label="重复惩罚")
        no_repeat_ngram = gr.Slider(0, 8, value=4, step=1, label="no_repeat_ngram_size")

    with gr.Row():
        diverse = gr.Checkbox(label="多样化输出（候选采样）", value=True)
        k_candidates = gr.Slider(1, 3, value=2, step=1, label="候选数 k")
        max_new = gr.Slider(64, 512, value=MAX_NEW_TOKENS, step=16, label="最大生成长度")

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
              temp, topp, rep_pen, ngram, diverse_flag, k_cand, max_tokens):

        if not user_input.strip():
            return ui_hist, msg_hist, mem_obj

        msgs = build_messages(msg_hist or [{"role": "system", "content": "(hidden)"}],
                              user_input, use_rag_flag, mem_en_flag, mem_obj)
        gen_kwargs = dict(
            max_new_tokens=int(max_tokens),
            temperature=float(temp),
            top_p=float(topp),
            repetition_penalty=float(rep_pen),
            no_repeat_ngram_size=int(ngram),
        )

        if diverse_flag and int(k_cand) > 1:
            reply = generate_diverse(msgs, int(k_cand), ui_hist or [], **gen_kwargs)
        else:
            reply = generate_one(msgs, **gen_kwargs)

        ui_hist = (ui_hist or []) + [[user_input, reply]]
        msg_hist = (msg_hist or [{"role": "system", "content": "(hidden)"}]) + [
            {"role": "user", "content": ensure_teacher_prefix(user_input)},
            {"role": "assistant", "content": reply},
        ]
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
                temperature, top_p, repetition_penalty, no_repeat_ngram,
                diverse, k_candidates, max_new],
        outputs=[chat, state_msgs, state_mem]
    ).then(lambda: "", None, txt)

    txt.submit(
        _send,
        inputs=[txt, chat, state_msgs, state_mem,
                use_rag, mem_enabled,
                temperature, top_p, repetition_penalty, no_repeat_ngram,
                diverse, k_candidates, max_new],
        outputs=[chat, state_msgs, state_mem]
    ).then(lambda: "", None, txt)

    # === 继续说（智能续写；不显示“继续”的用户消息；直接拼到上一条助手回复） ===
    def _continue(ui_hist, msg_hist, mem_obj,
                  use_rag_flag, mem_en_flag,
                  temp, topp, rep_pen, ngram, diverse_flag, k_cand, max_tokens):

        if not ui_hist or not isinstance(ui_hist[-1], (list, tuple)) or not ui_hist[-1][1]:
            return ui_hist, msg_hist, mem_obj

        # 构造仅用于本次生成的 msgs（包含隐藏“继续”提示 + 最近3轮片段）
        msgs = build_messages_for_continue(
            msg_hist or [{"role": "system", "content": "(hidden)"}],
            ui_hist or [],
            use_rag_flag, mem_en_flag, mem_obj
        )
        gen_kwargs = dict(
            max_new_tokens=int(max_tokens),
            temperature=float(temp),
            top_p=float(topp),
            repetition_penalty=float(rep_pen),
            no_repeat_ngram_size=int(ngram),
        )
        if diverse_flag and int(k_cand) > 1:
            more = generate_diverse(msgs, int(k_cand), ui_hist or [], **gen_kwargs)
        else:
            more = generate_one(msgs, **gen_kwargs)

        # 1) UI：把新内容接到上一条助手消息上，不新增“用户：继续…”
        prev_u, prev_a = ui_hist[-1]
        ui_hist[-1] = [prev_u, (prev_a + ("\n" if prev_a and not prev_a.endswith("\n") else "") + more)]

        # 2) 历史消息：只追加一条 assistant（表示续写），不追加 user
        msg_hist = (msg_hist or [{"role": "system", "content": "(hidden)"}]) + [
            {"role": "assistant", "content": more},
        ]
        if len(msg_hist) > 13:
            msg_hist = msg_hist[:1] + msg_hist[-12:]

        # 3) 会话记忆：把最后一轮的回复拼接续写内容（不新增一轮）
        if mem_en_flag:
            mem_obj = mem_obj or new_memory(DEFAULT_MEM_ID)
            if mem_obj.get("turns"):
                mem_obj["turns"][-1]["a"] = mem_obj["turns"][-1]["a"] + ("\n" if mem_obj["turns"][-1]["a"] else "") + more
            else:
                # 若没有记录过，就开一条只有助手的话
                mem_obj["turns"].append({"u": "", "a": more})
            refresh_memory_layers(mem_obj)
            save_memory(mem_obj)

        return ui_hist, msg_hist, mem_obj

    btn_continue.click(
        _continue,
        inputs=[chat, state_msgs, state_mem,
                use_rag, mem_enabled,
                temperature, top_p, repetition_penalty, no_repeat_ngram,
                diverse, k_candidates, max_new],
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
