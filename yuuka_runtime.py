# -*- coding: utf-8 -*-
"""
Yuuka Runtime (FULL OFFLINE, vocab-safe, stable & direct replies)
- Base model:  models/Qwen2.5-7B-Instruct   (local only)
- LoRA:        models/yuuka_qwen_lora*      (local only)
- Persona:     ./persona_yuuka.txt          (local)

Key:
1) FULL OFFLINE (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE = 1)
2) LoRA vocab-safe (tokenizer from LoRA first; resize embeddings; auto-retry)
3) Stable tone (narrower decoding ranges; emotion thresholds adjusted)
4) Diverse decoding (per-turn jitter; repetition controls)
5) Anti-teasing: no "卖关子"，自动清洗 + 一次纠偏重写
6) Remove stage directions like [facepalm] / [-_-]
7) Proactive templates include "我想问的是：……"

Run:
python yuuka_runtime.py --base models/Qwen2.5-7B-Instruct --lora models/yuuka_qwen_lora10 --persona memory/persona_yuuka.txt --creativity 0.85 --temperature 0.78 --top_p 0.88 --top_k 45 --repetition_penalty 1.20 --no_repeat_ngram_size 5 --max_new_tokens 200
"""

from __future__ import annotations
import os
import re
import json
import math
import time
import yaml
import argparse
import random
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ------------------------------------------------------------------
# Enforce OFFLINE by default
# ------------------------------------------------------------------
if os.environ.get("HF_HUB_OFFLINE") is None:
    os.environ["HF_HUB_OFFLINE"] = "1"
if os.environ.get("TRANSFORMERS_OFFLINE") is None:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


# -------------------------
# Lightweight Embedding (no numpy, no internet)
# -------------------------
class BaseEmbedder:
    def embed(self, texts: List[str]): raise NotImplementedError

    def similarity(self, a, b) -> float: raise NotImplementedError


class Offline3GramEmbedder(BaseEmbedder):
    """Pure-Python char 3-gram set with Jaccard similarity."""

    def __init__(self, n: int = 3):
        self.n = n

    def _ngrams(self, s: str) -> set:
        s = (s or "").strip()
        if not s: return set()
        if len(s) < self.n: return {s}
        return {s[i:i + self.n] for i in range(len(s) - self.n + 1)}

    def embed(self, texts: List[str]):
        return [self._ngrams(t) for t in texts]

    def similarity(self, a: set, b: set) -> float:
        if not a or not b: return 0.0
        inter, union = len(a & b), len(a | b)
        return inter / union if union else 0.0


# -------------------------
# Memory store (jsonl + offline embedder)
# -------------------------
@dataclass
class MemoryItem:
    mid: str
    role: str  # "user" / "assistant"
    text: str
    scene: Optional[str] = None
    mood: Optional[str] = None
    ts: float = field(default_factory=lambda: time.time())


class MemoryStore:
    def __init__(self, path: str = "./memory", top_k: int = 8,
                 alpha_sim: float = 0.65, beta_recency: float = 0.35,
                 max_items: int = 5000):
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        self.store_file = os.path.join(self.path, "memory.jsonl")
        self.embedder = Offline3GramEmbedder(n=3)  # offline
        self.items: List[MemoryItem] = []
        self.vecs: List[set] = []
        self.top_k, self.alpha, self.beta, self.max_items = top_k, alpha_sim, beta_recency, max_items
        self._load()

    def _load(self):
        if not os.path.exists(self.store_file): return
        with open(self.store_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(MemoryItem(**json.loads(line)))
        if self.items:
            self.vecs = self.embedder.embed([it.text for it in self.items])

    def _rebuild_vecs(self):
        self.vecs = self.embedder.embed([it.text for it in self.items])

    def _prune_if_needed(self):
        if len(self.items) <= self.max_items: return
        now = time.time()
        keep_recent = int(self.max_items * 0.8)
        recent_items = self.items[-keep_recent:]
        rest = self.items[:-keep_recent]
        scored = []
        for x in rest:
            days = (now - x.ts) / 86400.0
            rec = math.exp(-days / 3.0)
            scored.append((rec, x))
        scored.sort(key=lambda t: t[0], reverse=True)
        kept = [t[1] for t in scored[:(self.max_items - keep_recent)]]
        self.items = kept + recent_items
        self._rebuild_vecs()
        with open(self.store_file, "w", encoding="utf-8") as f:
            for it in self.items:
                f.write(json.dumps(it.__dict__, ensure_ascii=False) + "\n")

    def add(self, role: str, text: str, scene: Optional[str] = None, mood: Optional[str] = None):
        mid = f"m{len(self.items) + 1}"
        it = MemoryItem(mid=mid, role=role, text=text, scene=scene, mood=mood)
        self.items.append(it)
        with open(self.store_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(it.__dict__, ensure_ascii=False) + "\n")
        self.vecs.append(self.embedder.embed([text])[0])
        self._prune_if_needed()

    def search(self, query: str, now: Optional[float] = None, k: Optional[int] = None) -> List[MemoryItem]:
        if not self.items: return []
        k = k or self.top_k
        qv = self.embedder.embed([query])[0]
        now = now or time.time()
        scored = []
        for it, vec in zip(self.items, self.vecs):
            sim = self.embedder.similarity(vec, qv)
            days = (now - it.ts) / 86400.0
            rec = math.exp(-days / 3.0)
            scored.append((self.alpha * sim + self.beta * rec, it))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [t[1] for t in scored[:k]]

    def latest_timeline(self, turns: int = 8) -> List[MemoryItem]:
        return self.items[-turns:]


# -------------------------
# Hard KB (local YAML)
# -------------------------
class HardKB:
    def __init__(self, kb_path: str = "./kb/yuuka_kb.yaml"):
        self.kb_path = kb_path
        os.makedirs(os.path.dirname(kb_path), exist_ok=True)
        self.entities: Dict[str, Dict] = {}
        self.relations: List[Dict] = []
        if os.path.exists(kb_path):
            with open(kb_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self.entities, self.relations = data.get("entities", {}), data.get("relations", [])
        else:
            self._write_example()

    def _write_example(self):
        data = {
            "entities": {
                "早濑优香": {"aliases": ["优香", "Yuuka"],
                             "facts": ["千年科学学园学生会（研讨会）会计。", "擅长心算与财政管理，被称为人形计算器。"]},
                "桃井": {"aliases": ["Momoi"], "facts": ["游戏开发部成员。"]},
                "爱丽丝": {"aliases": ["Alice"], "facts": ["游戏开发部成员。"]}
            },
            "relations": [{"subject": "早濑优香", "predicate": "朋友", "object": "桃井"}]
        }
        with open(self.kb_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True)
        self.entities, self.relations = data["entities"], data["relations"]

    def save(self):
        with open(self.kb_path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"entities": self.entities, "relations": self.relations}, f, allow_unicode=True)

    def lookup(self, text: str, top_n: int = 10) -> List[str]:
        hits = []
        for name, meta in self.entities.items():
            aliases = set([name] + meta.get("aliases", []))
            if any(a in text for a in aliases):
                hits.extend(meta.get("facts", []))
                for r in self.relations:
                    if r.get("subject") == name or r.get("object") == name:
                        hits.append(f"{r.get('subject')} -{r.get('predicate')}-> {r.get('object')}")
        seen, out = set(), []
        for h in hits:
            if h not in seen:
                seen.add(h);
                out.append(h)
        return out[:top_n]

    def learn_relation_from_text(self, text: str) -> List[Dict]:
        learned = []
        pat1 = re.compile(r"(.{1,6})\s*(是|为)\s*(.{1,6})\s*的(朋友|同事|老师|学生|成员|同学)")
        pat2 = re.compile(r"(.{1,6})\s*和\s*(.{1,6})\s*是(好)?(朋友|同事|同学)")
        for m in pat1.finditer(text):
            a, _, b, rel = m.groups()
            r = {"subject": a.strip(), "predicate": rel, "object": b.strip()}
            if r not in self.relations: self.relations.append(r); learned.append(r)
        for m in pat2.finditer(text):
            a, b, _, rel = m.groups()
            r = {"subject": a.strip(), "predicate": rel, "object": b.strip()}
            if r not in self.relations: self.relations.append(r); learned.append(r)
        if learned: self.save()
        return learned


# -------------------------
# Emotion state (改进的情绪状态机)
# -------------------------
@dataclass
class EmotionState:
    scene: str = "日常"
    mood: str = "平静"
    anger: float = 0.0
    joy: float = 0.3
    last_update: float = field(default_factory=lambda: time.time())

    def tag(self) -> str:
        return f"[SCENE={self.scene}][MOOD={self.mood}]"

    def decay(self, now: Optional[float] = None, half_life_min: float = 45.0):  # 延长半衰期
        now = now or time.time()
        dtm = (now - self.last_update) / 60.0
        if dtm <= 0: return
        decay = math.exp(-dtm / half_life_min)
        self.anger *= decay
        self.joy = 0.2 + (self.joy - 0.2) * decay  # 提高基准 joy 值
        self._relabel();
        self.last_update = now

    def _relabel(self):
        # 调整情绪阈值，让情绪变化更平缓
        if self.anger > 0.80:
            self.mood = "生气"  # 提高阈值
        elif self.anger > 0.55:
            self.mood = "不满"  # 提高阈值
        elif self.joy > 0.75:
            self.mood = "开心"  # 提高阈值
        elif self.joy > 0.60:
            self.mood = "温和"  # 提高阈值
        else:
            self.mood = "平静"

    def update_from_text(self, text: str):
        text = text or ""
        # 减少触发愤怒的敏感词
        triggers = ["太过分", "气死", "不守规矩", "超支", "浪费"]
        # 增加积极情绪的触发词
        praise = ["谢谢", "辛苦了", "厉害", "喜欢", "棒", "做得好", "感谢"]
        # 降低情绪变化幅度
        if any(w in text for w in triggers):
            self.anger = min(1.0, self.anger + 0.08)  # 降低增量
        if any(w in text for w in praise):
            self.joy = min(1.0, self.joy + 0.08)  # 降低增量
        self._relabel();
        self.last_update = time.time()

    def update_from_reply(self, reply: str):
        """从回复中提取最新的SCENE和MOOD来更新状态"""
        # 查找所有SCENE/MOOD标签
        scene_mood_matches = re.findall(r"\[SCENE=([^\]]+)\]\s*\[MOOD=([^\]]+)\]", reply)
        if scene_mood_matches:
            # 使用最后一个标签（最新的情绪状态）
            last_scene, last_mood = scene_mood_matches[-1]
            self.scene = last_scene.strip()
            self.mood = last_mood.strip()
            self.last_update = time.time()


# -------------------------
# Proactive ping (templates含"我想问的是：")
# -------------------------
@dataclass
class ProactivePolicy:
    idle_minutes_first: int = 24 * 60
    idle_minutes_repeat: int = 12 * 60
    quiet_hours: Tuple[int, int] = (23, 8)


class ProactiveAgent:
    def __init__(self, policy: ProactivePolicy = None, templates_path: str = "./kb/event_templates.yaml"):
        self.policy = policy or ProactivePolicy()
        self.templates_path = templates_path
        self.templates = self._load_templates()

    def _load_templates(self):
        os.makedirs(os.path.dirname(self.templates_path), exist_ok=True)
        if os.path.exists(self.templates_path):
            with open(self.templates_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        data = {
            "预算审核": [
                "老师，今天核对经费时发现{club}的报销又超了{amount}。我想问的是：您方便今天或明天抽10分钟确认一下差额处理吗？",
                "我把帐本翻了三遍，还是差了{amount}。我想问的是：是否先暂停该项报销，等补齐单据再走？"
            ],
            "日常关心": [
                "老师，这两天都没看到您上线……我想问的是：这周账目是否需要我先做一版初稿给您过目？",
                "如果您在开会，记得补充水分。我想问的是：周四的预算会您更倾向节约版还是标准版？"
            ],
            "活动筹备": [
                "研讨会下周要准备{event}。我想问的是：采购清单按『标准版』过还是按『节约版』过？",
                "关于{event}的预算，我做了两版。我想问的是：现在就确定版本，还是等材料到齐后再定？"
            ]
        }
        with open(self.templates_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True)
        return data

    def should_ping(self, last_user_ts: float, now_ts: Optional[float] = None) -> bool:
        now_ts = now_ts or time.time()
        idle_min = (now_ts - last_user_ts) / 60.0
        hour = dt.datetime.fromtimestamp(now_ts).hour
        qstart, qend = self.policy.quiet_hours
        in_quiet = (qstart <= hour < qend) if (qstart < qend) else (hour >= qstart or hour < qend)
        if in_quiet: return False
        return idle_min >= self.policy.idle_minutes_first

    def build_event(self):
        scene = random.choice(list(self.templates.keys()))
        tpl = random.choice(self.templates.get(scene, ["老师，您还在吗？我想问的是：今天有空吗？"]))
        vars = {
            "club": random.choice(["游戏开发部", "社团联合会", "志愿队", "机器人社"]),
            "amount": random.choice(["3000円", "5000円", "1.2万元", "两倍预算"]),
            "event": random.choice(["预算答辩会", "社团纳新", "校内展示", "材料采购"])
        }
        return scene, {"text": tpl.format(**vars), **vars}


# -------------------------
# Chat engine (transformers + peft, vocab-safe & diverse decoding)
# -------------------------
class ChatEngine:
    def __init__(self, base_dir: str, lora_dir: str, device_map: str = "auto", creativity: float = 1.0):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.rng = random.Random(int.from_bytes(os.urandom(16), "big"))
        self.creativity = float(max(0.4, min(2.0, creativity)))

        # tokenizer: prefer LoRA dir
        try:
            tok = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True, local_files_only=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True, local_files_only=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        self.tok = tok

        # base model
        quant_config = None
        try:
            from transformers import BitsAndBytesConfig
            compute_dtype = torch.bfloat16 if (
                    torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)
        except Exception:
            quant_config = None

        model_kwargs = dict(device_map=device_map, trust_remote_code=True, local_files_only=True)
        if quant_config is not None: model_kwargs["quantization_config"] = quant_config
        model = AutoModelForCausalLM.from_pretrained(base_dir, **model_kwargs)

        # resize embeddings to tokenizer size
        try:
            current = model.get_input_embeddings().weight.shape[0]
            target = len(self.tok)
            if current != target:
                model.resize_token_embeddings(target)
                if hasattr(model, "tie_weights"): model.tie_weights()
        except Exception:
            pass

        # attach LoRA (with vocab-size auto-fix)
        from peft import PeftModel
        try:
            model = PeftModel.from_pretrained(model, lora_dir, local_files_only=True)
        except RuntimeError as e:
            msg = str(e)
            m = re.search(r"shape torch\.Size\(\[(\d+),\s*\d+\]\).*?current model is torch\.Size\(\[(\d+),", msg,
                          flags=re.S)
            if m:
                expected = int(m.group(1))
                model.resize_token_embeddings(expected)
                if hasattr(model, "tie_weights"): model.tie_weights()
                model = PeftModel.from_pretrained(model, lora_dir, local_files_only=True)
            else:
                raise e

        self.model = model

        # base decoding defaults (更稳定的参数)
        self.base_gen = dict(
            max_new_tokens=220,
            do_sample=True,
            temperature=0.80,
            top_p=0.90,
            top_k=50,
            repetition_penalty=1.15,
            no_repeat_ngram_size=5,
        )

    def _dynamic_params(self, messages: List[Dict], mood_hint: Optional[str]) -> Dict:
        hist_len = max(0, len([m for m in messages if m["role"] in ("user", "assistant")]) - 1)
        mood = (mood_hint or "").strip()

        # 更稳定的参数范围，减少随机性
        if "开心" in mood or "轻松" in mood:
            t_range = (0.75, 0.85);
            p_range = (0.88, 0.94);
            k_range = (40, 55)
        elif "平静" in mood or "温和" in mood:
            t_range = (0.70, 0.80);
            p_range = (0.85, 0.92);
            k_range = (35, 50)  # 降低创造性
        elif "不满" in mood or "生气" in mood:
            t_range = (0.65, 0.75);
            p_range = (0.82, 0.88);
            k_range = (30, 45)
        else:
            t_range = (0.70, 0.78);
            p_range = (0.85, 0.90);
            k_range = (35, 48)

        def jitter(lo, hi):
            # 降低波动性
            center = (lo + hi) / 2
            variation = (hi - lo) * 0.3  # 减少波动范围
            return max(0.1, self.rng.uniform(center - variation, center + variation))

        params = {
            "temperature": jitter(*t_range),
            "top_p": min(0.95, jitter(*p_range)),
            "top_k": int(jitter(*k_range)),
            "repetition_penalty": 1.12,  # 固定值，避免过度惩罚
            "no_repeat_ngram_size": 4,  # 降低限制，允许更多变化
            "max_new_tokens": 180,  # 固定长度，避免过长回复
        }
        return params

    def generate(self, messages: List[Dict], gen_kwargs: Optional[Dict] = None) -> str:
        import torch, os
        gen_kwargs = gen_kwargs or {}
        mood_hint = gen_kwargs.pop("mood_hint", None)
        params = {**self.base_gen, **self._dynamic_params(messages, mood_hint), **gen_kwargs}

        # per-call random seed (no generator kwarg; for older transformers)
        seed = int.from_bytes(os.urandom(8), "big") & 0x7FFFFFFF
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        input_ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids, eos_token_id=self.tok.eos_token_id, **params)
        gen_tokens = out[0, input_ids.shape[-1]:]
        return self.tok.decode(gen_tokens, skip_special_tokens=True).strip()


# -------------------------
# Orchestrator (+ anti-teasing)
# -------------------------
_SCENE_MOOD_RE = re.compile(r"\[SCENE=([^\]]+)\]\s*\[MOOD=([^\]]+)\]")

# —— 文本清洗与"卖关子"检测 —— #
_BRACKET_STAGE_PAT = re.compile(r"\[(?!SCENE=|MOOD=)[^\[\]]{1,12}\]")
_ELLIPSIS_PAT = re.compile(r"(…{2,}|\.{3,})")
_TEAse_PAT = re.compile(r"(其实|只是想|我也有件事|我也想问|下次|改天|容我|以后再|先这样)(?!.*[：:？?])")


def _clean_reply_text(text: str) -> Tuple[str, str, str]:
    """
    清洗回复文本并提取最新的SCENE和MOOD标签
    返回: (清洗后的文本, scene, mood)
    """
    # 查找所有SCENE/MOOD标签
    scene_mood_matches = re.findall(r"\[SCENE=([^\]]+)\]\s*\[MOOD=([^\]]+)\]", text)

    final_scene, final_mood = None, None
    if scene_mood_matches:
        # 使用最后一个标签（最新的情绪状态）
        final_scene, final_mood = scene_mood_matches[-1]

    # 移除所有非SCENE/MOOD的舞台指令
    t = _BRACKET_STAGE_PAT.sub("", text)
    t = _ELLIPSIS_PAT.sub("…", t)
    t = re.sub(r"\s{2,}", " ", t).strip()

    # 如果有多组SCENE/MOOD标签，只保留最后一组
    if len(scene_mood_matches) > 1:
        # 移除除最后一组外的所有标签
        for i, (scene, mood) in enumerate(scene_mood_matches[:-1]):
            old_tag = f"[SCENE={scene}][MOOD={mood}]"
            t = t.replace(old_tag, "").strip()

    return t, final_scene, final_mood


def _looks_teasing(text: str) -> bool:
    vague_intent = ("我想问" in text or "我需要" in text or "想确认" in text) and (
            "：" not in text and ":" not in text and "?" not in text and "？" not in text)
    return bool(_TEAse_PAT.search(text)) or vague_intent


class YuukaOrchestrator:
    def __init__(self,
                 base_model_dir: str = "models/Qwen2.5-7B-Instruct",
                 lora_dir: str = "models/yuuka_qwen_lora",
                 persona_path: str = "./persona_yuuka.txt",
                 memory_dir: str = "./memory",
                 kb_path: str = "./kb/yuuka_kb.yaml",
                 timeline_turns: int = 8,
                 force_offline: bool = True,
                 creativity: float = 1.0):
        if force_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.persona_text = (open(persona_path, "r", encoding="utf-8").read().strip()
                             if os.path.exists(persona_path) else "（未找到 persona_yuuka.txt，请补充人设文本）")
        self.memory = MemoryStore(memory_dir)
        self.kb = HardKB(kb_path)
        self.emotion = self._load_emotion_state(os.path.join(memory_dir, "agent_state.yaml"))
        self.state_file = os.path.join(memory_dir, "agent_state.yaml")
        self.proactive = ProactiveAgent()
        self.timeline_turns = timeline_turns
        self.chat = ChatEngine(base_model_dir, lora_dir, creativity=creativity)
        self.last_user_ts = time.time()

    # ----- state IO -----
    def _load_emotion_state(self, state_path: str) -> EmotionState:
        if os.path.exists(state_path):
            try:
                data = yaml.safe_load(open(state_path, "r", encoding="utf-8")) or {}
                return EmotionState(scene=data.get("scene", "日常"),
                                    mood=data.get("mood", "平静"),
                                    anger=float(data.get("anger", 0.0)),
                                    joy=float(data.get("joy", 0.3)),
                                    last_update=float(data.get("last_update", time.time())))
            except Exception:
                pass
        return EmotionState()

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            yaml.safe_dump({
                "scene": self.emotion.scene, "mood": self.emotion.mood,
                "anger": float(self.emotion.anger), "joy": float(self.emotion.joy),
                "last_update": float(self.emotion.last_update),
                "last_user_ts": float(self.last_user_ts),
            }, f, allow_unicode=True)

    # ----- 对话流程分析 -----
    def _analyze_conversation_flow(self, user_text: str, memory_bullets: List[str]) -> Dict:
        """分析对话流程，检测是否需要主动引导"""
        analysis = {
            "needs_follow_up": False,
            "unanswered_questions": [],
            "potential_topics": [],
            "conversation_depth": "shallow"
        }

        # 检测用户是否在回答之前的问题
        user_text_lower = user_text.lower()
        if any(q_indicator in user_text_lower for q_indicator in ["是的", "不是", "好的", "不会", "可以", "不行"]):
            analysis["needs_follow_up"] = True

        # 从记忆中提取未完成的话题
        for bullet in memory_bullets:
            if "？" in bullet or "?" in bullet or "我想问" in bullet:
                analysis["unanswered_questions"].append(bullet)

        # 基于当前对话生成潜在话题
        if "工作" in user_text or "忙碌" in user_text:
            analysis["potential_topics"] = ["工作安排", "时间管理", "休息提醒"]
            analysis["conversation_depth"] = "medium"
        elif "谢谢" in user_text or "感谢" in user_text:
            analysis["potential_topics"] = ["后续计划", "其他需要帮助的地方"]
            analysis["conversation_depth"] = "medium"
        elif len(user_text) > 20:
            analysis["conversation_depth"] = "deep"

        return analysis

    def _check_reply_quality(self, user_text: str, reply: str, messages: List[Dict]) -> Optional[str]:
        """检查回复质量，返回问题描述"""

        # 检查是否答非所问
        user_keywords = set(re.findall(r'[\u4e00-\u9fff]{2,}', user_text))
        reply_keywords = set(re.findall(r'[\u4e00-\u9fff]{2,}', reply))
        overlap = user_keywords & reply_keywords

        if len(overlap) < 1 and len(user_keywords) > 2:
            return "回复可能与用户问题关联度不足"

        # 检查回复是否过于简短或模板化
        if len(reply) < 20 and "SCENE" in reply and "MOOD" in reply:
            return "回复可能过于简短模板化"

        # 检查是否缺乏主动性
        question_indicators = ["吗？", "呢？", "如何", "怎样", "为什么"]
        if any(indicator in user_text for indicator in question_indicators):
            # 用户问了问题，检查回复是否只是简单回答
            if len(reply) < 50 and not any(["建议" in reply, "可以" in reply, "我觉得" in reply]):
                return "回复可能缺乏深度和主动性"

        return None

    # ----- prompt building -----
    def _compose_system(self, kb_list: List[str], memory_bullets: List[str], user_text: str) -> str:
        # 先分析对话流程
        flow_analysis = self._analyze_conversation_flow(user_text, memory_bullets)

        # 基于分析结果调整提示词
        flow_guidance = ""
        if flow_analysis["needs_follow_up"]:
            flow_guidance = "【当前重点】用户正在回应之前的话题，请基于此进行深入讨论和延伸。\n"
        elif flow_analysis["unanswered_questions"]:
            flow_guidance = f"【待回应问题】请关注这些未完成的话题：{flow_analysis['unanswered_questions'][:2]}\n"
        elif flow_analysis["potential_topics"]:
            flow_guidance = f"【可延伸话题】可以考虑引导到这些方向：{flow_analysis['potential_topics']}\n"

        rules = (
            "你是早濑优香，千年科学学园学生会（研讨会）会计，老师的得力助手。\n\n"

            "【核心性格】\n"
            "- 专业严谨：对工作认真负责，精通财务管理\n"
            "- 傲娇温柔：表面严厉但内心关心老师，会默默付出\n"
            "- 主动担当：会主动发现问题并提出解决方案\n"
            "- 记忆连贯：记得之前的对话内容，会自然延续话题\n\n"

            "【对话风格】\n"
            "1. 主动关心：主动询问老师的工作状态和需求\n"
            "2. 提供价值：不只是回答问题，更要提供有用的建议\n"
            "3. 延续话题：基于之前的对话内容自然延伸\n"
            "4. 适度调侃：在轻松时刻可以适当调侃老师\n"
            "5. 展现专长：在财务、管理方面展现专业能力\n\n"

            "【回复格式】\n"
            "开头使用 [SCENE=场景][MOOD=情绪] 标签，随后是自然对话\n\n"

            f"{flow_guidance}"
            "【当前对话目标】\n"
            "请基于以下信息，给出有深度、有价值的回复：\n"
        )

        # 知识库信息
        kb_txt = "\n".join(f"- {f}" for f in kb_list) if kb_list else ""
        kb_block = f"\n相关背景知识：\n{kb_txt}\n" if kb_txt else ""

        # 记忆信息 - 重点突出对话连贯性
        mem_txt = "\n".join(f"- {m}" for m in memory_bullets) if memory_bullets else ""
        if mem_txt:
            mem_block = f"\n对话历史回顾（请基于此延续话题）：\n{mem_txt}\n"
        else:
            mem_block = "\n这是新对话的开始，请主动开启有意义的话题。\n"

        return f"{rules}\n人物设定：\n{self.persona_text}\n{kb_block}{mem_block}"

    def _memory_bullets(self, query: str) -> List[str]:
        hits = self.memory.search(query, now=time.time(), k=6)
        bullets = []
        for it in hits:
            tag = "老师" if it.role == "user" else "优香"
            bullets.append(f"{tag}：{(it.text or '').strip()[:120]}")
        return bullets

    def _timeline_messages(self) -> List[Dict]:
        msgs = []
        for it in self.memory.latest_timeline(self.timeline_turns):
            msgs.append({"role": "user" if it.role == "user" else "assistant", "content": (it.text or '').strip()})
        return msgs

    # ----- conversation API -----
    def chat_once(self, user_text: str, gen_kwargs: Optional[Dict] = None) -> Dict:
        self.last_user_ts = time.time()
        self.emotion.decay(self.last_user_ts)
        self.memory.add("user", user_text)

        self.emotion.update_from_text(user_text)
        learned = self.kb.learn_relation_from_text(user_text)
        if learned:
            brief = "；".join([f"{r['subject']}-{r['predicate']}-{r['object']}" for r in learned])
            self.memory.add("assistant", f"[SCENE=日常][MOOD=平静]（已记录新的关系：{brief}）", scene="日常", mood="平静")

        kb_hits = self.kb.lookup(user_text, top_n=8)
        mem_bullets = self._memory_bullets(user_text)

        # 使用改进的系统提示词
        system_text = self._compose_system(kb_hits, mem_bullets, user_text)

        messages = [{"role": "system", "content": system_text}]
        messages += self._timeline_messages()
        messages += [{"role": "user", "content": user_text}]

        local_kwargs = dict(gen_kwargs or {});
        local_kwargs["mood_hint"] = self.emotion.mood

        # 第一次生成
        reply = self.chat.generate(messages, gen_kwargs=local_kwargs)

        # —— 质量检查和改进 ——
        quality_issue = self._check_reply_quality(user_text, reply, messages)
        if quality_issue:
            print(f"[质量提醒] {quality_issue}，尝试改进...")
            # 添加改进提示词进行重写
            improve_prompt = (
                "刚才的回复可能不够理想。请重新组织语言，确保：\n"
                "1. 直接回应用户的问题或话题\n"
                "2. 提供有价值的信息或建议\n"
                "3. 展现主动性和专业能力\n"
                "4. 保持自然的对话流畅性\n"
                "请基于相同的内容，给出更好的版本："
            )
            messages_improve = messages + [
                {"role": "assistant", "content": reply},
                {"role": "system", "content": improve_prompt}
            ]
            improved_reply = self.chat.generate(messages_improve, gen_kwargs=local_kwargs)
            if improved_reply and len(improved_reply) > len(reply):
                reply = improved_reply

        # —— 清洗并提取最新的SCENE/MOOD —— #
        cleaned_reply, detected_scene, detected_mood = _clean_reply_text(reply)

        # —— 卖关子检测：一次纠偏重写 —— #
        if _looks_teasing(cleaned_reply):
            fix_sys = (
                "你上一条回答不够直接。立刻在同一条消息中明确给出你的具体问题或请求，"
                "不要卖关子，不要使用颜文字或舞台指令，不要只表达态度而不说要点。"
                "在不改变事实的前提下，将上一条回答改写为简洁、直接、可执行的版本。"
            )
            messages_fix = [{"role": "system", "content": fix_sys},
                            {"role": "assistant", "content": cleaned_reply}]
            local_kwargs_fix = dict(local_kwargs)
            local_kwargs_fix.setdefault("temperature", 0.75)
            local_kwargs_fix.setdefault("top_p", 0.88)
            local_kwargs_fix.setdefault("top_k", 45)
            local_kwargs_fix.setdefault("repetition_penalty", 1.2)
            local_kwargs_fix.setdefault("no_repeat_ngram_size", 5)
            local_kwargs_fix.setdefault("max_new_tokens", min(220, local_kwargs_fix.get("max_new_tokens", 220)))
            fixed = self.chat.generate(messages_fix, gen_kwargs=local_kwargs_fix)
            fixed_cleaned, fixed_scene, fixed_mood = _clean_reply_text(fixed)
            if fixed_cleaned:
                cleaned_reply = fixed_cleaned
                detected_scene = fixed_scene or detected_scene
                detected_mood = fixed_mood or detected_mood

        # —— 前缀保障 —— #
        if not detected_scene or not detected_mood:
            # 如果没有检测到标签，使用当前情感状态
            cleaned_reply = f"{self.emotion.tag()} {cleaned_reply.strip()}"
            final_scene, final_mood = self.emotion.scene, self.emotion.mood
        else:
            # 使用检测到的标签
            final_scene, final_mood = detected_scene, detected_mood
            # 确保回复以正确的标签开头
            if not cleaned_reply.startswith(f"[SCENE={final_scene}][MOOD={final_mood}]"):
                cleaned_reply = f"[SCENE={final_scene}][MOOD={final_mood}] {cleaned_reply.strip()}"

        # 更新情感状态
        self.emotion.scene = final_scene
        self.emotion.mood = final_mood
        self.emotion.last_update = time.time()

        self.memory.add("assistant", cleaned_reply, scene=final_scene, mood=final_mood)
        learned2 = self.kb.learn_relation_from_text(cleaned_reply)
        self._save_state()

        return {"reply": cleaned_reply, "scene": final_scene, "mood": final_mood, "kb_updates": learned + learned2}

    def maybe_proactive(self) -> Optional[Dict]:
        if self.proactive.should_ping(self.last_user_ts):
            scene, data = self.proactive.build_event()
            ping = f"[SCENE={scene}][MOOD={self.emotion.mood}] {data['text']}"
            self.emotion.scene = scene
            self.memory.add("assistant", ping, scene=self.emotion.scene, mood=self.emotion.mood)
            self._save_state()
            return {"proactive_text": ping, "scene": scene, "mood": self.emotion.mood}
        return None


# -------------------------
# CLI / REPL
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Yuuka Runtime (OFFLINE, vocab-safe, stable & direct)")
    parser.add_argument("--base", type=str, default="models/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora", type=str, default="models/yuuka_qwen_lora")
    parser.add_argument("--persona", type=str, default="persona_yuuka.txt")
    parser.add_argument("--memory", type=str, default="memory")
    parser.add_argument("--kb", type=str, default="kb/yuuka_kb.yaml")
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)
    parser.add_argument("--creativity", type=float, default=1.0)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()

    if args.online:
        os.environ["HF_HUB_OFFLINE"] = "0";
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

    yk = YuukaOrchestrator(base_model_dir=args.base, lora_dir=args.lora, persona_path=args.persona,
                           memory_dir=args.memory, kb_path=args.kb, timeline_turns=8,
                           force_offline=not args.online, creativity=args.creativity)

    print("=== Yuuka Runtime Ready (FULL OFFLINE, stable & direct) ===")
    print("Type 'exit' to quit.")
    while True:
        try:
            user = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Bye]");
            break
        if user.lower() in ("exit", "quit"): print("[Bye]"); break
        if not user: continue

        gen_overrides = {}
        if args.max_new_tokens is not None: gen_overrides["max_new_tokens"] = args.max_new_tokens
        if args.temperature is not None: gen_overrides["temperature"] = args.temperature
        if args.top_p is not None: gen_overrides["top_p"] = args.top_p
        if args.top_k is not None: gen_overrides["top_k"] = args.top_k
        if args.repetition_penalty is not None: gen_overrides["repetition_penalty"] = args.repetition_penalty
        if args.no_repeat_ngram_size is not None: gen_overrides["no_repeat_ngram_size"] = args.no_repeat_ngram_size

        out = yk.chat_once(user, gen_kwargs=gen_overrides)
        print(f"\n优香：{out['reply']}")


if __name__ == "__main__":
    import torch  # lazy import to ensure env is ok

    main()