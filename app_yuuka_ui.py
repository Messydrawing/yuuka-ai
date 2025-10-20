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
import hashlib
import importlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Any, Set, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np  # type: ignore

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore

try:
    import faiss  # type: ignore
except ModuleNotFoundError:
    faiss = None  # type: ignore

GRADIO_AVAILABLE = True
try:
    import gradio as gr
except ModuleNotFoundError:
    if os.environ.get("YUKA_REQUIRE_GRADIO") == "1":
        raise
    GRADIO_AVAILABLE = False
    gr = None  # type: ignore

try:
    import torch
except ModuleNotFoundError:
    if os.environ.get("YUKA_REQUIRE_TORCH") == "1":
        raise
    torch = None  # type: ignore

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except ModuleNotFoundError:
    if os.environ.get("YUKA_REQUIRE_TRANSFORMERS") == "1":
        raise
    AutoTokenizer = AutoModelForCausalLM = BitsAndBytesConfig = None  # type: ignore

try:
    from peft import PeftModel
except ModuleNotFoundError:
    if os.environ.get("YUKA_REQUIRE_PEFT") == "1":
        raise
    PeftModel = None  # type: ignore

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
LONG_TERM_MEMORY_FILE = MEM_DIR / "hard_memory_runtime.jsonl"
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

# =============== 主动问候（空闲触发）配置 ===============
IDLE_MARKER_PREFIX = "\u200b:codex-terminal-citation[codex-terminal-citation]{"
IDLE_MARKER_SUFFIX = "】"
IDLE_MARKER_PATTERN = re.compile(r"^\u200b:codex-terminal-citation\\[codex-terminal-citation\\]\{(?P<meta>.+)】$")
IDLE_TRIGGER_CHECK_INTERVAL = 5.0
IDLE_TRIGGER_MIN_INTERVAL = 120.0
IDLE_TRIGGER_MAX_INTERVAL = 420.0
IDLE_TRIGGER_MAX_PER_DAY = 3


def _format_idle_duration(seconds: int) -> str:
    if seconds <= 0:
        return "片刻"
    if seconds < 60:
        return f"约 {seconds} 秒"
    minutes = seconds // 60
    if minutes < 60:
        return f"约 {minutes} 分钟"
    hours = minutes // 60
    return f"约 {hours} 小时"

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


def parse_idle_marker(text: str) -> Optional[Dict[str, str]]:
    if not text:
        return None
    match = IDLE_MARKER_PATTERN.match(text.strip())
    if not match:
        return None
    meta_str = match.group("meta")
    parts = [seg for seg in meta_str.split() if seg.strip()]
    result: Dict[str, str] = {}
    for seg in parts:
        if "=" not in seg:
            continue
        key, value = seg.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def build_idle_prompt(meta: Dict[str, str]) -> str:
    seconds_raw = meta.get("idle_seconds") if meta else None
    try:
        seconds_val = int(float(seconds_raw)) if seconds_raw is not None else None
    except (TypeError, ValueError):
        seconds_val = None
    duration_text = _format_idle_duration(seconds_val or 0)
    return (
        f"（系统主动问候）老师已经{duration_text}没有发言。"
        "请你主动、轻松地问候老师，询问是否需要帮助或分享一些轻松话题，但要保持简洁。"
    )


def prepare_user_event(user_text: str) -> Dict[str, Any]:
    idle_meta = parse_idle_marker(user_text)
    if idle_meta:
        prompt_text = build_idle_prompt(idle_meta)
        return {
            "prompt_text": prompt_text,
            "conversation_user_text": prompt_text,
            "ui_user_text": "（系统主动问候）",
            "memory_user_text": "",
            "is_idle": True,
            "idle_meta": idle_meta,
        }
    stripped = (user_text or "").strip()
    prefixed = ensure_teacher_prefix(stripped)
    return {
        "prompt_text": stripped,
        "conversation_user_text": prefixed,
        "ui_user_text": stripped,
        "memory_user_text": stripped,
        "is_idle": False,
        "idle_meta": None,
    }


def _idle_trigger(session_id: str, idle_seconds: float) -> Dict[str, Any]:
    seconds = int(max(0.0, float(idle_seconds)))
    meta = {
        "line_range_start": "770",
        "line_range_end": "820",
        "terminal_chunk_id": "IdleChat",
        "idle_seconds": str(seconds),
        "session": session_id,
    }
    marker = IDLE_MARKER_PREFIX + " ".join(f"{k}={v}" for k, v in meta.items()) + IDLE_MARKER_SUFFIX
    return {"marker": marker, "meta": meta, "is_idle": True}


class IdleManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._pending: Dict[str, List[Dict[str, Any]]] = {}
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _random_interval(self) -> float:
        return random.uniform(IDLE_TRIGGER_MIN_INTERVAL, IDLE_TRIGGER_MAX_INTERVAL)

    def _current_day(self):
        return datetime.now(timezone.utc).date()

    def _ensure_state(self, session_id: str) -> Dict[str, Any]:
        today = self._current_day()
        state = self._sessions.get(session_id)
        if not state:
            state = {
                "enabled": False,
                "last_input": time.time(),
                "next_deadline": None,
                "trigger_count": 0,
                "day": today,
            }
            self._sessions[session_id] = state
        elif state.get("day") != today:
            state["day"] = today
            state["trigger_count"] = 0
        return state

    def enable(self, session_id: str) -> None:
        with self._lock:
            state = self._ensure_state(session_id)
            state["enabled"] = True
            if state.get("next_deadline") is None:
                state["next_deadline"] = time.time() + self._random_interval()

    def disable(self, session_id: str) -> None:
        with self._lock:
            state = self._ensure_state(session_id)
            state["enabled"] = False
            state["next_deadline"] = None
            self._pending.pop(session_id, None)

    def mark_activity(self, session_id: str) -> None:
        with self._lock:
            state = self._ensure_state(session_id)
            state["last_input"] = time.time()
            if state.get("enabled"):
                state["next_deadline"] = state["last_input"] + self._random_interval()
                self._pending.pop(session_id, None)

    def pop_event(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            queue = self._pending.get(session_id) or []
            if not queue:
                return None
            event = queue.pop(0)
            if not queue:
                self._pending.pop(session_id, None)
            return event

    def _loop(self) -> None:
        while True:
            time.sleep(IDLE_TRIGGER_CHECK_INTERVAL)
            now = time.time()
            sessions: List[Tuple[str, Dict[str, Any]]] = []
            with self._lock:
                for sid, state in self._sessions.items():
                    sessions.append((sid, dict(state)))
            for session_id, state in sessions:
                if not state.get("enabled"):
                    continue
                if state.get("next_deadline") is None:
                    continue
                if self._pending.get(session_id):
                    continue
                if state.get("trigger_count", 0) >= IDLE_TRIGGER_MAX_PER_DAY:
                    continue
                if now < float(state.get("next_deadline") or 0):
                    continue
                last_input = float(state.get("last_input") or now)
                idle_seconds = max(0.0, now - last_input)
                with self._lock:
                    live_state = self._ensure_state(session_id)
                    if not live_state.get("enabled"):
                        continue
                    if live_state.get("trigger_count", 0) >= IDLE_TRIGGER_MAX_PER_DAY:
                        continue
                    if live_state.get("next_deadline") and now < float(live_state["next_deadline"]):
                        continue
                    event = _idle_trigger(session_id, idle_seconds)
                    self._pending.setdefault(session_id, []).append(event)
                    live_state["trigger_count"] = live_state.get("trigger_count", 0) + 1
                    live_state["last_input"] = now
                    live_state["next_deadline"] = now + self._random_interval()


idle_manager = IdleManager()


def _get_session_id(request: Optional["gr.Request"]) -> str:
    if request is None:
        return "default"
    session_hash = getattr(request, "session_hash", None)
    if session_hash:
        return str(session_hash)
    client_id = getattr(request, "client", None)
    if client_id:
        return str(client_id)
    return "default"

# =============== 系统提示模板与关键术语记忆 ===============
FRAME_PROMPT_TEMPLATE = (
    "你是早濑优香，请你基于以下内容给出对用户的回答：\n"
    "1.你的人物设定：{persona}\n"
    "2.你的历史记忆：{history}\n"
    "3.你的长期记忆：{long_term}\n"
    "4.当前优香情绪：{emotion}\n"
    "5.关键术语：{keywords}；用户（老师）输入：{user_input}"
)


def _extract_chinese_chars(text: str) -> Set[str]:
    return {ch for ch in (text or "") if "\u4e00" <= ch <= "\u9fff"}


class _FallbackIndex:
    def __init__(self, dim: int, vectors: Optional["np.ndarray"] = None):
        self.dim = dim
        if vectors is None:
            self.vectors = np.zeros((0, dim), dtype=np.float32)
        else:
            self.vectors = vectors.astype(np.float32)

    @property
    def ntotal(self) -> int:
        return int(self.vectors.shape[0])

    def add(self, vecs: "np.ndarray"):
        if vecs.size == 0:
            return
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError("Vector shape mismatch")
        self.vectors = np.concatenate([self.vectors, vecs.astype(np.float32)], axis=0)

    def search(self, query: "np.ndarray", top_k: int):
        if query.ndim != 2 or query.shape[1] != self.dim:
            raise ValueError("Query shape mismatch")
        n = query.shape[0]
        if self.ntotal == 0:
            distances = np.zeros((n, top_k), dtype=np.float32)
            indices = -np.ones((n, top_k), dtype=np.int64)
            return distances, indices
        sims = query @ self.vectors.T
        distances = np.zeros((n, top_k), dtype=np.float32)
        indices = -np.ones((n, top_k), dtype=np.int64)
        for i in range(n):
            scores = sims[i]
            order = np.argsort(-scores)
            take = order[:top_k]
            distances[i, : len(take)] = scores[take]
            indices[i, : len(take)] = take
        return distances, indices


class KeywordMemory:
    def __init__(self, path: Path):
        self.path = path
        self.entries: List[Dict[str, Any]] = []
        self.vector_dim = 384
        self._vector_ready = False
        self._batch_size = 6
        self.search_threshold = 0.24
        self._vector_index = None
        self._vector_meta: List[Dict[str, Any]] = []
        self._pending: List[Tuple["np.ndarray", Dict[str, Any]]] = [] if np is not None else []
        self._last_indexed_turn: Dict[str, int] = {}
        self._last_indexed_summary: Dict[str, int] = {}
        self._last_super_summary: Dict[str, str] = {}
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
        self._init_vector_index()

    def _init_vector_index(self):
        if np is None:
            return

        self._vector_meta = []
        self._pending = []
        self._vector_ready = False
        self._last_indexed_turn = {}
        self._last_indexed_summary = {}
        self._last_super_summary = {}

        meta_path = self.path.with_suffix(".meta.json")
        faiss_path = self.path.with_suffix(".faiss")
        numpy_path = self.path.with_suffix(".npy")

        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, list):
                        self._vector_meta = data
            except (json.JSONDecodeError, OSError):
                self._vector_meta = []

        if faiss is not None:
            if faiss_path.exists():
                try:
                    self._vector_index = faiss.read_index(str(faiss_path))
                except Exception:
                    self._vector_index = faiss.IndexFlatIP(self.vector_dim)
            else:
                self._vector_index = faiss.IndexFlatIP(self.vector_dim)
                if numpy_path.exists():
                    try:
                        arr = np.load(numpy_path)
                        if arr.ndim == 2 and arr.shape[1] == self.vector_dim and arr.size:
                            self._vector_index.add(arr.astype(np.float32))
                    except Exception:
                        pass
        else:
            existing = None
            if numpy_path.exists():
                try:
                    arr = np.load(numpy_path)
                    if arr.ndim == 2 and arr.shape[1] == self.vector_dim:
                        existing = arr.astype(np.float32)
                except Exception:
                    existing = None
            self._vector_index = _FallbackIndex(self.vector_dim, existing)

        if self._vector_index is None:
            return

        ntotal = getattr(self._vector_index, "ntotal", 0)
        if callable(ntotal):  # faiss swig can expose as method
            ntotal = ntotal()
        if not isinstance(ntotal, int):
            ntotal = int(ntotal)

        if len(self._vector_meta) > ntotal:
            self._vector_meta = self._vector_meta[:ntotal]
        elif len(self._vector_meta) < ntotal:
            for _ in range(ntotal - len(self._vector_meta)):
                self._vector_meta.append({})

        self._vector_ready = True
        self._rebuild_vector_caches()

    def _rebuild_vector_caches(self):
        self._last_indexed_turn = {}
        self._last_indexed_summary = {}
        self._last_super_summary = {}
        for meta in self._vector_meta:
            if not isinstance(meta, dict):
                continue
            mem_id = str(meta.get("mem_id") or DEFAULT_MEM_ID)
            if meta.get("type") == "turn":
                idx = int(meta.get("turn", -1))
                current = self._last_indexed_turn.get(mem_id, -1)
                if idx > current:
                    self._last_indexed_turn[mem_id] = idx
            elif meta.get("type") == "summary":
                idx = int(meta.get("summary_idx", -1))
                current = self._last_indexed_summary.get(mem_id, -1)
                if idx > current:
                    self._last_indexed_summary[mem_id] = idx
            elif meta.get("type") == "super_summary":
                marker = str(meta.get("hash") or "")
                if marker:
                    self._last_super_summary[mem_id] = marker

    def match(self, query: str) -> List[Dict[str, Any]]:
        user_chars = _extract_chinese_chars(query)
        if len(user_chars) < 2:
            return []
        matches: List[Dict[str, Any]] = []
        for entry in self.entries:
            if len(user_chars & entry["chars"]) >= 2:
                matches.append(entry)
        return matches

    # === 向量索引功能 ===
    def _should_index(self, text: str) -> bool:
        cleaned = (text or "").strip()
        if len(cleaned) < 12:
            return False
        unique = _extract_chinese_chars(cleaned)
        if len(unique) < 2 and len(cleaned.split()) <= 3:
            return False
        return True

    def _shorten(self, text: str, limit: int = 120) -> str:
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "…"

    def _tokenize_for_embedding(self, text: str) -> List[str]:
        lowered = text.lower()
        words = re.findall(r"[\w一-鿿]{2,12}", lowered)
        compact = re.sub(r"\s+", "", lowered)
        ngrams: List[str] = []
        for size in (2, 3):
            for i in range(len(compact) - size + 1):
                ngrams.append(compact[i:i+size])
        return words + ngrams

    def _embed_text(self, text: str) -> Optional["np.ndarray"]:
        if np is None or not self._vector_ready:
            return None
        content = (text or "").strip()
        if not content:
            return None
        tokens = self._tokenize_for_embedding(content)
        if not tokens:
            return None
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.vector_dim
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            return None
        return vec / norm

    def _queue_fragment(self, vector: "np.ndarray", meta: Dict[str, Any]):
        if np is None or not self._vector_ready:
            return
        self._pending.append((vector.astype(np.float32), meta))

    def queue_memory_fragments(self, mem: Dict[str, Any]):
        if np is None or not self._vector_ready:
            return
        mem_id = str(mem.get("id") or DEFAULT_MEM_ID)

        turns = mem.get("turns") or []
        last_turn_idx = self._last_indexed_turn.get(mem_id, -1)
        for idx in range(last_turn_idx + 1, len(turns)):
            turn = turns[idx]
            user_text = str(turn.get("u") or "").strip()
            bot_text = str(turn.get("a") or "").strip()
            combined = (f"[老师]{user_text}\n[优香]{bot_text}").strip()
            if not combined or not self._should_index(combined):
                continue
            vec = self._embed_text(combined)
            if vec is None:
                continue
            meta = {
                "type": "turn",
                "mem_id": mem_id,
                "turn": idx,
                "content": combined,
                "label": self._shorten(combined.replace("\n", " "), 140),
            }
            self._queue_fragment(vec, meta)
            self._last_indexed_turn[mem_id] = idx

        summaries = mem.get("block_summaries") or []
        last_summary_idx = self._last_indexed_summary.get(mem_id, -1)
        for idx in range(last_summary_idx + 1, len(summaries)):
            summary_text = str(summaries[idx].get("summary") or "").strip()
            if not summary_text or not self._should_index(summary_text):
                continue
            vec = self._embed_text(summary_text)
            if vec is None:
                continue
            meta = {
                "type": "summary",
                "mem_id": mem_id,
                "summary_idx": idx,
                "content": summary_text,
                "label": self._shorten(summary_text.replace("\n", " "), 160),
            }
            self._queue_fragment(vec, meta)
            self._last_indexed_summary[mem_id] = idx

        super_summary = str(mem.get("super_summary") or "").strip()
        if super_summary:
            marker = hashlib.md5(super_summary.encode("utf-8")).hexdigest()
            if marker != self._last_super_summary.get(mem_id):
                if self._should_index(super_summary):
                    vec = self._embed_text(super_summary)
                    if vec is not None:
                        meta = {
                            "type": "super_summary",
                            "mem_id": mem_id,
                            "hash": marker,
                            "content": super_summary,
                            "label": self._shorten(super_summary.replace("\n", " "), 160),
                        }
                        self._queue_fragment(vec, meta)
                self._last_super_summary[mem_id] = marker

    def flush_pending(self, force: bool = False) -> bool:
        if np is None or not self._vector_ready or not self._pending:
            return False
        if not force and len(self._pending) < self._batch_size:
            return False
        vectors = np.stack([item[0] for item in self._pending]).astype(np.float32)
        metas = [item[1] for item in self._pending]
        if faiss is not None and isinstance(self._vector_index, faiss.Index):
            self._vector_index.add(vectors)
        elif isinstance(self._vector_index, _FallbackIndex):
            self._vector_index.add(vectors)
        else:
            # Unknown index type
            self._pending.clear()
            return False
        self._vector_meta.extend(metas)
        self._pending.clear()
        self._persist_vector_store()
        return True

    def _persist_vector_store(self):
        if np is None or not self._vector_ready:
            return
        meta_path = self.path.with_suffix(".meta.json")
        faiss_path = self.path.with_suffix(".faiss")
        numpy_path = self.path.with_suffix(".npy")
        try:
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(self._vector_meta, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass

        if faiss is not None and isinstance(self._vector_index, faiss.Index):
            try:
                faiss.write_index(self._vector_index, str(faiss_path))
                if numpy_path.exists():
                    numpy_path.unlink()
            except Exception:
                pass
        elif isinstance(self._vector_index, _FallbackIndex):
            try:
                np.save(numpy_path, self._vector_index.vectors.astype(np.float32))
                if faiss_path.exists():
                    faiss_path.unlink()
            except Exception:
                pass

    def mark_dialogue_end(self):
        self.flush_pending(force=True)

    def _search_index(self, query: "np.ndarray", top_k: int):
        if faiss is not None and isinstance(self._vector_index, faiss.Index):
            return self._vector_index.search(query, top_k)
        if isinstance(self._vector_index, _FallbackIndex):
            return self._vector_index.search(query, top_k)
        return np.zeros((1, top_k), dtype=np.float32), -np.ones((1, top_k), dtype=np.int64)

    def search_similar_fragments(self, text: str, top_k: int = 3, min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        if np is None or not self._vector_ready:
            return []
        vec = self._embed_text(text)
        if vec is None:
            return []
        threshold = self.search_threshold if min_score is None else float(min_score)
        query = vec[np.newaxis, :]
        distances, indices = self._search_index(query, max(top_k * 2, top_k))
        results: List[Tuple[float, Dict[str, Any]]] = []
        if indices.size:
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self._vector_meta):
                    continue
                meta = self._vector_meta[idx]
                if not isinstance(meta, dict):
                    continue
                results.append((float(score), meta))

        if self._pending:
            pending_vecs = np.stack([item[0] for item in self._pending])
            pending_scores = pending_vecs @ vec
            for score, item in zip(pending_scores, self._pending):
                meta = item[1]
                results.append((float(score), meta))

        results.sort(key=lambda pair: pair[0], reverse=True)
        filtered: List[Dict[str, Any]] = []
        seen_labels: Set[str] = set()
        for score, meta in results:
            if score < threshold:
                continue
            label = str(meta.get("label") or meta.get("content") or "")
            if not label or label in seen_labels:
                continue
            seen_labels.add(label)
            enriched = dict(meta)
            enriched["score"] = score
            filtered.append(enriched)
            if len(filtered) >= top_k:
                break
        return filtered


DEFAULT_MOOD_LABEL = "平静"
DEFAULT_ATTITUDE_LABEL = "亲切"

_EMOTION_POSITIVE_TERMS = {
    "喜欢", "爱", "期待", "开心", "高兴", "满意", "支持", "信任", "感激", "谢谢", "太好了", "真好", "厉害",
}
_EMOTION_WARM_TERMS = {
    "抱抱", "安慰", "辛苦", "辛苦了", "慰问", "关心", "放心", "别怕", "温柔", "谢谢", "辛苦你", "拜托你",
}
_EMOTION_SAD_TERMS = {
    "难过", "伤心", "沮丧", "低落", "烦", "疲惫", "累", "焦虑", "紧张", "担心", "郁闷", "失望", "委屈",
}
_EMOTION_ANGRY_TERMS = {
    "生气", "气死", "烦死", "受不了", "讨厌", "怒", "恼", "火大", "很气", "不爽", "生氣",
}
_EMOTION_DISTANT_TERMS = {
    "算了", "别管", "不用", "不用了", "不用你", "不用帮", "别说", "闭嘴", "滚", "离开", "别理",
}


def _default_affect(label: str) -> Dict[str, Any]:
    now = int(time.time())
    return {
        "label": label,
        "score": 0.0,
        "updated_at": now,
        "source": "default",
    }


def new_dynamic_state() -> Dict[str, Any]:
    return {
        "turn": 0,
        "active": {},
        "mood": _default_affect(DEFAULT_MOOD_LABEL),
        "attitude": _default_affect(DEFAULT_ATTITUDE_LABEL),
    }


def _ensure_affect_record(state: Dict[str, Any], key: str, default_label: str) -> Dict[str, Any]:
    if key not in state or not isinstance(state.get(key), dict):
        state[key] = _default_affect(default_label)
        return state[key]

    record = state[key]
    if "label" not in record or not isinstance(record["label"], str):
        record["label"] = default_label
    if "score" not in record or not isinstance(record.get("score"), (int, float)):
        record["score"] = 0.0
    if "updated_at" not in record or not isinstance(record.get("updated_at"), (int, float)):
        record["updated_at"] = int(time.time())
    if "source" not in record or not isinstance(record.get("source"), str):
        record["source"] = "unknown"
    return record


def ensure_dynamic_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return new_dynamic_state()
    if "turn" not in state or "active" not in state:
        return new_dynamic_state()
    if not isinstance(state["active"], dict):
        state["active"] = {}
    try:
        state["turn"] = int(state.get("turn", 0))
    except Exception:
        state["turn"] = 0
    _ensure_affect_record(state, "mood", DEFAULT_MOOD_LABEL)
    _ensure_affect_record(state, "attitude", DEFAULT_ATTITUDE_LABEL)
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


def _normalize_affect_entry(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        label = value.get("label") or value.get("value") or value.get("text")
        score_val = value.get("score")
        if score_val is None:
            for key in ("intensity", "confidence", "weight"):
                if isinstance(value.get(key), (int, float)):
                    score_val = value[key]
                    break
        result: Dict[str, Any] = {}
        if isinstance(label, str) and label.strip():
            result["label"] = label.strip()
        if isinstance(score_val, (int, float)):
            result["score"] = float(score_val)
        if isinstance(value.get("source"), str):
            result["source"] = value["source"]
        return result
    if isinstance(value, str) and value.strip():
        return {"label": value.strip()}
    if isinstance(value, (list, tuple)) and value:
        label = str(value[0]).strip()
        result: Dict[str, Any] = {}
        if label:
            result["label"] = label
        if len(value) > 1 and isinstance(value[1], (int, float)):
            result["score"] = float(value[1])
        return result
    return {}


def _normalize_affect_result(raw: Any, source_name: str) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for key in ("mood", "attitude"):
        if key in raw:
            entry = _normalize_affect_entry(raw[key])
            if entry:
                entry.setdefault("source", source_name)
                normalized[key] = entry
    if "mood" not in normalized:
        entry = _normalize_affect_entry(raw.get("mood_label"))
        if entry:
            if isinstance(raw.get("mood_score"), (int, float)):
                entry["score"] = float(raw["mood_score"])
            entry.setdefault("source", source_name)
            normalized["mood"] = entry
    if "attitude" not in normalized:
        entry = _normalize_affect_entry(raw.get("attitude_label"))
        if entry:
            if isinstance(raw.get("attitude_score"), (int, float)):
                entry["score"] = float(raw["attitude_score"])
            entry.setdefault("source", source_name)
            normalized["attitude"] = entry
    return normalized


def _run_external_emotion_classifier(text: str) -> Optional[Dict[str, Dict[str, Any]]]:
    module_hints = []
    env_hint = os.environ.get("YUKA_EMOTION_CLASSIFIER")
    if env_hint:
        module_hints.extend([hint.strip() for hint in env_hint.split(",") if hint.strip()])
    module_hints.append("emotion_classifier")

    for name in module_hints:
        if not name:
            continue
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        classify = getattr(module, "classify", None)
        if not callable(classify):
            continue
        try:
            raw = classify(text)
        except Exception:
            continue
        normalized = _normalize_affect_result(raw, name)
        if normalized:
            return normalized
    return None


def _heuristic_affect_classification(text: str) -> Dict[str, Dict[str, Any]]:
    stripped = (text or "").strip()
    if not stripped:
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    positive = any(term in stripped for term in _EMOTION_POSITIVE_TERMS)
    warm = any(term in stripped for term in _EMOTION_WARM_TERMS)
    sad = any(term in stripped for term in _EMOTION_SAD_TERMS)
    angry = any(term in stripped for term in _EMOTION_ANGRY_TERMS)
    distant = any(term in stripped for term in _EMOTION_DISTANT_TERMS)

    exclamation = stripped.count("！") + stripped.count("!")
    question = stripped.count("？") + stripped.count("?")

    if angry:
        result["mood"] = {"label": "恼火", "score": -0.75}
        result["attitude"] = {"label": "抵触", "score": -0.65}
    elif sad:
        result["mood"] = {"label": "低落", "score": -0.55}
        result["attitude"] = {"label": "需要安慰", "score": 0.15}
    elif distant:
        result["mood"] = {"label": "冷淡", "score": -0.35}
        result["attitude"] = {"label": "疏离", "score": -0.45}
    elif positive or warm or exclamation >= 2:
        result["mood"] = {"label": "愉悦", "score": 0.6}
        result["attitude"] = {"label": "亲近", "score": 0.5}
    elif is_small_talk(stripped):
        result["mood"] = {"label": "轻松", "score": 0.35}
        result["attitude"] = {"label": "友好", "score": 0.3}
    elif question >= 1 and len(stripped) <= 48:
        result["mood"] = {"label": "专注", "score": 0.2}
        result["attitude"] = {"label": "求助", "score": 0.25}
    elif "抱歉" in stripped or "不好意思" in stripped:
        result["mood"] = {"label": "愧疚", "score": -0.2}
        result["attitude"] = {"label": "歉意", "score": 0.35}

    if not result and positive:
        result["mood"] = {"label": "愉悦", "score": 0.45}
        result["attitude"] = {"label": "亲近", "score": 0.4}

    for entry in result.values():
        entry["source"] = "heuristic"
    return result


def _apply_affect_update(state: Dict[str, Any], key: str, update: Dict[str, Any]) -> None:
    record = _ensure_affect_record(
        state,
        key,
        DEFAULT_MOOD_LABEL if key == "mood" else DEFAULT_ATTITUDE_LABEL,
    )
    if "label" in update and isinstance(update["label"], str) and update["label"].strip():
        record["label"] = update["label"].strip()
    if isinstance(update.get("score"), (int, float)):
        score = float(update["score"])
        record["score"] = max(-1.0, min(1.0, score))
    if isinstance(update.get("source"), str):
        record["source"] = update["source"].strip() or record.get("source", "heuristic")
    record["updated_at"] = int(time.time())


def _decay_affect(record: Dict[str, Any], default_label: str, factor: float = 0.2) -> None:
    score = float(record.get("score", 0.0))
    score *= (1.0 - factor)
    if abs(score) < 0.12:
        score = 0.0
        record["label"] = default_label
    record["score"] = score
    record["updated_at"] = int(time.time())


def _nudge_affect(record: Dict[str, Any], default_label: str, delta: float, positive_label: Optional[str] = None,
                  negative_label: Optional[str] = None) -> None:
    score = float(record.get("score", 0.0)) + delta
    score = max(-1.0, min(1.0, score))
    record["score"] = score
    if score > 0.35 and positive_label:
        record["label"] = positive_label
    elif score < -0.35 and negative_label:
        record["label"] = negative_label
    elif abs(score) < 0.15:
        record["label"] = default_label
    record["updated_at"] = int(time.time())


def _classify_user_affect(user_text: str) -> Dict[str, Dict[str, Any]]:
    external = _run_external_emotion_classifier(user_text)
    if external:
        return external
    return _heuristic_affect_classification(user_text)


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
    affect_updates = _classify_user_affect(user_text)
    if affect_updates:
        if "mood" in affect_updates:
            _apply_affect_update(state, "mood", affect_updates["mood"])
        if "attitude" in affect_updates:
            _apply_affect_update(state, "attitude", affect_updates["attitude"])
    _prune_dynamic_state(state)
    keywords = collect_dynamic_keywords(state) if enabled else []
    return state, keywords


def adjust_dynamic_state_after_reply(state: Dict[str, Any], reply_text: str) -> Dict[str, Any]:
    state = ensure_dynamic_state(state)
    reply = (reply_text or "").strip()
    mood_record = state["mood"]
    attitude_record = state["attitude"]

    _decay_affect(mood_record, DEFAULT_MOOD_LABEL)
    _decay_affect(attitude_record, DEFAULT_ATTITUDE_LABEL)

    if not reply:
        return state

    if "抱歉" in reply or "不好意思" in reply:
        _nudge_affect(attitude_record, DEFAULT_ATTITUDE_LABEL, 0.25, positive_label="关切")
        _nudge_affect(mood_record, DEFAULT_MOOD_LABEL, -0.15, negative_label="愧疚")
    if "谢谢" in reply or "感谢" in reply:
        _nudge_affect(attitude_record, DEFAULT_ATTITUDE_LABEL, 0.3, positive_label="感激")
        _nudge_affect(mood_record, DEFAULT_MOOD_LABEL, 0.1, positive_label="愉悦")
    if "放心" in reply or "别担心" in reply or "我在" in reply:
        _nudge_affect(attitude_record, DEFAULT_ATTITUDE_LABEL, 0.2, positive_label="安抚")
        _nudge_affect(mood_record, DEFAULT_MOOD_LABEL, 0.15, positive_label="笃定")
    if reply.count("！") + reply.count("!") >= 2:
        _nudge_affect(mood_record, DEFAULT_MOOD_LABEL, 0.2, positive_label="兴奋")
    if "呵" in reply or "哼" in reply:
        _nudge_affect(attitude_record, DEFAULT_ATTITUDE_LABEL, -0.1, negative_label="撒娇")

    return state


def _format_affect_value(record: Dict[str, Any], default_label: str) -> str:
    label = record.get("label") or default_label
    score = record.get("score")
    ts = record.get("updated_at")
    parts = [label]
    if isinstance(score, (int, float)) and abs(score) > 1e-3:
        parts.append(f"{score:+.2f}")
    if isinstance(ts, (int, float)) and ts > 0:
        try:
            ts_text = time.strftime("%H:%M:%S", time.localtime(int(ts)))
            parts.append(f"更新于{ts_text}")
        except Exception:
            pass
    source = record.get("source")
    if isinstance(source, str) and source and source not in {"default", "heuristic", "unknown"}:
        parts.append(f"来源：{source}")
    return "（" + "，".join(parts[1:]) + "）" if len(parts) > 1 else parts[0]


def render_dynamic_emotion_summary(state: Dict[str, Any]) -> str:
    state = ensure_dynamic_state(state)
    mood_label = state["mood"].get("label") or DEFAULT_MOOD_LABEL
    attitude_label = state["attitude"].get("label") or DEFAULT_ATTITUDE_LABEL
    mood_desc = f"心情：{mood_label}"
    attitude_desc = f"对老师的态度：{attitude_label}"
    mood_meta = _format_affect_value(state["mood"], DEFAULT_MOOD_LABEL)
    attitude_meta = _format_affect_value(state["attitude"], DEFAULT_ATTITUDE_LABEL)
    if mood_meta.startswith("（"):
        mood_desc += mood_meta
    if attitude_meta.startswith("（"):
        attitude_desc += attitude_meta
    return f"{mood_desc}；{attitude_desc}"


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
        "super_summary": "",
        "dynamic_state": new_dynamic_state()
    }

def load_memory(mem_id: str) -> Dict:
    p = mem_path(mem_id)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            mem = json.load(f)
    else:
        mem = new_memory(mem_id)
    if "dynamic_state" not in mem:
        mem["dynamic_state"] = new_dynamic_state()
    else:
        mem["dynamic_state"] = ensure_dynamic_state(mem.get("dynamic_state"))
    return mem


def save_memory(mem: Dict):
    mem["dynamic_state"] = ensure_dynamic_state(mem.get("dynamic_state"))
    with open(mem_path(mem["id"]), "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)


# =============== 长期记忆（信息型） ===============
_FACT_LABELS = {
    "name": "姓名",
    "alias": "别称",
    "age": "年龄",
    "birthday": "生日",
    "interest": "兴趣",
    "like": "喜好",
    "dislike": "厌恶",
    "location": "所在地",
    "origin": "籍贯",
    "profession": "职业",
}


def load_long_term_memory() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not LONG_TERM_MEMORY_FILE.exists():
        return records
    with open(LONG_TERM_MEMORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(obj.get("key") or obj.get("type") or "fact").strip()
            value = str(obj.get("value") or obj.get("text") or "").strip()
            if not value:
                continue
            obj["key"] = key
            obj["value"] = value
            records.append(obj)
    records.sort(key=lambda x: int(x.get("created_at", 0)))
    return records


def save_long_term_memory(records: List[Dict[str, Any]]):
    LONG_TERM_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LONG_TERM_MEMORY_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def clear_long_term_memory():
    if LONG_TERM_MEMORY_FILE.exists():
        LONG_TERM_MEMORY_FILE.unlink()


def export_long_term_memory() -> Optional[str]:
    if LONG_TERM_MEMORY_FILE.exists() and LONG_TERM_MEMORY_FILE.stat().st_size > 0:
        return str(LONG_TERM_MEMORY_FILE)
    return None


def _sanitize_fact_value(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^[：:的地得\-\s]+", "", t)
    t = re.sub(r"[。！!？?,，；;、\s]+$", "", t)
    return t.strip()


_NAME_PATTERNS = [
    re.compile(r"(?:我叫|我的名字叫|我的名字是|叫我)(?P<name>[\u4e00-\u9fffA-Za-z]{2,12})"),
    re.compile(r"(?:可以叫我|大家都叫我)(?P<name>[\u4e00-\u9fffA-Za-z]{2,12})"),
]

_LOCATION_PATTERNS = [
    re.compile(r"(?:来自|来自于|住在|生活在|出生在)(?P<location>[\u4e00-\u9fffA-Za-z]{2,15})"),
    re.compile(r"在(?P<location>[\u4e00-\u9fffA-Za-z]{2,15})(?:工作|上学|读书|生活)"),
]

_AGE_PATTERN = re.compile(r"(?:我|今年|现在)(?:今年)?(?P<age>\d{1,2})岁")
_BIRTHDAY_PATTERN = re.compile(r"(?:生日|生日至|生在)(?P<birthday>\d{1,2}月\d{1,2}日)")

_INTEREST_KEYWORDS = {
    "喜欢": "interest",
    "热爱": "interest",
    "爱好": "interest",
    "痴迷": "interest",
    "偏爱": "interest",
    "最喜欢": "interest",
    "爱吃": "like",
    "爱喝": "like",
    "爱看": "like",
    "讨厌": "dislike",
    "不喜欢": "dislike",
}

_PROFESSION_KEYWORDS = [
    "老师", "会计", "财务", "学生", "社长", "部长", "经理", "顾问", "秘书", "设计师",
    "工程师", "程序员", "医生", "护士", "律师", "作家", "研究员", "分析师", "队长", "指挥",
]


def _detect_profession(text: str) -> List[Tuple[str, str]]:
    found: List[Tuple[str, str]] = []
    for job in _PROFESSION_KEYWORDS:
        phrase = f"我是{job}"
        if phrase in text:
            found.append(("profession", job))
        phrase_alt = f"担任{job}"
        if phrase_alt in text:
            found.append(("profession", job))
    return found


def _extract_keyword_values(text: str) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    for kw, key in _INTEREST_KEYWORDS.items():
        start = 0
        while True:
            idx = text.find(kw, start)
            if idx == -1:
                break
            frag = text[idx + len(kw): idx + len(kw) + 18]
            value = re.split(r"[，。！？,.；;\s]", frag)[0]
            value = _sanitize_fact_value(value)
            if len(value) >= 2:
                results.append((key, value))
            start = idx + len(kw)
    return results


def detect_long_term_facts(user_text: str) -> List[Dict[str, Any]]:
    text = (user_text or "").strip()
    if not text:
        return []

    candidates: List[Tuple[str, str]] = []
    for pat in _NAME_PATTERNS:
        for match in pat.finditer(text):
            value = _sanitize_fact_value(match.group("name"))
            if len(value) >= 2:
                candidates.append(("name", value))

    for pat in _LOCATION_PATTERNS:
        for match in pat.finditer(text):
            value = _sanitize_fact_value(match.group("location"))
            if len(value) >= 2:
                candidates.append(("location", value))

    for match in _AGE_PATTERN.finditer(text):
        value = _sanitize_fact_value(match.group("age"))
        if value:
            candidates.append(("age", f"{value}岁"))

    for match in _BIRTHDAY_PATTERN.finditer(text):
        value = _sanitize_fact_value(match.group("birthday"))
        if value:
            candidates.append(("birthday", value))

    candidates.extend(_extract_keyword_values(text))
    candidates.extend(_detect_profession(text))

    facts: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for key, value in candidates:
        if not value:
            continue
        norm_key = key.lower().strip() or "fact"
        norm_value = value.strip()
        if not norm_value or (norm_key, norm_value) in seen:
            continue
        seen.add((norm_key, norm_value))
        facts.append({
            "type": "fact",
            "key": norm_key,
            "value": norm_value,
            "source": "user",
        })
    return facts


def append_long_term_memory(new_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not new_entries:
        return load_long_term_memory()
    existing = load_long_term_memory()
    seen = {(rec.get("key", ""), rec.get("value", "")) for rec in existing}
    updated = list(existing)
    for rec in new_entries:
        key = _sanitize_fact_value(rec.get("key", "")) or "fact"
        value = _sanitize_fact_value(rec.get("value", "") or rec.get("text", ""))
        if not value or (key, value) in seen:
            continue
        entry = {
            "type": rec.get("type", "fact"),
            "key": key,
            "value": value,
            "source": rec.get("source", "user"),
            "created_at": int(time.time()),
        }
        if rec.get("meta"):
            entry["meta"] = rec["meta"]
        updated.append(entry)
        seen.add((key, value))
    if len(updated) > 200:
        updated = updated[-200:]
    save_long_term_memory(updated)
    return updated


def build_long_term_memory_block(records: List[Dict[str, Any]], limit: int = 12) -> str:
    if not records:
        return ""
    sorted_records = sorted(records, key=lambda r: int(r.get("created_at", 0)) or 0)
    take = sorted_records[-limit:]
    lines = []
    for rec in take:
        key = (rec.get("key") or "fact").lower()
        value = _sanitize_fact_value(rec.get("value") or rec.get("text") or "")
        if not value:
            continue
        label = _FACT_LABELS.get(key, key)
        lines.append(f"- {label}：{value}")
    return "\n".join(lines)


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
            eos_token_id=EOS_IDS if 'EOS_IDS' in globals() and EOS_IDS else getattr(tokenizer, "eos_token_id", None),
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
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None or BitsAndBytesConfig is None or PeftModel is None:
        raise RuntimeError("缺少模型依赖，无法加载模型。")

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

    # ===== 多 EOS：遇到对话终止标记更快收尾 =====
    eos_ids = []
    if tok.eos_token_id is not None:
        eos_ids.append(tok.eos_token_id)
    for sp in ["<|im_end|>", "<|endoftext|>"]:
        try:
            tid = tok.convert_tokens_to_ids(sp)
            if isinstance(tid, int) and tid >= 0 and tid not in eos_ids:
                eos_ids.append(tid)
        except Exception:
            pass

    return tok, mdl, eos_ids

# ====== 初始化 ======
tokenizer = None
model = None
EOS_IDS: List[int] = []

if os.environ.get("YUKA_SKIP_MODEL_LOAD") == "1":
    EOS_IDS = []
else:
    tokenizer, model, EOS_IDS = load_model()

BAN_STRINGS = ["http", "https", "www.", "sourceMappingURL", "#", "Teacher", "teacher", "sensei"]
BAD_WORDS_IDS = [ids for s in BAN_STRINGS if tokenizer is not None and (ids := tokenizer.encode(s, add_special_tokens=False))]
keyword_memory = KeywordMemory(KB_MEMORY_FILE)

# =============== Chat 模板与编码（Qwen 官方方式） ===============
def _apply_chat_template(msgs: List[Dict[str, str]]) -> Dict[str, Any]:
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
    if tokenizer is None or model is None:
        raise RuntimeError("模型尚未加载，无法生成摘要。")

    text_lines = []
    for t in turns:
        text_lines.append(f"[老师]{t['u']}")
        text_lines.append(f"[优香]{t['a']}")
    convo = "\n".join(text_lines[-60:])
    sys = (
        "你是总结助手。请把以下对话整理为结构化摘要，使用简体中文并严格按以下顺序输出：\n"
        "【对话概览】列出2-3个要点描述对话的主要事实或事件；\n"
        "【当前优香对老师的态度/情绪】用一句话概括优香此刻对老师的态度与情绪；\n"
        "【关系进展要点】列出1-2个要点说明关系上的变化、承诺或后续行动。\n"
        "不要编造信息，保持简洁并保留关键名词。"
    )
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


def _compress_summary_if_needed(summary: str, max_chars: int = 360, depth: int = 0, max_depth: int = 2) -> str:
    if not summary or len(summary) <= max_chars or depth >= max_depth:
        return summary.strip()

    pseudo_turns = [{"u": line.strip(), "a": ""} for line in summary.splitlines() if line.strip()]
    if not pseudo_turns:
        return summary[:max_chars]

    try:
        shorter = summarize_turns(pseudo_turns, max_tokens=120)
    except RuntimeError:
        return summary.strip()[:max_chars]

    shorter = shorter.strip()
    if not shorter or shorter == summary.strip():
        return summary.strip()[:max_chars]

    if len(shorter) >= len(summary):
        return summary.strip()[:max_chars]

    return _compress_summary_if_needed(shorter, max_chars=max_chars, depth=depth + 1, max_depth=max_depth)


def _dedupe_preserve(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def parse_structured_summary(summary: str) -> Dict[str, Any]:
    sections: Dict[str, Any] = {
        "overview": [],
        "attitude": "",
        "progress": [],
    }
    if not summary:
        return sections

    current: Optional[str] = None
    for raw_line in summary.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if "对话" in line and ("概" in line or "要点" in line):
            current = "overview"
            value = ""
            if "：" in line:
                value = line.split("：", 1)[1].strip()
            elif "】" in line:
                value = line.split("】", 1)[1].strip()
            if value:
                sections["overview"].append(value)
            continue
        if "当前优香对老师的态度" in line or "态度/情绪" in line:
            current = "attitude"
            value = ""
            if "：" in line:
                value = line.split("：", 1)[1].strip()
            elif "】" in line:
                value = line.split("】", 1)[1].strip()
            if value:
                sections["attitude"] = value
            continue
        if "关系" in line and "进展" in line:
            current = "progress"
            value = ""
            if "：" in line:
                value = line.split("：", 1)[1].strip()
            elif "】" in line:
                value = line.split("】", 1)[1].strip()
            if value:
                sections["progress"].append(value)
            continue

        cleaned = line.lstrip("-·• ").strip()
        if current == "overview":
            sections["overview"].append(cleaned)
        elif current == "progress":
            sections["progress"].append(cleaned)
        elif current == "attitude" and not sections["attitude"]:
            sections["attitude"] = cleaned

    sections["overview"] = _dedupe_preserve(sections["overview"])
    sections["progress"] = _dedupe_preserve(sections["progress"])

    if not sections["overview"] and summary.strip():
        sections["overview"] = [summary.strip()]

    if not sections["attitude"]:
        sections["attitude"] = ""

    return sections


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def render_structured_summary(summary: str, line_limit: int = 200) -> List[str]:
    sections = parse_structured_summary(summary)
    lines: List[str] = []

    if not sections["overview"] and not sections["attitude"] and not sections["progress"]:
        if summary.strip():
            lines.append(f"- {_truncate(summary.strip(), line_limit)}")
        return lines

    if sections["overview"]:
        lines.append("- 对话概览：")
        for item in sections["overview"][:3]:
            lines.append(f"  · {_truncate(item, line_limit)}")

    if sections["attitude"]:
        lines.append(f"- 当前优香对老师的态度/情绪：{_truncate(sections['attitude'], line_limit)}")

    if sections["progress"]:
        lines.append("- 关系进展要点：")
        for item in sections["progress"][:3]:
            lines.append(f"  · {_truncate(item, line_limit)}")

    return lines

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
            summ = _compress_summary_if_needed(summ)
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
            super_summ = summarize_turns(sample, max_tokens=120)
            mem["super_summary"] = _compress_summary_if_needed(super_summ, max_chars=400)

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
            lines.extend(render_structured_summary(s, line_limit=200))
    if mem.get("super_summary"):
        lines.append("【更早的整体回顾】")
        ss = mem["super_summary"].strip()
        lines.extend(render_structured_summary(ss, line_limit=220))
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
    # ---- 新增：采样参数统一门面 ----
    def _resolve_sampling(temperature: float, top_p: float, typical_p: Optional[float]):
        """
        - clamp: 防越界
        - 典型采样与 Top-p 二选一
        - 近似确定性时自动改为贪心（do_sample=False）
        """
        t = float(max(0.05, min(temperature, 1.5)))     # 温度下限 0.05，避免数值不稳
        use_typical = (typical_p is not None) and (float(typical_p) < 0.999)
        if use_typical:
            tp = None
            typ = float(max(0.50, min(typical_p, 0.999)))  # 典型采样有效区间
        else:
            tp = float(max(0.10, min(top_p, 1.0)))
            typ = None

        # 采样 or 贪心：当几乎没有随机性时切到贪心更稳
        almost_greedy = (t <= 0.12) and (not use_typical) and (tp >= 0.999 - 1e-6)
        do_sample = not almost_greedy
        return t, tp, typ, do_sample

    if seed is not None:
        torch.manual_seed(seed); random.seed(seed)

    inputs = _apply_chat_template(msgs)

    min_new = int(min_new_tokens) if min_new_tokens else 0
    if max_new_tokens and min_new >= max_new_tokens:
        min_new = max(1, max_new_tokens - 1)

    with torch.inference_mode():
        t, tp, typ, do_sample = _resolve_sampling(temperature, top_p, typical_p)

        gen_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            min_new_tokens=max(1, int(min_new)) if max_new_tokens else None,
            do_sample=do_sample,
            temperature=t,
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            length_penalty=float(length_penalty),
            bad_words_ids=BAD_WORDS_IDS,
            use_cache=True,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            # ★ 用多 EOS：列表形式让模型更早停在对话边界
            eos_token_id=EOS_IDS if EOS_IDS else getattr(tokenizer, "eos_token_id", None),
        )
        if do_sample:
            if typ is not None:
                gen_kwargs["typical_p"] = typ
            else:
                gen_kwargs["top_p"] = tp

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
def format_system_prompt(persona_text: str, history_text: str, long_term_text: str,
                         keyword_texts: List[str], dynamic_state: Dict[str, Any],
                         user_input: str) -> str:
    persona = persona_text.strip() if persona_text else "（未设置人物设定）"
    history = history_text.strip() if history_text else "（暂无历史记忆）"
    long_term = long_term_text.strip() if long_term_text else "（暂无长期记忆）"
    keywords = "；".join(keyword_texts) if keyword_texts else "（暂无关键术语）"
    user = user_input.strip() if user_input else "（空输入）"
    emotion = render_dynamic_emotion_summary(dynamic_state)
    return FRAME_PROMPT_TEMPLATE.format(
        persona=persona,
        history=history,
        long_term=long_term,
        emotion=emotion,
        keywords=keywords,
        user_input=user,
    )


def build_messages(history_msgs: List[Dict[str, str]], user_text: str,
                   persona_text: str, history_text: str, long_term_text: str,
                   keyword_texts: List[str], dynamic_state: Dict[str, Any]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    event_info = prepare_user_event(user_text)
    sys_content = format_system_prompt(
        persona_text,
        history_text,
        long_term_text,
        keyword_texts,
        dynamic_state,
        event_info["prompt_text"],
    )
    msgs.append({"role": "system", "content": sys_content})
    for m in history_msgs:
        msgs.append(m)
    msgs.append({"role": "user", "content": event_info["conversation_user_text"]})
    return msgs


def build_messages_for_continue(history_msgs: List[Dict[str, str]],
                                ui_hist: List[Tuple[str, str]],
                                persona_text: str, history_text: str, long_term_text: str,
                                keyword_texts: List[str], dynamic_state: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
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

    sys_content = format_system_prompt(persona_text, history_text, long_term_text, keyword_texts, dynamic_state, hidden_user)
    msgs: List[Dict[str, str]] = []
    msgs.append({"role": "system", "content": sys_content})
    for m in history_msgs:
        msgs.append(m)
    msgs.append({"role": "user", "content": hidden_user})
    return msgs, hidden_user


# =============== Gradio UI ===============
def _init_gradio_app():
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
            typical_slider = gr.Slider(0.5, 1.0, value=1.0, step=0.01, label="typical_p（=1 关闭；与 Top-p 二选一）")
            repetition_penalty = gr.Slider(1.0, 1.3, value=1.2, step=0.01, label="重复惩罚")
            no_repeat_ngram = gr.Slider(0, 8, value=4, step=1, label="no_repeat_ngram_size")

        with gr.Row():
            diverse = gr.Checkbox(label="多样化输出（候选采样）", value=True)
            k_candidates = gr.Slider(1, 3, value=2, step=1, label="候选数 k")
            max_new = gr.Slider(64, 512, value=MAX_NEW_TOKENS, step=16, label="最大生成长度")
            refine_enabled = gr.Checkbox(label="启用二阶段精修", value=False)
            idle_chat_enabled = gr.Checkbox(label="启用主动问候", value=False)

        chat = gr.Chatbot(height=520, label="与优香对话", type="tuples")
        txt = gr.Textbox(placeholder="和优香聊点什么吧…（直接输入即可）", show_label=False)
        with gr.Row():
            send = gr.Button("发送", variant="primary")
            btn_continue = gr.Button("继续说 ▶", variant="secondary")
            btn_repeat = gr.Button("重新说 ⟳", variant="secondary")
            clear = gr.Button("清空对话", variant="secondary")

        mem_export = gr.File(label="长期记忆导出", interactive=False)

        state_msgs = gr.State([{"role":"system","content":"(hidden)"}])
        state_mem = gr.State(new_memory(DEFAULT_MEM_ID))
        state_dynamic = gr.State(new_dynamic_state())

        # === 记忆按钮 ===
        def do_load_mem(mem_id_text):
            mem = load_memory(mem_id_text or DEFAULT_MEM_ID)
            ui = [[t["u"], t["a"]] for t in mem["turns"][-10:]]
            dyn_state = ensure_dynamic_state(mem.get("dynamic_state"))
            gr.Info(f"已加载记忆：{mem_id_text or DEFAULT_MEM_ID}")
            return mem, ui, dyn_state
        btn_load_mem.click(do_load_mem, inputs=[mem_id], outputs=[state_mem, chat, state_dynamic])

        def do_save_mem(mem_obj, mem_id_text):
            mem_obj["id"] = mem_id_text or DEFAULT_MEM_ID
            save_memory(mem_obj)
            try:
                keyword_memory.queue_memory_fragments(mem_obj)
                keyword_memory.mark_dialogue_end()
            except Exception:
                pass
            export_path = export_long_term_memory()
            if export_path:
                gr.Info(f"已保存记忆：{mem_obj['id']}；长期记忆可下载")
            else:
                gr.Info(f"已保存记忆：{mem_obj['id']}（暂无长期记忆）")
            return export_path
        btn_save_mem.click(do_save_mem, inputs=[state_mem, mem_id], outputs=[mem_export])

        def do_clear_mem(mem_id_text):
            m = new_memory(mem_id_text or DEFAULT_MEM_ID)
            save_memory(m)
            clear_long_term_memory()
            try:
                keyword_memory.mark_dialogue_end()
            except Exception:
                pass
            gr.Info("记忆、长期记忆与对话已清空")
            return m, [], [{"role": "system", "content": "(hidden)"}], new_dynamic_state(), None
        btn_clear_mem.click(do_clear_mem, inputs=[mem_id], outputs=[state_mem, chat, state_msgs, state_dynamic, mem_export])

        # === 发消息 ===
        def _send(user_input, ui_hist, msg_hist, mem_obj, dyn_state,
                  use_rag_flag, mem_en_flag,
                  temp, topp, typical_val, rep_pen, ngram, diverse_flag, k_cand, max_tokens, refine_flag,
                  idle_enabled_flag, request: "gr.Request", event_context=None):

            session_id = _get_session_id(request)
            if idle_enabled_flag:
                idle_manager.enable(session_id)
            else:
                idle_manager.disable(session_id)

            normalized = prepare_user_event(user_input)
            if not normalized["is_idle"] and not normalized["prompt_text"]:
                return ui_hist, msg_hist, mem_obj, ensure_dynamic_state(dyn_state)

            if not normalized["is_idle"]:
                idle_manager.mark_activity(session_id)

            mem_obj = mem_obj or new_memory(DEFAULT_MEM_ID)
            dyn_state = ensure_dynamic_state(dyn_state)

            persona_text = build_persona_block()
            history_text = build_memory_context(mem_obj) if mem_obj.get("turns") else ""

            if mem_en_flag:
                long_term_records = load_long_term_memory()
                if not normalized["is_idle"]:
                    new_facts = detect_long_term_facts(normalized["prompt_text"])
                    if new_facts:
                        long_term_records = append_long_term_memory(new_facts)
                long_term_text = build_long_term_memory_block(long_term_records)
            else:
                long_term_text = ""

            if normalized["is_idle"]:
                keywords_for_prompt = collect_dynamic_keywords(dyn_state) if use_rag_flag else []
            else:
                dyn_state, keyword_texts = advance_dynamic_state(
                    dyn_state,
                    normalized["prompt_text"],
                    keyword_memory,
                    bool(use_rag_flag),
                )
                keywords_for_prompt = list(keyword_texts)

            vector_snippets: List[str] = []
            search_basis = normalized["prompt_text"]
            if not normalized["is_idle"] and bool(use_rag_flag) and hasattr(keyword_memory, "search_similar_fragments"):
                try:
                    hits = keyword_memory.search_similar_fragments(search_basis, top_k=3)
                except Exception:
                    hits = []
                for hit in hits:
                    label = str(hit.get("label") or hit.get("content") or "").strip()
                    if not label:
                        continue
                    snippet = f"【回顾】{label}"
                    if snippet not in keywords_for_prompt and snippet not in vector_snippets:
                        vector_snippets.append(snippet)
            if vector_snippets:
                keywords_for_prompt.extend(vector_snippets)
            if not normalized["is_idle"] and any(kw in (normalized["prompt_text"] or "") for kw in ACCOUNTING_KEYWORDS):
                keywords_for_prompt = [ACCOUNTING_ALERT] + keywords_for_prompt

            msgs = build_messages(
                msg_hist or [{"role": "system", "content": "(hidden)"}],
                user_input,
                persona_text,
                history_text,
                long_term_text,
                keywords_for_prompt,
                dyn_state,
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
            ui_hist_next.append([normalized["ui_user_text"], reply])
            msg_hist_next = list(msg_hist or [{"role": "system", "content": "(hidden)"}])
            msg_hist_next.extend([
                {"role": "user", "content": normalized["conversation_user_text"]},
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
                        long_term_text,
                        keywords_for_prompt,
                        dyn_state,
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

            dyn_state = adjust_dynamic_state_after_reply(dyn_state, full_reply)
            mem_obj["dynamic_state"] = dyn_state

            if mem_en_flag:
                mem_obj["turns"].append({"u": normalized["memory_user_text"], "a": reply})
                refresh_memory_layers(mem_obj)
                try:
                    keyword_memory.queue_memory_fragments(mem_obj)
                    keyword_memory.flush_pending(force=False)
                except Exception:
                    pass
                save_memory(mem_obj)
            else:
                try:
                    keyword_memory.flush_pending(force=True)
                except Exception:
                    pass

            return ui_hist, msg_hist, mem_obj, dyn_state

        send.click(
            _send,
            inputs=[txt, chat, state_msgs, state_mem, state_dynamic,
                    use_rag, mem_enabled,
                    temperature, top_p, typical_slider, repetition_penalty, no_repeat_ngram,
                    diverse, k_candidates, max_new, refine_enabled,
                    idle_chat_enabled],
            outputs=[chat, state_msgs, state_mem, state_dynamic]
        ).then(lambda: "", None, txt)

        txt.submit(
            _send,
            inputs=[txt, chat, state_msgs, state_mem, state_dynamic,
                    use_rag, mem_enabled,
                    temperature, top_p, typical_slider, repetition_penalty, no_repeat_ngram,
                    diverse, k_candidates, max_new, refine_enabled,
                    idle_chat_enabled],
            outputs=[chat, state_msgs, state_mem, state_dynamic]
        ).then(lambda: "", None, txt)

        # === 继续说（智能续写；不显示“继续”的用户消息；直接拼到上一条助手回复） ===
        def _continue(ui_hist, msg_hist, mem_obj, dyn_state,
                      use_rag_flag, mem_en_flag,
                      temp, topp, typical_val, rep_pen, ngram, diverse_flag, k_cand, max_tokens, refine_flag,
                      idle_enabled_flag, request: "gr.Request"):

            session_id = _get_session_id(request)
            if idle_enabled_flag:
                idle_manager.enable(session_id)
                idle_manager.mark_activity(session_id)
            else:
                idle_manager.disable(session_id)

            if not ui_hist or not isinstance(ui_hist[-1], (list, tuple)) or not ui_hist[-1][1]:
                return ui_hist, msg_hist, mem_obj, ensure_dynamic_state(dyn_state)

            mem_obj = mem_obj or new_memory(DEFAULT_MEM_ID)
            dyn_state = ensure_dynamic_state(dyn_state)

            persona_text = build_persona_block()
            history_text = build_memory_context(mem_obj) if mem_obj.get("turns") else ""
            if mem_en_flag:
                long_term_records = load_long_term_memory()
                long_term_text = build_long_term_memory_block(long_term_records)
            else:
                long_term_text = ""
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
                long_term_text,
                keywords_for_prompt,
                dyn_state,
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
                length_penalty=float(DEFAULT_LENGTH_PENALTY),
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

            dyn_state = adjust_dynamic_state_after_reply(dyn_state, truncated)
            mem_obj["dynamic_state"] = dyn_state

            if mem_en_flag:
                save_memory(mem_obj)

            return ui_hist, msg_hist, mem_obj, dyn_state

        btn_continue.click(
            _continue,
            inputs=[chat, state_msgs, state_mem, state_dynamic,
                    use_rag, mem_enabled,
                    temperature, top_p, typical_slider, repetition_penalty, no_repeat_ngram,
                    diverse, k_candidates, max_new, refine_enabled,
                    idle_chat_enabled],
            outputs=[chat, state_msgs, state_mem, state_dynamic]
        )

        def _repeat(ui_hist, msg_hist, mem_obj, dyn_state, idle_enabled_flag, request: "gr.Request"):
            session_id = _get_session_id(request)
            if idle_enabled_flag:
                idle_manager.enable(session_id)
                idle_manager.mark_activity(session_id)
            else:
                idle_manager.disable(session_id)
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
            inputs=[chat, state_msgs, state_mem, state_dynamic, idle_chat_enabled],
            outputs=[chat, state_msgs, state_mem, state_dynamic]
        )

        def _poll_idle(ui_hist, msg_hist, mem_obj, dyn_state,
                       use_rag_flag, mem_en_flag,
                       temp, topp, typical_val, rep_pen, ngram, diverse_flag, k_cand, max_tokens, refine_flag,
                       idle_enabled_flag, request: "gr.Request"):
            session_id = _get_session_id(request)
            if idle_enabled_flag:
                idle_manager.enable(session_id)
            else:
                idle_manager.disable(session_id)
                return ui_hist, msg_hist, mem_obj, ensure_dynamic_state(dyn_state)
            event = idle_manager.pop_event(session_id)
            if not event:
                return ui_hist, msg_hist, mem_obj, ensure_dynamic_state(dyn_state)
            marker = event.get("marker")
            if not marker:
                return ui_hist, msg_hist, mem_obj, ensure_dynamic_state(dyn_state)
            return _send(
                marker,
                ui_hist,
                msg_hist,
                mem_obj,
                dyn_state,
                use_rag_flag,
                mem_en_flag,
                temp,
                topp,
                typical_val,
                rep_pen,
                ngram,
                diverse_flag,
                k_cand,
                max_tokens,
                refine_flag,
                idle_enabled_flag,
                request,
                event_context=event,
            )

        gr.Timer(interval=IDLE_TRIGGER_CHECK_INTERVAL).tick(
            _poll_idle,
            inputs=[chat, state_msgs, state_mem, state_dynamic,
                    use_rag, mem_enabled,
                    temperature, top_p, typical_slider, repetition_penalty, no_repeat_ngram,
                    diverse, k_candidates, max_new, refine_enabled,
                    idle_chat_enabled],
            outputs=[chat, state_msgs, state_mem, state_dynamic],
        )

        # === 重载关键术语记忆 ===
        def reload_hard():
            keyword_memory.load()
            gr.Info("关键术语记忆已重载")
            return None
        gr.Button("重载关键术语", variant="secondary").click(reload_hard, outputs=[])

    return demo


if GRADIO_AVAILABLE:
    demo = _init_gradio_app()
else:
    demo = None

if __name__ == "__main__" and GRADIO_AVAILABLE:
    (demo or _init_gradio_app()).queue().launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=False,
        show_error=True
    )
