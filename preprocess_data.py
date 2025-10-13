#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预处理优香对话数据（CharacterGLM 友好版）
- 滑窗短样本，同时可合并多句优香台词生成更长回复
- 可选地嵌入 persona few-shot 模板，示范 system/persona 提示
- 简易情绪提示增强，让模型学习带情绪的小心声表达
- 角色统一：优香 vs 用户（含老师/其他人）
- 生成 prompt-completion JSONL
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

YUUKA_ROLES = {"Yuuka", "优香", "assistant"}  # 数据里出现过的优香别名
USER_ROLES = {"user", "用户"}  # 老师及其他归为“用户”
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")

PERSONA_FEWSHOT = (
    "系统：你是早濑优香，千年学院研讨会的会计。默认使用中文与老师交流，"
    "保持嘴硬心软但条理清晰的语气。遇到预算或风险议题时要先盘点→列问题→给补救→定约束说明，"
    "说明结论前务必核对事实；如缺乏信息要礼貌说明不确定性，绝不捏造不存在的情报。"
)

STYLE_HINT = (
    "系统提示：请作为优香保持克制且逻辑顺序清晰的回答，"
    "优先回应老师的问题，必要时用编号/短句分层说明；若信息不足要说明缺口并给出需要的事实。"
)

STYLE_REQUIREMENTS = [
    "- 先确认老师的问题，再给结论和原因。",
    "- 一次说明一个重点，可用编号或分行避免混杂。",
    "- 缺少资料时直接说明不知道，并提出需要的事实。",
    "- 情绪或心声要简短，不要盖过主要内容。",
]

EMOTION_HINTS = {
    "高兴": {
        "keywords": {"谢谢", "太好了", "开心", "喜欢", "成功"},
        "heart": "（心声：嗯，进度比预期顺利，可以冷静报告。）",
    },
    "担心": {
        "keywords": {"没事吧", "小心", "注意", "受伤", "危险"},
        "heart": "（心声：有点担心，但得先确认事实。）",
    },
    "害羞": {
        "keywords": {"脸红", "抱歉", "害羞", "喜欢你", "约会", "拥抱"},
        "heart": "（心声：有点害羞，不过还是要讲清楚重点。）",
    },
    "生气": {
        "keywords": {"乱花钱", "预算爆", "超支", "胡闹", "气", "没凭证"},
        "heart": "（心声：要压住情绪，提醒老师遵守预算。）",
    },
    "惊讶": {
        "keywords": {"真的吗", "突然", "竟然", "诶", "什么情况"},
        "heart": "（心声：有点意外，但要先理顺状况。）",
    },
}

MEMORY_TYPE_ALIAS: Dict[str, str] = {
    "persona": "persona",
    "speech": "persona",
    "preference": "persona",
    "relationship": "relationship",
    "rule": "rule",
    "taboo": "rule",
    "world": "world",
    "academy": "world",
    "memory_event": "memory",
}

DEFAULT_RNG = random.Random()


def load_conversations(path: Path) -> List[Dict]:
    conversations: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                conversations.append(json.loads(line))
            except Exception:
                pass
    return conversations


def norm_role(role: str) -> str:
    r = (role or "").strip()
    if r in YUUKA_ROLES:
        return "优香"
    if r in USER_ROLES:
        return "用户"
    # 其他具名角色也并入“用户”，但保留名字到文本里更像真人群聊
    return "用户"


def normalize_text(role: str, text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # 老师句子尽量加前缀，形成“社交场”
    if role == "用户":
        if not t.startswith("[") and not t.startswith("【"):
            t = f"[老师] {t}"
    return t


def sliding_windows(msgs: List[Tuple[str, str]], min_turn=3, max_turn=6, step=2):
    """在对话上构造滑窗片段。"""
    n = len(msgs)
    for wnd in range(min_turn, min(max_turn + 1, n + 1)):
        i = 0
        while i + wnd <= n:
            seg = msgs[i : i + wnd]
            if seg[-1][0] != "优香":
                i += step
                continue
            yield seg
            i += step


def load_memory_bank(path: Path | None) -> Dict[str, List[str]]:
    bank: Dict[str, List[str]] = defaultdict(list)
    if not path:
        return bank
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            t = item.get("type")
            text = (item.get("text") or "").strip()
            if not t or not text:
                continue
            key = MEMORY_TYPE_ALIAS.get(t, t)
            bank[key].append(text)
    return bank


def sample_memory(
    bank: Dict[str, List[str]],
    key: str,
    count: int,
    rng: random.Random,
) -> List[str]:
    items = bank.get(key) or []
    if not items or count <= 0:
        return []
    if len(items) <= count:
        return items.copy()
    return rng.sample(items, count)


def compute_chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    chinese = sum(1 for ch in chars if CHINESE_RE.match(ch))
    return chinese / max(1, len(chars))


def build_samples(
    convs: List[Dict],
    min_turn=3,
    max_turn=6,
    min_c_len=2,
    max_c_len=160,
    merge_prev_prob: float = 0.3,
    persona_prob: float = 0.35,
    emotion_prob: float = 0.1,
    min_completion_zh_ratio: float = 0.5,
    memory_bank: Dict[str, List[str]] | None = None,
    persona_memory_count: int = 3,
    relationship_memory_count: int = 2,
    rule_memory_count: int = 2,
    world_memory_count: int = 1,
    memory_event_count: int = 1,
    rng: random.Random | None = None,
) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    seen = set()
    rng = rng or DEFAULT_RNG
    memory_bank = memory_bank or {}

    def add_sample(prompt: str, completion: str):
        prompt_norm = (prompt or "").strip("\n")
        completion_norm = (completion or "").strip()
        if not prompt_norm or not completion_norm:
            return
        if not CHINESE_RE.search(completion_norm):
            return
        if compute_chinese_ratio(completion_norm) < min_completion_zh_ratio:
            return
        if completion_norm.count("（心声") > 1:
            return
        segments = [
            seg
            for seg in re.split(r"[。！？!?\n]+", completion_norm)
            if seg.strip()
        ]
        if len(segments) > 8:
            return
        key = hash(prompt_norm + "\n" + completion_norm)
        if key in seen:
            return
        seen.add(key)
        samples.append({"prompt": prompt_norm, "completion": completion_norm})

    def detect_emotion(text: str) -> Tuple[str, str]:
        t = text or ""
        if not t:
            return "", ""
        for emo, meta in EMOTION_HINTS.items():
            if any(kw in t for kw in meta["keywords"]):
                return emo, meta["heart"]
        return "", ""

    for conv in convs:
        raw_msgs = conv.get("messages", [])
        msgs: List[Tuple[str, str]] = []
        for m in raw_msgs:
            role = norm_role(m.get("role", ""))
            text = normalize_text(role, m.get("content", ""))
            if text:
                msgs.append((role, text))

        if sum(bool(CHINESE_RE.search(t)) for _, t in msgs) < max(1, len(msgs) // 3):
            continue

        for seg in sliding_windows(msgs, min_turn=min_turn, max_turn=max_turn, step=2):
            ctx = seg[:-1]
            last = seg[-1]
            if last[0] != "优香":
                continue
            completion = last[1].strip()
            if not (min_c_len <= len(completion) <= max_c_len):
                continue

            style_hint_text = "".join(STYLE_HINT)

            lines = []
            for r, t in ctx:
                if r == "优香":
                    lines.append(f"优香: {t}")
                else:
                    lines.append(f"用户: {t}")
            prompt = "\n".join(lines + [style_hint_text, "优香: "])

            add_sample(prompt, completion)

            if ctx and ctx[-1][0] == "优香" and rng.random() < merge_prev_prob:
                merged = ctx[-1][1].strip()
                if merged:
                    merged = merged + ("\n" if completion else "") + completion
                else:
                    merged = completion
                if min_c_len <= len(merged) <= max_c_len * 2:
                    prompt_long = "\n".join(
                        [
                            ("优香: " + c[1]) if c[0] == "优香" else ("用户: " + c[1])
                            for c in ctx[:-1]
                        ]
                        + ["优香: "]
                    )
                    add_sample(prompt_long, merged)

            if rng.random() < persona_prob:
                ctx_plain = "\n".join(
                    ("优香：" + t) if r == "优香" else ("老师：" + t) for r, t in ctx
                )
                style_hint_text = "".join(STYLE_HINT)
                persona_lines = sample_memory(
                    memory_bank, "persona", persona_memory_count, rng
                )
                relationship_lines = sample_memory(
                    memory_bank, "relationship", relationship_memory_count, rng
                )
                rule_lines = sample_memory(memory_bank, "rule", rule_memory_count, rng)
                world_lines = sample_memory(memory_bank, "world", world_memory_count, rng)
                memory_events = sample_memory(memory_bank, "memory", memory_event_count, rng)

                persona_sections: List[str] = [PERSONA_FEWSHOT]
                if persona_lines:
                    persona_sections.append("角色要点：")
                    persona_sections.extend(f"- {line}" for line in persona_lines)
                if relationship_lines:
                    persona_sections.append("重要关系：")
                    persona_sections.extend(f"- {line}" for line in relationship_lines)
                if rule_lines:
                    persona_sections.append("行为约束：")
                    persona_sections.extend(f"- {line}" for line in rule_lines)
                if world_lines:
                    persona_sections.append("世界观提示：")
                    persona_sections.extend(f"- {line}" for line in world_lines)
                if memory_events:
                    persona_sections.append("共同经历：")
                    persona_sections.extend(f"- {line}" for line in memory_events)

                persona_sections.append("回答要求：")
                persona_sections.extend(STYLE_REQUIREMENTS)
                persona_sections.append(style_hint_text)

                persona_prompt = (
                    "\n".join(persona_sections)
                    + "\n对话记录：\n"
                    + ctx_plain
                    + "\n优香："
                )
                add_sample(persona_prompt, completion)

            emotion, heart = detect_emotion(completion)
            if emotion and heart and rng.random() < emotion_prob:
                enriched = completion
                if heart not in completion:
                    enriched = completion + ("\n" if not completion.endswith("\n") else "") + heart
                add_sample(prompt, enriched)

    return samples


def split_dataset(samples: List[Dict[str, str]], val_ratio: float, seed: int):
    rnd = random.Random(seed)
    rnd.shuffle(samples)
    n = len(samples)
    k = max(1, int(n * val_ratio))
    return samples[k:], samples[:k]


def save_jsonl(path: Path, items: List[Dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def parse_args():
    p = argparse.ArgumentParser("预处理优香对话（CharacterGLM）")
    p.add_argument("--input", type=Path, default=Path("data/yuuka_dialogues_zh.jsonl"))
    p.add_argument("--train-output", type=Path, default=Path("data/processed/train.jsonl"))
    p.add_argument("--val-output", type=Path, default=Path("data/processed/val.jsonl"))
    p.add_argument("--val-ratio", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-turn", type=int, default=3)
    p.add_argument("--max-turn", type=int, default=6)
    p.add_argument(
        "--max-completion-len",
        type=int,
        default=220,
        help="单条 completion 的最大长度（字符数）",
    )
    p.add_argument(
        "--persona-prob",
        type=float,
        default=0.35,
        help="生成带 system persona few-shot 模板样本的概率",
    )
    p.add_argument(
        "--merge-prev-prob",
        type=float,
        default=0.3,
        help="把上一句优香台词并入 completion 的概率（降低可减少杂糅段落）",
    )
    p.add_argument(
        "--emotion-prob",
        type=float,
        default=0.08,
        help="为 completion 附加情绪心声示例的概率（保持克制的小幅提示）",
    )
    p.add_argument(
        "--min-completion-zh-ratio",
        type=float,
        default=0.55,
        help="completion 中中文字符占比低于该阈值则丢弃样本",
    )
    p.add_argument(
        "--memory-file",
        type=str,
        default="hard_memory.jsonl",
        help="包含 persona/世界观记忆的 JSONL，可为空字符串以关闭",
    )
    p.add_argument(
        "--persona-memory-count",
        type=int,
        default=3,
        help="persona 模板中抽取角色要点的条目数",
    )
    p.add_argument(
        "--relationship-memory-count",
        type=int,
        default=2,
        help="persona 模板中抽取关系设定的条目数",
    )
    p.add_argument(
        "--rule-memory-count",
        type=int,
        default=2,
        help="persona 模板中抽取回答约束的条目数",
    )
    p.add_argument(
        "--world-memory-count",
        type=int,
        default=1,
        help="persona 模板中抽取世界观提示的条目数",
    )
    p.add_argument(
        "--memory-event-count",
        type=int,
        default=1,
        help="persona 模板中抽取共同经历的条目数",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    random.seed(args.seed)
    rng = random.Random(args.seed)
    convs = load_conversations(args.input)
    memory_file = Path(args.memory_file) if args.memory_file else None
    memory_bank = load_memory_bank(memory_file) if memory_file else {}

    samples = build_samples(
        convs,
        min_turn=args.min_turn,
        max_turn=args.max_turn,
        max_c_len=args.max_completion_len,
        merge_prev_prob=args.merge_prev_prob,
        persona_prob=args.persona_prob,
        emotion_prob=args.emotion_prob,
        min_completion_zh_ratio=args.min_completion_zh_ratio,
        memory_bank=memory_bank,
        persona_memory_count=args.persona_memory_count,
        relationship_memory_count=args.relationship_memory_count,
        rule_memory_count=args.rule_memory_count,
        world_memory_count=args.world_memory_count,
        memory_event_count=args.memory_event_count,
        rng=rng,
    )
    if not samples:
        raise ValueError("没有构建出可用样本，请检查原始数据。")

    train, val = split_dataset(samples, args.val_ratio, args.seed)
    save_jsonl(args.train_output, train)
    save_jsonl(args.val_output, val if val else train[:1])
    print(f"[OK] 训练样本 {len(train)} 条，验证样本 {len(val)} 条")
    print(f"train -> {args.train_output}")
    print(f"val   -> {args.val_output}")

if __name__ == "__main__":
    main()
