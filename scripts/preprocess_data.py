#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预处理优香对话数据（CharacterGLM 友好版）
- 滑窗短样本，强调优香口吻与情绪
- 角色统一：优香 vs 用户（含老师/其他人）
- 生成 prompt-completion JSONL
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple

YUUKA_ROLES = {"Yuuka", "优香", "assistant"}  # 数据里出现过的优香别名
USER_ROLES  = {"user", "用户"}                 # 老师及其他归为“用户”
CHINESE_RE  = re.compile(r"[\u4e00-\u9fff]")

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

def sliding_windows(msgs: List[Tuple[str,str]], min_turn=3, max_turn=6, step=2):
    """
    msgs: [(role, text), ...]  role in {"优香","用户"}
    每个window保留3~6句（含双方），用于构成 prompt + completion（completion = 最后一条“优香”）
    """
    n = len(msgs)
    for wnd in range(min_turn, min(max_turn+1, n+1)):
        i = 0
        while i + wnd <= n:
            seg = msgs[i:i+wnd]
            # 只保留优香作为最后一句（让模型学“下一句优香怎么说”）
            if seg[-1][0] != "优香":
                i += step
                continue
            yield seg
            i += step

def build_samples(convs: List[Dict],
                  min_turn=3, max_turn=6,
                  min_c_len=2, max_c_len=160) -> List[Dict[str,str]]:
    samples: List[Dict[str,str]] = []
    seen = set()

    for conv in convs:
        raw_msgs = conv.get("messages", [])
        msgs: List[Tuple[str,str]] = []
        for m in raw_msgs:
            role = norm_role(m.get("role", ""))
            text = normalize_text(role, m.get("content",""))
            if text:
                msgs.append((role, text))

        # 过滤明显非中文/噪音会话
        if sum(bool(CHINESE_RE.search(t)) for _, t in msgs) < max(1, len(msgs)//3):
            continue

        for seg in sliding_windows(msgs, min_turn=min_turn, max_turn=max_turn, step=2):
            # prompt = seg[:-1]  completion = seg[-1](优香)
            ctx = seg[:-1]
            last = seg[-1]
            if last[0] != "优香":
                continue
            completion = last[1].strip()
            if not (min_c_len <= len(completion) <= max_c_len):
                continue
            if not CHINESE_RE.search(completion):
                continue

            # 组装prompt：以“优香: … / 用户: …”短句为主
            lines = []
            for r, t in ctx:
                if r == "优香":
                    lines.append(f"优香: {t}")
                else:
                    lines.append(f"用户: {t}")
            # 末尾给出“优香:”引导
            prompt = "\n".join(lines + ["优香: "])

            key = hash(prompt + "\n" + completion)
            if key in seen:
                continue
            seen.add(key)
            samples.append({"prompt": prompt, "completion": completion})

    return samples

def split_dataset(samples: List[Dict[str,str]], val_ratio: float, seed: int):
    rnd = random.Random(seed)
    rnd.shuffle(samples)
    n = len(samples)
    k = max(1, int(n * val_ratio))
    return samples[k:], samples[:k]

def save_jsonl(path: Path, items: List[Dict[str,str]]):
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
    return p.parse_args()

def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    convs = load_conversations(args.input)
    samples = build_samples(convs, min_turn=args.min_turn, max_turn=args.max_turn)
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
