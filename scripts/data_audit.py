#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据质量巡检脚本

从预处理后的 prompt-completion JSONL 中统计语言占比、persona 模板采样
比例、长度分布等指标，帮助快速定位导致模型输出混乱的潜在原因。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Iterator, Tuple

CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")


def load_samples(path: Path) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            prompt = data.get("prompt")
            completion = data.get("completion")
            if not isinstance(prompt, str) or not isinstance(completion, str):
                continue
            yield {"prompt": prompt, "completion": completion}


def count_chars(text: str) -> Tuple[int, int, int]:
    total = 0
    chinese = 0
    ascii_letters = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if CHINESE_RE.match(ch):
            chinese += 1
        elif ASCII_LETTER_RE.match(ch):
            ascii_letters += 1
    return total, chinese, ascii_letters


def ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def summarise_lengths(values: Iterable[int]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {"min": 0, "max": 0, "mean": 0, "p90": 0}
    vals.sort()
    return {
        "min": float(vals[0]),
        "max": float(vals[-1]),
        "mean": float(mean(vals)),
        "p90": float(vals[int(math.ceil(0.9 * len(vals))) - 1]),
    }


def analyse_dataset(path: Path) -> Dict[str, object]:
    total = 0
    persona_count = 0
    heart_count = 0
    zh_ratios = []
    en_ratios = []
    completion_lens = []
    prompt_lens = []
    mood_counter: Counter[str] = Counter()

    for sample in load_samples(path):
        total += 1
        prompt = sample["prompt"]
        completion = sample["completion"]

        if prompt.startswith("系统："):
            persona_count += 1
        if "心声" in completion:
            heart_count += 1

        total_chars, zh_chars, en_chars = count_chars(completion)
        zh_ratios.append(ratio(zh_chars, total_chars))
        en_ratios.append(ratio(en_chars, total_chars))

        completion_lens.append(len(completion))
        prompt_lens.append(len(prompt))

        # 粗略情绪统计
        if "（心声" in completion:
            if "高兴" in completion:
                mood_counter["高兴"] += 1
            elif "担心" in completion:
                mood_counter["担心"] += 1
            elif "害羞" in completion:
                mood_counter["害羞"] += 1
            elif "生气" in completion:
                mood_counter["生气"] += 1
            elif "惊讶" in completion:
                mood_counter["惊讶"] += 1

    zh_mean = ratio(sum(zh_ratios), len(zh_ratios)) if zh_ratios else 0.0
    en_mean = ratio(sum(en_ratios), len(en_ratios)) if en_ratios else 0.0

    return {
        "samples": total,
        "persona_share": ratio(persona_count, total),
        "heart_share": ratio(heart_count, total),
        "zh_ratio_mean": zh_mean,
        "en_ratio_mean": en_mean,
        "completion_length": summarise_lengths(completion_lens),
        "prompt_length": summarise_lengths(prompt_lens),
        "mood_distribution": mood_counter,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Analyse Yuuka prompt-completion JSONL quality")
    p.add_argument("path", type=Path, help="训练或验证数据的 JSONL 路径")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.path.exists():
        raise FileNotFoundError(args.path)

    report = analyse_dataset(args.path)
    print("总样本数:", report["samples"])
    print(f"persona 模板占比: {report['persona_share']*100:.1f}%")
    print(f"带心声情绪样本占比: {report['heart_share']*100:.1f}%")
    print(f"completion 平均中文占比: {report['zh_ratio_mean']*100:.1f}%")
    print(f"completion 平均字母占比: {report['en_ratio_mean']*100:.1f}%")
    comp = report["completion_length"]
    print(
        "completion 长度统计: min={min:.0f}, mean={mean:.1f}, p90={p90:.0f}, max={max:.0f}".format(
            **comp
        )
    )
    prompt = report["prompt_length"]
    print(
        "prompt 长度统计:     min={min:.0f}, mean={mean:.1f}, p90={p90:.0f}, max={max:.0f}".format(
            **prompt
        )
    )
    if report["mood_distribution"]:
        print("情绪心声分布:")
        for mood, cnt in report["mood_distribution"].most_common():
            print(f"  - {mood}: {cnt}")


if __name__ == "__main__":
    main()
