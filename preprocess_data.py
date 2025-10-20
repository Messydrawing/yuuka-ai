#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预处理优香对话数据，生成训练用 JSONL，其中每条样本包含可直接传入
`tokenizer.apply_chat_template` 的 `messages` 列表。

1) 原生支持 ModelScope/ChatML 常见格式：顶层 {"messages": [...]}，其中每个 turn 含 {"role","content"}。
2) 统一角色：'user'/'老师' → '用户'；'assistant'/'优香' → '优香'。
3) 默认使用“滑动窗口切片”产样：凡是以“优香”作结且前面至少有一条“用户”，都会产一条样本（受 min/max-turn 约束）。
   - 如需旧行为（只取每段对话的最后一问一答），将 --window-mode 设为 last。
4) 默认限制上下文为最近4句，并确保紧邻优香回答的发言来自“用户”。
5) 过滤无意义问答（如只有语气词/标点的回复或“？”式提问）。
6) 继续支持：Persona 注入、记忆库提示、中文比例过滤、验证集切分，并打印统计信息便于排错。

用法示例：
python preprocess_data.py \
  --input "data/yuuka_dialogues_zh.jsonl" \
  --train-output "data/processed/train.jsonl" \
  --val-output "data/processed/val.jsonl" \
  --val-ratio 0.08 --seed 42 --min-turn 2 --max-turn 5 \
  --persona-prob 0.35 --merge-prev-prob 0.0 --emotion-prob 0.0 \
  --min-completion-zh-ratio 0.55 \
  --memory-file "kb/hard_memory.jsonl" \
  --persona-memory-count 3 --relationship-memory-count 2 \
  --rule-memory-count 2 --world-memory-count 1 --memory-event-count 1
"""
import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ========= 基础工具 =========

def detect_chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    return zh / max(total, 1)


MEANINGLESS_COMPLETION_PATTERN = re.compile(r"^[啊哈嗯噢哦唔呀哎呜唉…\.\!\?！？。\s]+$")


def is_meaningful_completion(text: str) -> bool:
    """过滤只有语气词/标点的无意义回复。"""
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < 2:
        return False
    if MEANINGLESS_COMPLETION_PATTERN.match(stripped):
        return False
    return True


def is_meaningful_user_query(text: str) -> bool:
    """过滤缺乏实际信息的用户提问。"""
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < 2:
        return False
    if re.fullmatch(r"[?？…\.\!！]+", stripped):
        return False
    return True

def save_jsonl(path: Path, items: List[Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def split_dataset(samples: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rnd = random.Random(seed)
    rnd.shuffle(samples)
    if not samples:
        return [], []
    k = max(1, int(len(samples) * val_ratio))
    return samples[k:], samples[:k]

# ========= 记忆库 / Persona =========

def load_memory_bank(memory_file: Path) -> Dict[str, List[str]]:
    """记忆库 JSONL：每行形如 {"type":"persona|relationship|rule|world|memory","text":"..."}"""
    bank: Dict[str, List[str]] = {}
    if not memory_file.exists():
        return bank
    with memory_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = str(obj.get("type") or obj.get("category") or "").strip()
            txt = str(obj.get("text") or obj.get("content") or "").strip()
            if t and txt:
                bank.setdefault(t, []).append(txt)
    return bank

def sample_memory(bank: Dict[str, List[str]], key: str, n: int, rng: random.Random) -> List[str]:
    if n <= 0 or key not in bank or not bank[key]:
        return []
    items = bank[key]
    if len(items) <= n:
        out = items[:]
    else:
        out = rng.sample(items, n)
    return [x.strip() for x in out if x.strip()]

# ========= 数据读取（支持多格式，新增 messages）=========

def normalize_role(role: str) -> str:
    r = str(role).strip()
    if r in {"user", "USER", "User", "老师", "Teacher", "teacher"}:
        return "用户"
    if r in {"assistant", "Assistant", "bot", "优香", "YUUKA", "Yuuka", "yuuka", "助手"}:
        return "优香"
    # 兜底：未知角色按原样返回
    return r

def load_conversations(input_path: Path) -> List[List[Dict[str, str]]]:
    """
    统一返回形如：
    [
      [ {"speaker":"用户","text":"..."}, {"speaker":"优香","text":"..."} , ... ],
      ...
    ]
    支持：
      - {"messages":[{"role","content"}...]}
      - {"dialogue":[...]} / {"conversation":[...]} / {"utterances":[...]}
      - 直接 list[dict] 或 list[str("角色：内容")]
    """
    convs: List[List[Dict[str, str]]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            conv: List[Dict[str, str]] = []

            # --- 优先：messages 结构 ---
            if isinstance(data, dict) and isinstance(data.get("messages"), list):
                for turn in data["messages"]:
                    role = normalize_role(turn.get("role") or turn.get("speaker") or "")
                    text = str(turn.get("content") or turn.get("text") or turn.get("utterance") or "").strip()
                    if not text:
                        continue
                    conv.append({"speaker": role, "text": text})

            # 兼容：dialogue / conversation / utterances
            elif isinstance(data, dict) and any(k in data for k in ["dialogue", "conversation", "utterances"]):
                lst = data.get("dialogue") or data.get("conversation") or data.get("utterances") or []
                for turn in lst:
                    if isinstance(turn, dict):
                        role = normalize_role(turn.get("role") or turn.get("speaker") or turn.get("identity") or "")
                        text = str(turn.get("content") or turn.get("text") or turn.get("utterance") or "").strip()
                        if text:
                            conv.append({"speaker": role, "text": text})
                    elif isinstance(turn, str):
                        seg = turn.strip()
                        if not seg:
                            continue
                        if "：" in seg:
                            spk, utt = seg.split("：", 1)
                        elif ":" in seg:
                            spk, utt = seg.split(":", 1)
                        else:
                            continue
                        conv.append({"speaker": normalize_role(spk.strip()), "text": utt.strip()})

            # 兼容：整行就是列表
            elif isinstance(data, list):
                for turn in data:
                    if isinstance(turn, dict):
                        role = normalize_role(turn.get("role") or turn.get("speaker") or "")
                        text = str(turn.get("content") or turn.get("text") or turn.get("utterance") or "").strip()
                        if text:
                            conv.append({"speaker": role, "text": text})
                    elif isinstance(turn, str):
                        seg = turn.strip()
                        if not seg:
                            continue
                        if "：" in seg:
                            spk, utt = seg.split("：", 1)
                        elif ":" in seg:
                            spk, utt = seg.split(":", 1)
                        else:
                            continue
                        conv.append({"speaker": normalize_role(spk.strip()), "text": utt.strip()})

            if conv:
                convs.append(conv)

    return convs

# ========= 样本构建 =========

def build_samples(
    convs: List[List[Dict[str, str]]],
    min_turn: int,
    max_turn: int,
    persona_prob: float,
    merge_prev_prob: float,
    emotion_prob: float,
    min_completion_zh_ratio: float,
    memory_bank: Dict[str, List[str]],
    persona_memory_count: int,
    relationship_memory_count: int,
    rule_memory_count: int,
    world_memory_count: int,
    memory_event_count: int,
    window_mode: str,
    rng: random.Random
) -> List[Dict[str, Any]]:

    samples: List[Dict[str, Any]] = []

    # 尝试读取人设文本（可选）
    persona_text_full = ""
    persona_file_path = Path("persona_yuuka.txt")
    if persona_file_path.exists():
        persona_text_full = persona_file_path.read_text(encoding="utf-8").strip()
    persona_lines_all = [ln.strip() for ln in persona_text_full.splitlines() if ln.strip()] if persona_text_full else []

    def make_persona_prompt() -> str:
        """构造 persona / 记忆 形式的系统提示。"""
        sections: List[str] = []
        if persona_lines_all:
            sections.append(persona_lines_all[0])  # 概述

        pts = sample_memory(memory_bank, "persona", persona_memory_count, rng) or (
            persona_lines_all[1:1+persona_memory_count] if persona_lines_all else []
        )
        if pts:
            sections.append("角色要点：")
            sections.extend(f"- {x}" for x in pts)

        rel = sample_memory(memory_bank, "relationship", relationship_memory_count, rng)
        if rel:
            sections.append("重要关系：")
            sections.extend(f"- {x}" for x in rel)

        rule = sample_memory(memory_bank, "rule", rule_memory_count, rng)
        if rule:
            sections.append("行为约束：")
            sections.extend(f"- {x}" for x in rule)

        world = sample_memory(memory_bank, "world", world_memory_count, rng)
        if world:
            sections.append("世界观提示：")
            sections.extend(f"- {x}" for x in world)

        mem = sample_memory(memory_bank, "memory", memory_event_count, rng)
        if mem:
            sections.append("共同经历：")
            sections.extend(f"- {x}" for x in mem)

        # 回答要求
        sections.append("回答要求：")
        sections.extend([
            "1. 先描述现状；",
            "2. 指出问题与风险；",
            "3. 给出预算与流程安排；",
            "4. 提出结论与行动清单。",
            "（若涉及预算或报销，请先盘点再给建议。）"
        ])

        return "\n".join(sections)

    def speaker_to_role(speaker: str) -> str:
        return "assistant" if speaker == "优香" else "user"

    # 遍历每段对话，产样
    for conv in convs:
        # 生成 candidate 片段（最后一条为优香，前面至少有一条用户）
        indices = []
        if window_mode == "last":
            # 只取最后一个“优香”作为答案
            last_i = None
            for i in range(len(conv)-1, -1, -1):
                if conv[i]["speaker"] == "优香":
                    last_i = i
                    break
            if last_i is not None:
                indices = [last_i]
        else:
            # sliding：所有以“优香”作结的位置
            indices = [i for i, t in enumerate(conv) if t["speaker"] == "优香"]

        for i in indices:
            # 取不超过 max_turn 的片段
            start = max(0, i - (max_turn - 1))
            seg = conv[start:i+1]

            # 长度 & 角色约束：至少 min_turn；且 seg[:-1] 至少有一条“用户”
            if len(seg) < min_turn:
                continue
            if not any(t["speaker"] == "用户" for t in seg[:-1]):
                continue
            if len(seg) < 2 or seg[-2]["speaker"] != "用户":
                # 最后一轮优香发言前必须是用户提问
                continue

            last_user_text = seg[-2]["text"]
            if not is_meaningful_user_query(last_user_text):
                continue

            completion_text = seg[-1]["text"]
            if not is_meaningful_completion(completion_text):
                continue

            # 构建消息上下文
            context_messages: List[Dict[str, str]] = []
            for t in seg[:-1]:
                context_messages.append({
                    "role": speaker_to_role(t["speaker"]),
                    "content": t["text"],
                })

            # 可选：并入上一句优香
            if merge_prev_prob > 0 and rng.random() < merge_prev_prob:
                if len(seg) >= 3 and seg[-2]["speaker"] == "优香":
                    prev_ans = seg[-2]["text"]
                    if prev_ans and not completion_text.startswith(prev_ans):
                        completion_text = prev_ans.rstrip("\n") + "\n" + completion_text

            # 预算/报销提醒：若最后一条“用户”中包含关键词
            if re.search(r"预算|报销", last_user_text):
                reminder = "请先盘点再给建议。"
                if not completion_text.startswith(reminder):
                    completion_text = f"{reminder}\n{completion_text}"

            # 情感（可选，默认0，不启用）
            if emotion_prob > 0 and rng.random() < emotion_prob:
                completion_text = completion_text.rstrip() + "\n（心声）我会尽量稳住情绪，理性分析。"

            # 中文比例过滤
            if min_completion_zh_ratio > 0:
                if detect_chinese_ratio(completion_text) < min_completion_zh_ratio:
                    continue

            # 常规样本
            base_messages = context_messages + [{"role": "assistant", "content": completion_text}]
            samples.append({"messages": base_messages})

            # persona 样本（概率触发）
            if rng.random() < persona_prob:
                persona_full_prompt = make_persona_prompt()
                persona_messages = [{"role": "system", "content": persona_full_prompt}] + base_messages
                samples.append({"messages": persona_messages})

    return samples

# ========= CLI =========

def parse_args():
    p = argparse.ArgumentParser(description="预处理优香对话数据，生成训练用JSONL")
    p.add_argument("--input", type=Path, default=Path("data/yuuka_dialogues_zh.jsonl"))
    p.add_argument("--train-output", type=Path, default=Path("data/processed/train.jsonl"))
    p.add_argument("--val-output", type=Path, default=Path("data/processed/val.jsonl"))
    p.add_argument("--val-ratio", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--min-turn", type=int, default=2, help="片段最少发言轮数（含最终优香回答）")
    p.add_argument("--max-turn", type=int, default=5, help="片段最大发言轮数（含最多4句上下文）")
    p.add_argument("--window-mode", choices=["sliding", "last"], default="sliding",
                   help="sliding：所有以优香作结的片段都产样；last：只取每段最后一个优香回答")
    p.add_argument("--persona-prob", type=float, default=0.35)
    p.add_argument("--merge-prev-prob", type=float, default=0.0)
    p.add_argument("--emotion-prob", type=float, default=0.0)
    p.add_argument("--min-completion-zh-ratio", type=float, default=0.55)

    p.add_argument("--memory-file", type=str, default="kb/hard_memory.jsonl")
    p.add_argument("--persona-memory-count", type=int, default=3)
    p.add_argument("--relationship-memory-count", type=int, default=2)
    p.add_argument("--rule-memory-count", type=int, default=2)
    p.add_argument("--world-memory-count", type=int, default=1)
    p.add_argument("--memory-event-count", type=int, default=1)
    return p.parse_args()

def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"找不到输入文件: {args.input}")

    random.seed(args.seed)
    rng = random.Random(args.seed)

    convs = load_conversations(args.input)
    print(f"[INFO] 读取到对话段数: {len(convs)}")

    memory_bank = {}
    if args.memory_file:
        mf = Path(args.memory_file)
        if mf.exists():
            memory_bank = load_memory_bank(mf)
            print(f"[INFO] 记忆库加载完成: {sum(len(v) for v in memory_bank.values())} 条")
        else:
            print(f"[WARN] 未找到记忆库文件: {mf}（将跳过记忆扩展）")

    samples = build_samples(
        convs=convs,
        min_turn=args.min_turn,
        max_turn=args.max_turn,
        persona_prob=args.persona_prob,
        merge_prev_prob=args.merge_prev_prob,
        emotion_prob=args.emotion_prob,
        min_completion_zh_ratio=args.min_completion_zh_ratio,
        memory_bank=memory_bank,
        persona_memory_count=args.persona_memory_count,
        relationship_memory_count=args.relationship_memory_count,
        rule_memory_count=args.rule_memory_count,
        world_memory_count=args.world_memory_count,
        memory_event_count=args.memory_event_count,
        window_mode=args.window_mode,
        rng=rng,
    )

    if not samples:
        raise ValueError("没有生成任何样本，请检查原始数据格式或参数设置。"
                         "（提示：确认顶层是否为 messages 列表；尝试 --window-mode sliding；放宽 --min-turn）")

    train_samples, val_samples = split_dataset(samples, args.val_ratio, args.seed)
    save_jsonl(args.train_output, train_samples)
    save_jsonl(args.val_output, val_samples if val_samples else train_samples[:1])

if __name__ == "__main__":
    main()
