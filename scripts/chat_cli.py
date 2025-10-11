#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""命令行聊天脚本，使用微调后的优香模型。"""

import argparse
from typing import List, Dict
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with fine-tuned Yuuka model")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct", help="基础模型 ID 或本地路径")
    parser.add_argument("--peft-model", default="models/qwen_yuuka_lora1/checkpoint-626", help="LoRA 权重路径")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="生成的最大 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling 的 top-p")
    parser.add_argument("--system-prompt",
        default="你是早濑优香（简体中文），正在与老师交流。",
        help="系统提示词")
    parser.add_argument("--offline", action="store_true", help="启用离线模式（不访问 HF Hub）")
    return parser.parse_args()

def load_model(base_model: str, peft_model_path: str, offline: bool=False):
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=offline)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=offline,
    )
    model = PeftModel.from_pretrained(model, peft_model_path, local_files_only=offline)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 同步 PAD/EOS，避免警告
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        try:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        except Exception:
            pass
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    return tokenizer, model, device

def ensure_teacher_prefix(s: str) -> str:
    # 没有显式说话人时，默认加 [老师] 前缀（贴合训练分布）
    if s.startswith("[") and "]" in s.split(" ", 1)[0]:
        return s
    return f"[老师] {s}"

def generate_reply(tokenizer, model, device, history: List[Dict[str, str]],
                   max_new_tokens: int, temperature: float, top_p: float) -> str:
    conversation = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([conversation], return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    reply = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return reply

def main() -> None:
    args = parse_args()
    tokenizer, model, device = load_model(args.base_model, args.peft_model, args.offline)

    history: List[Dict[str, str]] = []
    if args.system_prompt:
        history.append({"role": "system", "content": args.system_prompt})

    print("开始与优香的对话（输入 exit 结束）：")
    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("结束对话，再见！")
            break
        if not user_input:
            continue
        user_input = ensure_teacher_prefix(user_input)
        history.append({"role": "user", "content": user_input})

        reply = generate_reply(tokenizer, model, device, history,
                               args.max_new_tokens, args.temperature, args.top_p)
        print("优香:", reply)
        history.append({"role": "assistant", "content": reply})

        # 截断历史，保留最近 6 轮（12 条）
        if len(history) > 12:
            history = [history[0]] + history[-11:]

if __name__ == "__main__":
    main()
