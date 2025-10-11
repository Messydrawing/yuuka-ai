#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune CharacterGLM-6B with QLoRA on Yuuka dialogues."""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODEL_NAME = "models/CharacterGLM-6B"   # 本地目录；也可传 HuggingFace ID
DEFAULT_DATA_DIR   = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("models/yuuka_characterglm_lora")
MAX_SEQ_LEN = 1024

def get_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # ChatGLM/GLM 一般有 pad/eos，保险起见：
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def guess_lora_target_modules(model) -> List[str]:
    """根据模型模块名自动选择 LoRA 目标层（GLM系优先）"""
    # 收集所有子模块名字
    names = []
    for n, m in model.named_modules():
        names.append(n)

    # 优先：ChatGLM/GLM 常见层
    glm_candidates = ["query_key_value", "dense_h_to_4h", "dense_4h_to_h"]
    if any(any(c in n for n in names) for c in glm_candidates):
        return glm_candidates

    # 回退：Llama/通用注意力
    fallback = ["q_proj", "k_proj", "v_proj", "o_proj"]
    return fallback

def load_model(model_name: str):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    target_modules = guess_lora_target_modules(model)
    lora_cfg = LoraConfig(
        r=8,                 # 8 更省显存；若显存足可调到16
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model

class PromptCompletionCollator:
    """把 prompt+completion 拼在一起；只对 completion 计算损失"""
    def __init__(self, tok, max_len: int):
        self.tok = tok
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        input_ids, attn, labels = [], [], []
        pad_id = self.tok.pad_token_id

        for ex in batch:
            prompt = ex["prompt"]
            completion = ex["completion"] + self.tok.eos_token

            p_ids = self.tok.encode(prompt, add_special_tokens=False)
            c_ids = self.tok.encode(completion, add_special_tokens=False)

            ids = p_ids + c_ids
            lbl = [-100] * len(p_ids) + c_ids
            if len(ids) > self.max_len:
                # 从左侧截断，保留末尾（更接近最近上下文）
                ids  = ids[-self.max_len:]
                lbl  = lbl[-self.max_len:]

            mask = [1] * len(ids)
            input_ids.append(ids)
            attn.append(mask)
            labels.append(lbl)

        # 动态padding到batch内最大长度
        maxlen = max(len(x) for x in input_ids)
        for i in range(len(input_ids)):
            pad = maxlen - len(input_ids[i])
            input_ids[i] += [pad_id] * pad
            attn[i]      += [0] * pad
            labels[i]    += [-100] * pad

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def load_data(data_dir: Path):
    files = {"train": data_dir / "train.jsonl"}
    v = data_dir / "val.jsonl"
    if v.exists():
        files["validation"] = v
    ds = load_dataset("json", data_files={k: str(p) for k, p in files.items()})
    return ds

def parse_args():
    p = argparse.ArgumentParser("LoRA fine-tune CharacterGLM-6B on Yuuka dialogues")
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-epochs", type=float, default=2.0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    return p.parse_args()

def main():
    args = parse_args()

    tok = get_tokenizer(args.model_name)
    model = load_model(args.model_name)
    ds = load_data(args.data_dir)
    collator = PromptCompletionCollator(tok, max_len=args.max_seq_len)

    has_val = "validation" in ds

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,   # 1*16 => 近似有效 batch 16
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        do_eval=has_val,                  # 避免旧版 eval_strategy 兼容问题
        fp16=True,
        bf16=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,      # 关键：保留 prompt/completion 列
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        tokenizer=tok,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        data_collator=collator,
    )

    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"[DONE] LoRA saved to {args.output_dir}")

if __name__ == "__main__":
    main()
