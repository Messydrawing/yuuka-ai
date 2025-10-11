#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune ChatGLM3-6B (or CharacterGLM-6B) with QLoRA on Yuuka dialogues."""

from __future__ import annotations
import argparse
import os
import platform
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

# ---- Paths ----
DEFAULT_MODEL_NAME = "models/chatglm3-6b"        # 本地已解压好的 ChatGLM3-6B 目录
DEFAULT_DATA_DIR   = Path("data/processed")       # 里面有 train.jsonl / val.jsonl
DEFAULT_OUTPUT_DIR = Path("models/yuuka_glm_lora")
MAX_SEQ_LEN = 1024

def get_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True
    )
    # ChatGLM 通常有 eos_token，确保有 pad_token
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def guess_lora_target_modules(model) -> List[str]:
    """
    ChatGLM/GLM 常见 LoRA 位置：attention 的 qkv 融合层 & MLP 两个线性
    """
    names = [n for n, _ in model.named_modules()]
    glm_candidates = ["query_key_value", "dense_h_to_4h", "dense_4h_to_h"]
    if any(any(c in n for n in names) for c in glm_candidates):
        return glm_candidates
    # 兜底（Llama-like）
    return ["q_proj", "k_proj", "v_proj", "o_proj"]

def load_model(model_name: str):
    # 自动选择是否使用 safetensors（目录下如果存在 *.safetensors 就用）
    use_safetensors = any(p.suffix == ".safetensors"
                          for p in Path(model_name).glob("*.safetensors"))

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
        use_safetensors=use_safetensors,
        local_files_only=True,
        low_cpu_mem_usage=True,
        dtype=torch.float16,  # 代替旧的 torch_dtype
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    target_modules = guess_lora_target_modules(model)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model

# ---------------- Collator（手动分词与padding，绕开 encode/pad 兼容问题） ----------------
class PromptCompletionCollator:
    """
    把 prompt + completion 拼接；仅对 completion 计算loss。
    关键：不用 tokenizer.encode/encode_plus/pad，避免 ChatGLM 老 tokenizer 的 _pad 签名不兼容。
    """
    def __init__(self, tok, max_len: int):
        self.tok = tok
        self.max_len = max_len
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.eos_id = tok.eos_token_id

    def _encode_plain(self, text: str) -> List[int]:
        # 纯手动：tokenize -> ids
        if not text:
            return []
        toks = self.tok.tokenize(text)
        if not toks:
            return []
        ids = self.tok.convert_tokens_to_ids(toks)
        # 转 int，避免 torch.tensor dtype 问题
        return [int(x) for x in ids]

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        input_ids, attn, labels = [], [], []
        completion_lengths = []

        for ex in batch:
            prompt = ex["prompt"]
            completion = ex["completion"]

            p_ids = self._encode_plain(prompt)
            c_ids = self._encode_plain(completion) + ([self.eos_id] if self.eos_id is not None else [])

            ids = p_ids + c_ids
            lbl = [-100] * len(p_ids) + c_ids

            # 右侧截断（保留末尾）
            if len(ids) > self.max_len:
                ids = ids[-self.max_len:]
                lbl = lbl[-self.max_len:]

            mask = [1] * len(ids)
            input_ids.append(ids)
            attn.append(mask)
            labels.append(lbl)
            completion_lengths.append(sum(1 for tok in lbl if tok != -100))

        # 动态 padding 到当前 batch 的最长长度
        maxlen = max(len(x) for x in input_ids)
        for i in range(len(input_ids)):
            pad = maxlen - len(input_ids[i])
            if pad > 0:
                input_ids[i] += [self.pad_id] * pad
                attn[i]      += [0] * pad
                labels[i]    += [-100] * pad

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "completion_lengths": torch.tensor(completion_lengths, dtype=torch.float),
        }


class LengthAwareTrainer(Trainer):
    def __init__(self, *args, length_prior: float = 1.0,
                 repeat_penalty_weight: float = 0.0,
                 repeat_penalty_ngram: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.length_prior = max(length_prior, 0.0)
        self.repeat_penalty_weight = max(repeat_penalty_weight, 0.0)
        self.repeat_penalty_ngram = max(repeat_penalty_ngram, 2)

    @staticmethod
    def _count_repeated_ngrams(labels: torch.Tensor, n: int) -> torch.Tensor:
        counts = []
        for row in labels:
            toks = [int(t) for t in row.tolist() if t != -100]
            if len(toks) < n:
                counts.append(0.0)
                continue
            seen = set()
            repeated = 0
            for i in range(len(toks) - n + 1):
                ng = tuple(toks[i : i + n])
                if ng in seen:
                    repeated += 1
                else:
                    seen.add(ng)
            counts.append(float(repeated))
        return torch.tensor(counts, device=labels.device, dtype=torch.float)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = inputs.copy()
        inputs.pop("completion_lengths", None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        completion_lengths = inputs.pop("completion_lengths", None)
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(labels.size(0), -1)

        mask = (shift_labels != -100).float()
        per_sample_loss = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        if completion_lengths is not None:
            weights = completion_lengths.to(per_sample_loss.device)
            weights = weights.clamp_min(1.0)
            if self.length_prior and self.length_prior != 1.0:
                weights = torch.pow(weights, self.length_prior)
            weights = weights / weights.mean().clamp_min(1e-6)
        else:
            weights = torch.ones_like(per_sample_loss)

        loss = (per_sample_loss * weights).mean()

        if self.repeat_penalty_weight > 0:
            repeats = self._count_repeated_ngrams(labels, self.repeat_penalty_ngram)
            if repeats.numel():
                loss = loss + self.repeat_penalty_weight * repeats.mean()

        return (loss, outputs) if return_outputs else loss

# ---------------- Data ----------------
def load_data(data_dir: Path):
    files = {"train": data_dir / "train.jsonl"}
    v = data_dir / "val.jsonl"
    if v.exists():
        files["validation"] = v
    ds = load_dataset("json", data_files={k: str(p) for k, p in files.items()})
    return ds

def parse_args():
    p = argparse.ArgumentParser("QLoRA fine-tune ChatGLM3/CharacterGLM on Yuuka dialogues")
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-epochs", type=float, default=3.0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--length-prior", type=float, default=1.15,
                   help="completion 越长权重越高的指数（=1 表示不加权）")
    p.add_argument("--repeat-penalty", type=float, default=0.03,
                   help="对训练样本中重复 n-gram 的额外惩罚权重")
    p.add_argument("--repeat-ngram", type=int, default=4,
                   help="重复惩罚所检查的 n-gram 长度")
    return p.parse_args()

def main():
    args = parse_args()

    tok = get_tokenizer(args.model_name)
    model = load_model(args.model_name)
    ds = load_data(args.data_dir)
    collator = PromptCompletionCollator(tok, max_len=args.max_seq_len)

    has_val = "validation" in ds
    # Windows 上更稳：DataLoader 单线程
    num_workers = 0 if platform.system().lower().startswith("win") else 2

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,   # 1*16≈有效 batch 16（8GB 显存友好）
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        do_eval=has_val,                  # 兼容旧版：不显式用 evaluation_strategy
        fp16=True,
        bf16=False,
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,      # 保留 prompt/completion 列
        report_to=[],
    )

    trainer = LengthAwareTrainer(
        model=model,
        tokenizer=tok,  # FutureWarning 可忽略；或改用 processing_class
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        data_collator=collator,
        length_prior=args.length_prior,
        repeat_penalty_weight=args.repeat_penalty,
        repeat_penalty_ngram=args.repeat_ngram,
    )

    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"[DONE] LoRA saved to {args.output_dir}")

if __name__ == "__main__":
    main()
