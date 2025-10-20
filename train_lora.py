#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本：使用 QLoRA 微调 Qwen2.5-7B-Instruct 以拟合“优香”角色风格。

特性
- 基座：Qwen/Qwen2.5-7B-Instruct（4bit QLoRA）。
- 兼容 transformers 多版本：evaluation_strategy / eval_strategy / evaluate_during_training / do_eval 自适应。
- 自定义 Trainer：token 级加权、长度偏好、（若可用则）label smoothing。
- Collator：以 chat template 方式拼接 messages，仅对最终 assistant 回复计算 loss。
- LoRA 目标自动适配（q/k/v/o + gate/up/down_proj；兼容 GLM 命名）。
- Windows 友好：dataloader_num_workers=0；bitsandbytes 4bit 已配置。
"""

import argparse
import platform
import inspect
from pathlib import Path
from typing import List, Dict, Optional, Any

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

# ===================== 默认参数 =====================
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("models/yuuka_qwen_lora")
MAX_SEQ_LEN = 1024


# ===================== Tokenizer / Model =====================
def get_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=False,
    )
    # 设置 pad_token
    if tok.pad_token is None:
        if tok.eos_token:
            tok.pad_token = tok.eos_token
        else:
            tok.pad_token = "<pad>"
            tok._add_tokens([tok.pad_token])
    tok.padding_side = "right"
    return tok


def guess_lora_target_modules(model) -> List[str]:
    """自动推断 LoRA 注入层；兼容 Qwen/LLaMA 与 GLM 命名。"""
    names = [n for n, _ in model.named_modules()]
    glm_patterns = ["query_key_value", "dense_h_to_4h", "dense_4h_to_h"]
    if any(any(pat in name for name in names) for pat in glm_patterns):
        return glm_patterns
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_model(model_name: str):
    model_path = Path(model_name)
    use_safetensors = any(p.suffix == ".safetensors" for p in model_path.glob("*")) if model_path.is_dir() else False

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config,
        use_safetensors=use_safetensors,
        local_files_only=False,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,  # 新版会提示用 dtype；保留以兼容旧版
    )

    model = prepare_model_for_kbit_training(model)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    target_modules = guess_lora_target_modules(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ===================== Data Collator =====================
class ChatTemplateCollator:
    """使用 chat template 构造输入，仅对最后一条 assistant 回复计算 loss。"""

    def __init__(self, tokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.chat_template = getattr(tokenizer, "chat_template", None)

    def _encode_text(self, text: str) -> List[int]:
        if not text:
            return []
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _build_prompt_ids(self, messages: List[Dict[str, str]]) -> List[int]:
        if not messages:
            raise ValueError("每条样本至少需要一条上下文消息（非 assistant）。")
        normalized = []
        for msg in messages:
            role = str(msg.get("role") or "")
            content = str(msg.get("content") or "")
            normalized.append({"role": role, "content": content})
        prompt_text = self.tok.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=self.chat_template,
        )
        return self._encode_text(prompt_text)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_id_list, attention_mask_list, labels_list, completion_lengths = [], [], [], []

        for ex in batch:
            messages = ex.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("样本缺少 messages 字段或为空。")
            last_msg = messages[-1]
            if str(last_msg.get("role")) != "assistant":
                raise ValueError("最后一条消息必须是 assistant 回复。")

            context_messages = messages[:-1]
            prompt_ids = self._build_prompt_ids(context_messages)
            completion_ids = self._encode_text(str(last_msg.get("content") or ""))
            if self.eos_id is not None:
                completion_ids = completion_ids + [self.eos_id]

            ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids.copy()

            if len(ids) > self.max_len:
                ids = ids[-self.max_len:]
                labels = labels[-self.max_len:]

            mask = [1] * len(ids)
            comp_len = sum(1 for x in labels if x != -100)

            input_id_list.append(ids)
            attention_mask_list.append(mask)
            labels_list.append(labels)
            completion_lengths.append(comp_len)

        max_length = max(len(x) for x in input_id_list) if input_id_list else 0
        padded_ids, padded_mask, padded_labels = [], [], []
        for ids, mask, labels in zip(input_id_list, attention_mask_list, labels_list):
            pad_len = max_length - len(ids)
            padded_ids.append(ids + [self.pad_id] * pad_len)
            padded_mask.append(mask + [0] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "completion_lengths": torch.tensor(completion_lengths, dtype=torch.float),
        }


# ===================== 自定义 Trainer =====================
class CustomTrainer(Trainer):
    def __init__(
        self,
        *args,
        length_prior: float = 1.0,
        token_weights: Optional[Dict[str, float]] = None,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.length_prior = max(length_prior, 0.0)

        # 兼容 tokenizer / processing_class
        self._proc = getattr(self, "tokenizer", None) or getattr(self, "processing_class", None)
        if self._proc is None:
            raise RuntimeError("Neither tokenizer nor processing_class is available for CustomTrainer.")

        # 词表权重（关键词加权）
        vocab_size = len(self._proc)
        weight_tensor = torch.ones(vocab_size)
        if token_weights:
            for token_str, factor in token_weights.items():
                if not token_str:
                    continue
                ids = self._proc.encode(token_str, add_special_tokens=False)
                for tid in ids:
                    if 0 <= tid < vocab_size:
                        weight_tensor[tid] = max(weight_tensor[tid], float(factor))
        # 普通属性保存（Trainer 不是 nn.Module）
        self.token_weight_tensor = weight_tensor

        # label smoothing（若当前 PyTorch 支持则启用）
        ce_sig = inspect.signature(torch.nn.CrossEntropyLoss.__init__)
        if "label_smoothing" in ce_sig.parameters:
            self.label_smoothing = max(label_smoothing, 0.0)
        else:
            self.label_smoothing = 0.0

    # 关键修复：labels 用 pop() 取出，避免重复传参；同时 **kwargs 兼容 num_items_in_batch 等新参
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        lengths = inputs.pop("completion_lengths", None)
        labels = inputs.pop("labels", None)  # ← 这里用 pop
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous() if labels is not None else None
        vocab_size = shift_logits.size(-1)

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.token_weight_tensor.to(shift_logits.device),
            ignore_index=-100,
            reduction="none",
            label_smoothing=self.label_smoothing if self.label_smoothing > 0 else 0.0,
        )
        token_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        token_loss = token_loss.view(shift_labels.size(0), -1)

        mask = (shift_labels != -100).float()
        per_sample_loss = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        if lengths is not None and self.length_prior != 1.0:
            lengths = lengths.to(per_sample_loss.device).clamp_min(1.0)
            weights = torch.pow(lengths, self.length_prior)
            weights = weights / weights.mean().clamp_min(1e-6)
            per_sample_loss = per_sample_loss * weights

        loss = per_sample_loss.mean()
        return (loss, outputs) if return_outputs else loss


# ===================== Argparse =====================
def parse_args():
    p = argparse.ArgumentParser(description="使用 QLoRA 微调 Qwen2.5-7B-Instruct（优香 Persona 风格）")
    p.add_argument("--model-name", type=str, default=str(DEFAULT_MODEL_NAME), help="预训练模型名称或路径")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="训练数据目录，应包含 train.jsonl 和 val.jsonl")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="LoRA 权重输出目录")
    p.add_argument("--num-epochs", type=float, default=3.0, help="训练 epoch 数")
    p.add_argument("--logging-steps", type=int, default=10, help="日志记录间隔")
    p.add_argument("--lr", type=float, default=5e-5, help="学习率")
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN, help="最大序列长度")
    p.add_argument("--length-prior", type=float, default=1.0, help="长度偏好指数（>1 偏好更长回答）")
    p.add_argument("--persona-prefix", type=str, default=None, help="可选：追加到对话的 persona 文本")
    p.add_argument("--persona-file", type=Path, default=None, help="可选：人设文本文件；与 persona-prefix 互斥")
    p.add_argument("--persona-apply", choices=["prompt", "completion", "both"], default="prompt", help="人设注入位置（prompt=system 提示，completion=拼接到最终回答）")
    p.add_argument("--persona-separator", type=str, default="\n\n", help="人设与原对话之间的分隔符")
    p.add_argument("--token-weights", type=str, default="", help='关键词权重，如 "预算=1.5,报销=1.5"')
    p.add_argument("--label-smoothing", type=float, default=0.02, help="若本地 PyTorch 支持则启用")
    return p.parse_args()


# ===================== TrainingArguments 兼容层 =====================
def make_training_args(args, has_val: bool, num_workers: int):
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    kw = dict(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        bf16=False,
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,
    )

    if "gradient_checkpointing" in params:
        kw["gradient_checkpointing"] = True
    if "optim" in params:
        kw["optim"] = "paged_adamw_8bit"
    if "report_to" in params:
        kw["report_to"] = []
    if "label_smoothing_factor" in params and args.label_smoothing > 0:
        kw["label_smoothing_factor"] = args.label_smoothing

    # 评估策略（多版本兼容）
    if has_val:
        if "evaluation_strategy" in params:
            kw["evaluation_strategy"] = "epoch"
        elif "eval_strategy" in params:
            kw["eval_strategy"] = "epoch"
        elif "evaluate_during_training" in params:
            kw["evaluate_during_training"] = True
        elif "do_eval" in params:
            kw["do_eval"] = True
    else:
        if "evaluation_strategy" in params:
            kw["evaluation_strategy"] = "no"
        elif "eval_strategy" in params:
            kw["eval_strategy"] = "no"
        elif "evaluate_during_training" in params:
            kw["evaluate_during_training"] = False
        elif "do_eval" in params:
            kw["do_eval"] = False

    return TrainingArguments(**kw)


# ===================== 主流程 =====================
def main():
    args = parse_args()

    tokenizer = get_tokenizer(args.model_name)
    model = load_model(args.model_name)

    data_files = {"train": str(args.data_dir / "train.jsonl")}
    val_path = args.data_dir / "val.jsonl"
    if val_path.exists():
        data_files["validation"] = str(val_path)
    dataset = load_dataset("json", data_files=data_files)

    # 可选：追加 persona 文本
    persona_text: Optional[str] = None
    if args.persona_file and Path(args.persona_file).exists():
        persona_text = Path(args.persona_file).read_text(encoding="utf-8").strip()
    elif args.persona_prefix:
        persona_text = args.persona_prefix.strip()

    if persona_text:
        print("[INFO] 在数据集中追加 Persona 文本以强化语气/口癖")
        sep = args.persona_separator

        def _add_persona(ex):
            msgs = [dict(m) for m in ex.get("messages", [])]
            if not msgs:
                return ex

            if args.persona_apply in {"prompt", "both"}:
                if msgs[0].get("role") == "system":
                    merged = f"{persona_text}{sep}{msgs[0].get('content', '')}".strip()
                    msgs[0] = {"role": "system", "content": merged}
                else:
                    msgs = [{"role": "system", "content": persona_text}] + msgs

            if args.persona_apply in {"completion", "both"} and msgs:
                last = dict(msgs[-1])
                if last.get("role") == "assistant":
                    last["content"] = f"{persona_text}{sep}{last.get('content', '')}"
                    msgs[-1] = last

            new_ex = dict(ex)
            new_ex["messages"] = msgs
            return new_ex

        dataset = dataset.map(_add_persona)

    # 解析关键词权重
    token_weights_map: Dict[str, float] = {}
    if args.token_weights:
        for pair in args.token_weights.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" in pair:
                key, val = pair.split("=", 1)
            elif ":" in pair:
                key, val = pair.split(":", 1)
            else:
                key, val = pair, "1.0"
            key = key.strip()
            try:
                factor = float(val)
            except ValueError:
                factor = 1.0
            if key:
                token_weights_map[key] = factor
        if token_weights_map:
            print(f"[INFO] 关键词加权: {token_weights_map}")

    collator = ChatTemplateCollator(tokenizer, max_len=args.max_seq_len)
    has_val = "validation" in dataset
    num_workers = 0 if platform.system().lower().startswith("win") else 2
    training_args = make_training_args(args, has_val=has_val, num_workers=num_workers)

    # 兼容新版的 processing_class 与旧版的 tokenizer 参数
    trainer_sig = inspect.signature(Trainer.__init__)
    base_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=collator,
        length_prior=args.length_prior,
        token_weights=token_weights_map,
        label_smoothing=args.label_smoothing,
    )
    if "processing_class" in trainer_sig.parameters:
        trainer = CustomTrainer(processing_class=tokenizer, **base_kwargs)
    else:
        trainer = CustomTrainer(tokenizer=tokenizer, **base_kwargs)

    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[DONE] LoRA 模型已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
