
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进版训练脚本：微调“早濑优香”多轮对话角色模型
- 目标：给定 system 人设 + 对话历史（含多轮），预测下一轮优香回复（要求含 [SCENE=...][MOOD=...] 前缀）
- 样本构造：<|system|>人设+约束 + {带 role 标签的历史}<|assistant|> -> completion 为优香该轮回复
- 训练：仅对 completion 计算损失；保留数值稳定性设置；附加关键 token 加权；监控 SCENE/MOOD 学习情况
- 兼容数据：每行形如 {"messages": [{"role": "...", "content": "..."}, ...]}
"""

import argparse
import platform
import inspect
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

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

# ===== 默认参数 =====
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("models/yuuka_qwen_lora")
MAX_SEQ_LEN = 1024

# ===== 工具函数 =====
def bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def parse_token_weights(weight_str: str) -> Dict[str, float]:
    """解析 token 权重字符串，格式: token1:weight1,token2:weight2"""
    if not weight_str:
        return {}
    weights = {}
    for pair in weight_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" in pair:
            token, weight = pair.split(":", 1)
        elif "=" in pair:
            token, weight = pair.split("=", 1)
        else:
            token, weight = pair, "1.0"
        try:
            weights[token.strip()] = float(weight.strip())
        except ValueError:
            print(f"[WARN] 无法解析权重 '{pair}'，已跳过。")
    return weights

# ===== Tokenizer / Model =====
def get_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def guess_lora_target_modules(model) -> List[str]:
    names = [n for n, _ in model.named_modules()]
    glm_patterns = ["query_key_value", "dense_h_to_4h", "dense_4h_to_h"]
    if any(any(p in n for n in names) for p in glm_patterns):
        return glm_patterns
    # 降显存可只保留 q/k/v/o
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def load_model(model_name: str):
    model_path = Path(model_name)
    use_safetensors = any(p.suffix == ".safetensors" for p in model_path.glob("*")) if model_path.is_dir() else False

    compute_dtype = torch.bfloat16 if bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config,
        use_safetensors=use_safetensors,
        local_files_only=False,
        low_cpu_mem_usage=True,
        torch_dtype=compute_dtype,
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
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

# ===== Data Collator =====
class PromptCompletionCollator:
    """拼接 prompt+completion，仅对 completion 计算 loss；右侧 padding；末尾追加 EOS"""
    def __init__(self, tokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.eos_id = tokenizer.eos_token_id

    def _encode(self, text: str) -> List[int]:
        return [] if not text else self.tok.encode(text, add_special_tokens=False)

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        input_ids, masks, labels_list, comp_lens = [], [], [], []
        for ex in batch:
            pid = self._encode(ex["prompt"])
            cid = self._encode(ex["completion"])
            if self.eos_id is not None:
                cid = cid + [self.eos_id]
            ids = pid + cid
            lbl = [-100] * len(pid) + cid.copy()

            if len(ids) > self.max_len:
                ids = ids[-self.max_len:]
                lbl = lbl[-self.max_len:]

            mask = [1] * len(ids)
            comp_len = sum(1 for x in lbl if x != -100)
            input_ids.append(ids); masks.append(mask); labels_list.append(lbl); comp_lens.append(comp_len)

        max_len = max(len(x) for x in input_ids)
        pad_ids, pad_masks, pad_labels = [], [], []
        for ids, mk, lbl in zip(input_ids, masks, labels_list):
            pad = max_len - len(ids)
            pad_ids.append(ids + [self.pad_id]*pad)
            pad_masks.append(mk + [0]*pad)
            pad_labels.append(lbl + [-100]*pad)

        return {
            "input_ids": torch.tensor(pad_ids, dtype=torch.long),
            "attention_mask": torch.tensor(pad_masks, dtype=torch.long),
            "labels": torch.tensor(pad_labels, dtype=torch.long),
            "completion_lengths": torch.tensor(comp_lens, dtype=torch.float),
        }


# ===== Custom Trainer =====
class CustomTrainer(Trainer):
    def __init__(self, *args, length_prior: float = 1.0,
                 token_weights: Optional[Dict[str, float]] = None,
                 label_smoothing: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.length_prior = max(length_prior, 0.0)
        # 修复 tokenizer 获取方式
        self.processing_class = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if self.processing_class is None:
            # 尝试从 args 中获取
            for arg in args:
                if hasattr(arg, 'tokenizer') or hasattr(arg, 'processing_class'):
                    self.processing_class = getattr(arg, 'processing_class', getattr(arg, 'tokenizer', None))
                    break

        if self.processing_class is None:
            raise RuntimeError("Tokenizer/ProcessingClass is required for CustomTrainer")

        self.model_vocab_size = self.model.get_output_embeddings().weight.shape[0]
        weight_tensor = torch.ones(self.model_vocab_size)
        if token_weights:
            for token_str, factor in token_weights.items():
                if not token_str:
                    continue
                for tid in self.processing_class.encode(token_str, add_special_tokens=False):
                    if 0 <= tid < self.model_vocab_size:
                        weight_tensor[tid] = max(weight_tensor[tid].item(), float(factor))
        self.token_weight_tensor = weight_tensor

        ce_sig = inspect.signature(torch.nn.CrossEntropyLoss.__init__)
        self.label_smoothing = max(label_smoothing, 0.0) if "label_smoothing" in ce_sig.parameters else 0.0

        self._scene_mood_regex = re.compile(r"\[SCENE=[^\]]+\]\s*\[MOOD=[^\]]+\]")
        self._training_stats = {"total_steps": 0, "scene_mood_match_rate": 0.0}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """修复版本：兼容新版本 transformers 的 compute_loss 签名"""
        # 从 kwargs 中获取可能的 num_items_in_batch 参数（如果有）
        _ = kwargs.get("num_items_in_batch", None)

        lengths = inputs.pop("completion_lengths", None)
        labels = inputs.pop("labels", None)
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

        # ===== 参考 NLL / PPL + SCENE/MOOD 监控 =====
        with torch.no_grad():
            lsm = torch.log_softmax(shift_logits.float(), dim=-1)
            valid = (shift_labels != -100)
            safe_labels = shift_labels.clone()
            safe_labels[~valid] = 0

            nll_tok = -lsm.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            nll_tok = nll_tok * valid.float()
            ref_nll = (nll_tok.sum(dim=1) / valid.float().sum(dim=1).clamp_min(1.0)).mean()
            ref_ppl = torch.exp(ref_nll)

            # 训练进度统计（每步指数滑动平均）
            self._training_stats["total_steps"] += 1
            batch_size = shift_labels.size(0)
            scene_mood_matches = 0
            for i in range(min(batch_size, 4)):  # 采样检查前 4 条
                if valid[i].sum() > 0:
                    one = safe_labels[i][valid[i]].tolist()
                    decoded = self.processing_class.decode(one[:128], skip_special_tokens=True)
                    if self._scene_mood_regex.search(decoded):
                        scene_mood_matches += 1
            current_rate = scene_mood_matches / max(batch_size, 1)
            self._training_stats["scene_mood_match_rate"] = (
                    0.9 * self._training_stats["scene_mood_match_rate"] + 0.1 * current_rate
            )

            if self._training_stats["total_steps"] % 100 == 0:
                try:
                    self.log({
                        "scene_mood_match_rate": float(self._training_stats["scene_mood_match_rate"]),
                        "ref_nll": float(ref_nll.item()),
                        "ref_ppl": float(ref_ppl.item()),
                    })
                except Exception:
                    pass

        return (loss, outputs) if return_outputs else loss

# ===== 参数解析 =====
def parse_args():
    p = argparse.ArgumentParser(description="微调“早瀬优香”角色模型 (QLoRA, Qwen2.5-7B)")
    p.add_argument("--model-name", type=str, default=str(DEFAULT_MODEL_NAME))
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--data-file", type=Path, default=None, help="可选：直接指定单一 jsonl 文件（含 messages 字段）")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-epochs", type=float, default=3.0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--max-history-turns", type=int, default=4,
                   help="用于构造样本的最大历史 user 轮数（每轮含 user+assistant），None 表示不限制")
    p.add_argument("--persona-file", type=Path, default=None, required=True,
                   help="系统角色人设文本文件路径（persona_yuuka.txt）")
    p.add_argument("--token-weights", type=str, default="",
                   help="格式: token:weight,token2:weight2")
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--length-prior", type=float, default=1.0, help=">1.0 轻度偏好较长 completion")
    return p.parse_args()

# ===== TrainingArguments =====
def make_training_args(args, has_val: bool, num_workers: int):
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters
    kw = dict(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=2,          # 适度增大
        gradient_accumulation_steps=8,          # 相应调整
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.10,                      # 增加 warmup
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=3,
        bf16=bf16_supported(), fp16=not bf16_supported(),
        max_grad_norm=0.5,                      # 更温和的裁剪
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,
        weight_decay=0.01,
        adam_epsilon=1e-8,
    )
    # 评估省显存
    if "per_device_eval_batch_size" in params:
        kw["per_device_eval_batch_size"] = 1
    if "eval_accumulation_steps" in params:
        kw["eval_accumulation_steps"] = 1
    if "prediction_loss_only" in params:
        kw["prediction_loss_only"] = True
    if "gradient_checkpointing" in params:
        kw["gradient_checkpointing"] = True
    if "optim" in params:
        kw["optim"] = "paged_adamw_8bit"
    if "report_to" in params:
        kw["report_to"] = []
    if "label_smoothing_factor" in params:
        kw["label_smoothing_factor"] = max(0.0, float(args.label_smoothing))

    # 评估策略
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

# ===== 对话 -> 样本（修复后的历史构建 + role tag + 优化 system prompt） =====
def conversation_to_examples(persona_text: str, messages: List[Dict], max_turns: Optional[int]):
    """
    遍历对话，针对每个 assistant 回复，构造一个样本：
    prompt: <|system|>{人设+规则}\n{历史(含role标签)}<|assistant|>
    completion: 当前 assistant 回复（需含 SCENE/MOOD）
    max_turns: 限制向前追溯的 user 轮数（每轮≈ user+assistant），None 不限
    """
    examples = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        if i == 0 or messages[i - 1].get("role") != "user":
            continue

        # 由当前位置向前收集历史，限定最多包含 max_turns 个 user 轮
        history_msgs: List[Dict] = []
        turns_count = 0
        j = i - 1
        while j >= 0:
            m = messages[j]
            history_msgs.insert(0, m)  # 插到开头保持顺序
            if m.get("role") == "user":
                turns_count += 1
            j -= 1
            if max_turns is not None and turns_count >= max_turns:
                break

        # 构造带 role tag 的历史文本
        history_text = []
        for m in history_msgs:
            role = m.get("role", "").strip()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                history_text.append(f"<|user|>{content}")
            elif role == "assistant":
                history_text.append(f"<|assistant|>{content}")
            elif role == "system":
                continue
            else:
                history_text.append(f"<|user|>{content}")
        history_text = "\n".join(history_text).strip()

        # —— 优化后的 system 提示模板 ——
        sys_prompt = (
            "你是早濑优香，千年科学学院的学生会会计。请严格遵循人物设定，"
            "并在每次回复开头显式输出 [SCENE=场景名][MOOD=情绪] 标签。\n\n"
            "人物设定：\n"
            f"{persona_text}\n\n"
            "回复规则：\n"
            "1. 开头必须包含 [SCENE=...][MOOD=...]\n"
            "2. 保持傲娇但负责的性格\n"
            "3. 对预算管理严格，对老师关心但嘴硬\n\n"
            "对话记录："
        )

        # 组装最终 prompt
        if history_text:
            prompt = f"<|system|>{sys_prompt}\n{history_text}\n<|assistant|>"
        else:
            prompt = f"<|system|>{sys_prompt}\n<|assistant|>"

        completion = (msg.get("content") or "").strip()
        examples.append({"prompt": prompt, "completion": completion})

    return examples

# ===== 主流程 =====
def main():
    args = parse_args()
    tokenizer = get_tokenizer(args.model_name)
    model = load_model(args.model_name)

    # 注册特殊 token，并扩充词表
    special_tokens = {"additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # 数据文件解析：优先 --data-file；否则 data_dir/{train,val}.jsonl；最后回退到 data_dir/trainer.jsonl
    data_files = {}
    if args.data_file and args.data_file.exists():
        data_files["train"] = str(args.data_file)
    else:
        train1 = args.data_dir / "train.jsonl"
        train2 = args.data_dir / "trainer.jsonl"
        if train1.exists():
            data_files["train"] = str(train1)
        elif train2.exists():
            data_files["train"] = str(train2)
        else:
            fallback = Path("/mnt/data/trainer.jsonl")
            if fallback.exists():
                data_files["train"] = str(fallback)
            else:
                raise FileNotFoundError("未找到训练数据文件。请提供 --data-file 或在 data_dir 下放置 train.jsonl / trainer.jsonl")

        val_path = args.data_dir / "val.jsonl"
        if val_path.exists():
            data_files["validation"] = str(val_path)

    raw_ds = load_dataset("json", data_files=data_files)

    # 读取 Persona 文本
    persona_path = args.persona_file if args.persona_file else Path("/mnt/data/persona_yuuka.txt")
    if not persona_path.exists():
        raise FileNotFoundError("Persona file not found: " + str(persona_path))
    persona_text = persona_path.read_text(encoding="utf-8").strip()

    # 数据预处理：生成 prompt/completion，并加强验证与统计
    def preprocess_split(examples):
        all_prompts, all_completions = [], []
        msgs_list = examples.get("messages", [])
        total_skipped = 0

        for idx, msgs in enumerate(msgs_list):
            # 验证消息格式
            if not isinstance(msgs, list) or len(msgs) < 2:
                total_skipped += 1
                continue

            conv_examples = conversation_to_examples(persona_text, msgs, args.max_history_turns)
            for item in conv_examples:
                # 更严格的 SCENE/MOOD 验证
                if not re.search(r"\[SCENE=[^\]]+\]\s*\[MOOD=[^\]]+\]", item["completion"]):
                    total_skipped += 1
                    continue
                all_prompts.append(item["prompt"])
                all_completions.append(item["completion"])

        if total_skipped > 0:
            print(f"[INFO] 跳过 {total_skipped} 个无效样本")
        return {"prompt": all_prompts, "completion": all_completions}

    # 移除原 messages 列，得到纯 prompt/completion 数据集
    remove_cols = []
    for split, ds in raw_ds.items():
        for col in ds.column_names:
            if col != "messages":
                remove_cols.append(col)
        break
    dataset = raw_ds.map(preprocess_split, batched=True, remove_columns=remove_cols + ["messages"])

    # 关键词权重解析（若未提供则给默认优香风格权重）
    token_weights_map = parse_token_weights(args.token_weights)
    if not token_weights_map:
        token_weights_map = {
            "[SCENE": 1.25, "[MOOD": 1.25, "老师": 1.12,
            "才不是": 1.05, "真是的": 1.10, "哼": 1.10, "笨蛋": 1.05, "唔": 1.05
        }
        print(f"[INFO] 使用默认 token 权重: {token_weights_map}")

    collator = PromptCompletionCollator(tokenizer, max_len=args.max_seq_len)
    has_val = "validation" in dataset
    num_workers = 0 if platform.system().lower().startswith("win") else 2
    training_args = make_training_args(args, has_val=has_val, num_workers=num_workers)

    trainer_sig = inspect.signature(Trainer.__init__)
    base_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=collator,
    )

    # 使用 processing_class 而不是 tokenizer
    if "processing_class" in trainer_sig.parameters:
        trainer = CustomTrainer(
            processing_class=tokenizer,
            length_prior=args.length_prior,
            token_weights=token_weights_map,
            label_smoothing=args.label_smoothing,
            **base_kwargs
        )
    else:
        # 回退到 tokenizer 参数
        trainer = CustomTrainer(
            tokenizer=tokenizer,
            length_prior=args.length_prior,
            token_weights=token_weights_map,
            label_smoothing=args.label_smoothing,
            **base_kwargs
        )

    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[DONE] LoRA 模型已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
