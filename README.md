# Yuuka AI Fine-tuning Toolkit

用于构建、诊断与微调基于 ChatGLM3/CharacterGLM 的“早濑优香”角色模型的工具集。

## 目录结构

- `preprocess_data.py`：将原始对话 JSONL 生成 prompt/completion 样本，支持 persona、情绪与硬记忆注入。
- `train_lora.py`：基于 QLoRA 进行微调的训练脚本。
- `scripts/data_audit.py`：数据质量巡检脚本，统计语言占比、模板比例等指标。
- `hard_memory.jsonl`：角色硬记忆/世界观素材，可在预处理阶段抽样注入。
- `docs/`：调优方案与流程建议。

## 数据预处理

```bash
python preprocess_data.py \
  --input data/yuuka_dialogues_zh.jsonl \
  --train-output data/processed/train.jsonl \
  --val-output data/processed/val.jsonl
```

常用参数：

- `--min-completion-zh-ratio`：丢弃中文占比过低的样本（默认 0.55）。
- `--memory-file`：指定硬记忆库，默认为 `hard_memory.jsonl`，可设为空字符串关闭。
- `--persona-memory-count`/`--relationship-memory-count` 等：控制每类记忆采样条数。
- `--emotion-prob`：情绪心声样本占比，默认 0.18。

更详细的调参建议见《docs/yuuka_alignment_plan.md》。

## 数据质量巡检

```bash
python scripts/data_audit.py data/processed/train.jsonl
```

脚本会输出 persona 模板占比、情绪样本比例、中文/字母占比及长度分布，用于快速判断数据是否偏离目标风格。

## 训练

```bash
python train_lora.py \
  --model-name models/chatglm3-6b \
  --data-dir data/processed \
  --output-dir models/yuuka_glm_lora
```

- 默认使用 4bit QLoRA，batch=1、grad_accum=16，适合 8GB 显存。
- `--length-prior` 与 `--repeat-penalty` 可用于调节长回复与重复 n-gram 的损失权重。

## 参考

- [docs/yuuka_alignment_plan.md](docs/yuuka_alignment_plan.md)：针对语言混杂、情绪失衡、记忆引用等问题的系统性调优方案。
