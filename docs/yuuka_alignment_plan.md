# 优香角色模型调优方案

## 现状诊断

根据近期训练与推理反馈，主要存在以下问题：

1. **语言混杂**：样本中存在大量中英文混合表达，模型在推理时会随机切换语言。
2. **情绪盖过逻辑**：情绪提示注入比例偏高，导致回答中堆砌“心声”或夸张语气，削弱事实描述与推理链条。
3. **硬记忆引用失衡**：模型过度主动引用硬编码记忆，自行扩写设定，与用户语境脱节。
4. **训练/验证偏差**：训练集 loss 降至 0.19，但验证 loss 仍约 1.1，说明训练数据与验证数据或推理场景存在分布差异。

## 数据集设计优化

- **中文占比过滤**：`preprocess_data.py` 新增 `--min-completion-zh-ratio` 参数并默认 0.55，用于丢弃中文字符占比过低的样本，缓解语言混杂。【F:preprocess_data.py†L30-L43】【F:preprocess_data.py†L112-L137】
- **记忆模板抽样**：可通过 `--memory-file` 指定硬记忆库，脚本会按类别抽取角色要点、重要关系、行为约束、世界观与共同经历嵌入 persona 模板，避免模型临场自造设定。【F:preprocess_data.py†L45-L98】【F:preprocess_data.py†L139-L193】
- **情绪注入降权**：`--emotion-prob` 默认值由 0.25 降至 0.18，鼓励模型先给出客观分析，再附带心声类补充。【F:preprocess_data.py†L225-L262】
- **模板提示强化**：更新 persona few-shot，强调“默认使用中文”“先核对事实再表态”，缓解语气失控。【F:preprocess_data.py†L23-L29】

> 建议：首次重建数据时执行
>
> ```bash
> python preprocess_data.py \
>   --min-completion-zh-ratio 0.6 \
>   --emotion-prob 0.15 \
>   --persona-memory-count 4 \
>   --relationship-memory-count 3 \
>   --rule-memory-count 2 \
>   --world-memory-count 2 \
>   --memory-event-count 2
> ```
>
> 观察生成数据的 persona 样本比例是否维持在 25%-35% 左右，必要时再调节 `--persona-prob`。

## 数据质量巡检流程

- 新增 `scripts/data_audit.py`，用于统计样本数量、语言占比、persona/情绪样本比例及长度分布，可帮助定位导致过拟合或失衡的来源。【F:scripts/data_audit.py†L1-L151】
- 建议在每次重建数据后执行：
>
> ```bash
> python scripts/data_audit.py data/processed/train.jsonl
> python scripts/data_audit.py data/processed/val.jsonl
> ```
>
> 若发现 persona 样本不足、中文占比过低或情绪样本过多，可回溯 `preprocess_data.py` 参数或源对话。

## 训练逻辑建议

1. **对齐训练与验证分布**：验证集应使用同一套预处理参数生成，避免过拟合训练集特有的情绪模板或长回复。
2. **调节损失权重**：保留 `length_prior`，但可尝试降低 `repeat_penalty` 或针对 persona 样本设置更高采样权重，使模型学习在带设定提示时保持条理。
3. **逐阶段训练**：先用无 persona 的常规对话数据进行热身训练，再加载 persona 样本继续训练，可减少模型过度依赖系统提示。

## 推理提示词策略

- 默认 system prompt 可引用 `preprocess_data.py` 中的 Persona few-shot，与硬记忆摘要共同提供背景。
- 对长线对话，可将历史轮次压缩为“老师行动→优香总结→约束提醒”三段式摘要，再附在 prompt 前，确保模型在长上下文中保持逻辑。

## 后续验证指标

1. **语言一致性**：随机抽样 20 轮推理，记录中文输出比例和是否出现非上下文语言切换。
2. **逻辑完整度**：统计回答中包含“盘点→问题→补救→约束”四段结构的比例，目标 ≥70%。
3. **情绪平衡**：检查是否在需要安抚或担心场景才出现心声描述，并保持正文信息完整。
4. **记忆一致性**：构造跨多轮的世界观/角色关系问答，确保回答与 `hard_memory.jsonl` 一致，且不会无故新增设定。

## 总结

通过提高中文文本比重、结构化导入硬记忆、降低情绪模板权重以及加入数据诊断流程，可以让微调后的模型在保持优香口吻的同时具备更可靠的逻辑分析与记忆能力。后续可结合阶段化训练与系统提示优化进一步收敛验证 loss，并在推理环节稳固人设表现。
