# Yuuka AI Fine-tuning Toolkit

![效果图](yuuka-ai.jpg)

## 训练

```bash
python train_lora.py \
  --model-name models/Qwen2.5-7B-Instruct \
  --persona-file memory/persona_yuuka.txt \
  --data-dir data/processed \
  --output-dir models/yuuka_qwen_lora \
  --num-epochs 2.5 \
  --lr 3e-5 \
  --max-seq-len 1536 \
  --max-history-turns 4 \
  --token-weights "SCENE:1.3, MOOD:1.3, 老师:1.1" \
  --label-smoothing 0.03 \
  --length-prior 0.95
```
- model-name 是基座模型所在目录，如果电脑支持huggingface的连接，可以改为Qwen/Qwen2.5-7B-Instruct，否则请将该模型下载到本地后，将模型所在目录放置于此
- persona-file 是人设提示词，可以编辑。
- data-dir 是数据集和验证集所在目录，可以编辑其中内容。
- output-dir 是最终微调出来的参数所在位置，是训练的核心结果。
- token-weights 是需要特别学习的输出关键词，这里SCENE和MOOD是输出格式需要，老师是示范。
- lr 是学习率，num-epoches 是训练轮数，太多会过拟合，太少会没学到。
- 其他是训练相关的参数，可以酌情考虑调整。

## 使用

如果你没有进行训练直接准备使用，请先运行脚本`download_lora.py`将微调参数下载到本地目录./models中
```bash
 python yuuka_runtime.py  --base models/Qwen2.5-7B-Instruct --lora models/yuuka_qwen_lora11 --persona memory/persona_yuuka.txt 
```
- base是基座模型目录。
- lora是微调参数目录。
- persona是人设所在目录。
