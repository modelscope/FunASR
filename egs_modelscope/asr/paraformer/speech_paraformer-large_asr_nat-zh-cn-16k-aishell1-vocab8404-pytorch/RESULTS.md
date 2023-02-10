# Paraformer-Large
- Model link: <https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-aishell1-vocab8404-pytorch/summary>
- Model size: 220M

# Environments
- date: `Fri Feb 10 13:34:24 CST 2023`
- python version: `3.7.12`
- FunASR version: `0.1.6`
- pytorch version: `pytorch 1.7.0`
- Git hash: ``
- Commit date: ``

# Beachmark Results

## AISHELL-1
- Decode config:
  - Decode without CTC
  - Decode without LM

| testset CER(%) | base model|finetune model |
|:--------------:|:---------:|:-------------:|
| dev            | 1.75      |1.62           |
| test           | 1.95      |1.78           |
