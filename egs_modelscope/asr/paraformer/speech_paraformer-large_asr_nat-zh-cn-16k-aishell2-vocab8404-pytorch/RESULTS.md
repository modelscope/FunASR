# Paraformer-Large
- Model link: <https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-aishell2-vocab8404-pytorch/summary>
- Model size: 220M

# Environments
- date: `Fri Feb 10 13:34:24 CST 2023`
- python version: `3.7.12`
- FunASR version: `0.1.6`
- pytorch version: `pytorch 1.7.0`
- Git hash: ``
- Commit date: ``

# Beachmark Results

## AISHELL-2
- Decode config: 
  - Decode without CTC
  - Decode without LM

| testset      | base model|finetune model|
|:------------:|:---------:|:------------:|
| dev_ios      | 2.80      |2.60          |
| test_android | 3.13      |2.84          |
| test_ios     | 2.85      |2.82          |
| test_mic     | 3.06      |2.88          |
