# Paraformer-Large
- Model link: <https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary>
- Model size: 220M
- Train config: conf/train_asr_paraformer_sanm_50e_16d_2048_512_lfr6.yaml

# Environments
- date: `Tue Nov 22 18:48:39 CST 2022`
- python version: `3.7.12`
- FunASR version: `0.1.0`
- pytorch version: `pytorch 1.7.0`
- Git hash: ``
- Commit date: ``

# Beachmark Results

## AISHELL-1
- Decode config: conf/decode_asr_transformer_noctc_1best.yaml
  - Decode without CTC
  - Decode without LM

| testset   | CER(%)|
|:---------:|:-----:|
| dev       | 1.75  |
| test      | 1.95  |
