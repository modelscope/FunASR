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


## SpeechIO TIOBE
- Decode config 1: conf/decode_asr_transformer_noctc_1best.yaml
  - Decode without CTC
  - Decode without LM
- Decode config 2: conf/decode_asr_transformer_noctc_10best_lm_weight_0.15.yaml
  - Decode without CTC
  - Decode with Transformer-LM
  - LM weight: 0.15

| testset | w/o LM | w/ LM |
|:------------------:|:----:|:----:|
|SPEECHIO_ASR_ZH00001| 0.49 | 0.35 |
|SPEECHIO_ASR_ZH00002| 3.23 | 2.86 |
|SPEECHIO_ASR_ZH00003| 1.13 | 0.80 |
|SPEECHIO_ASR_ZH00004| 1.33 | 1.10 |
|SPEECHIO_ASR_ZH00005| 1.41 | 1.18 |
|SPEECHIO_ASR_ZH00006| 5.25 | 4.85 |
|SPEECHIO_ASR_ZH00007| 5.51 | 4.97 |
|SPEECHIO_ASR_ZH00008| 3.69 | 3.18 |
|SPEECHIO_ASR_ZH00009| 3.02 | 2.78 |
|SPEECHIO_ASR_ZH000010| 3.35 | 2.99 |
|SPEECHIO_ASR_ZH000011| 1.54 | 1.25 |
|SPEECHIO_ASR_ZH000012| 2.06 | 1.68 |
|SPEECHIO_ASR_ZH000013| 2.57 | 2.25 |
|SPEECHIO_ASR_ZH000014| 3.86 | 3.08 |
|SPEECHIO_ASR_ZH000015| 3.34 | 2.67 |
