# Pretrained Models on Huggingface

## Model License
-  Apache License 2.0

## Model Zoo
Here we provided several pretrained models on different datasets. The details of models and datasets can be found on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition).

### Speech Recognition Models
#### Paraformer Models

|                               Model Name                                | Language |           Training Data            | Vocab Size | Parameter | Offline/Online | Notes                                                                                                                           |
|:-----------------------------------------------------------------------:|:--------:|:----------------------------------:|:----------:|:---------:|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------|
| [Paraformer-large](https://huggingface.co/funasr/paraformer-large)      | CN & EN  | Alibaba Speech Data (60000hours)   |    8404    |   220M    |    Offline     | Duration of input wav <= 20s                                                                                                    |

[//]: # (| [Paraformer-large-long]&#40;https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary&#41; | CN & EN  | Alibaba Speech Data &#40;60000hours&#41; |    8404    |   220M    |    Offline     | Which ould deal with arbitrary length input wav                                                                                 |)

[//]: # (| [paraformer-large-contextual]&#40;https://www.modelscope.cn/models/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/summary&#41; | CN & EN  | Alibaba Speech Data &#40;60000hours&#41; |    8404    |   220M    |    Offline     | Which supports the hotword customization based on the incentive enhancement, and improves the recall and precision of hotwords. |)

[//]: # (|              [Paraformer]&#40;https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary&#41;              | CN & EN  | Alibaba Speech Data &#40;50000hours&#41; |    8358    |    68M    |    Offline     | Duration of input wav <= 20s                                                                                                    |)

[//]: # (|          [Paraformer-online]&#40;https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary&#41;           | CN & EN  | Alibaba Speech Data &#40;50000hours&#41; |    8404    |    68M    |     Online     | Which could deal with streaming input                                                                                           |)

[//]: # (|       [Paraformer-tiny]&#40;https://www.modelscope.cn/models/damo/speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch/summary&#41;       |    CN    |  Alibaba Speech Data &#40;200hours&#41;  |    544     |   5.2M    |    Offline     | Lightweight Paraformer model which supports Mandarin command words recognition                                                  |)

[//]: # (|                   [Paraformer-aishell]&#40;https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-aishell1-pytorch/summary&#41;                   |    CN    |        AISHELL &#40;178hours&#41;        |    4234    |    43M    |    Offline     |                                                                                                                                 |)

[//]: # (|       [ParaformerBert-aishell]&#40;https://modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary&#41;       |    CN    |        AISHELL &#40;178hours&#41;        |    4234    |    43M    |    Offline     |                                                                                                                                 |)

[//]: # (|        [Paraformer-aishell2]&#40;https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary&#41;         |    CN    |      AISHELL-2 &#40;1000hours&#41;       |    5212    |    64M    |    Offline     |                                                                                                                                 |)

[//]: # (|    [ParaformerBert-aishell2]&#40;https://www.modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary&#41;     |    CN    |      AISHELL-2 &#40;1000hours&#41;       |    5212    |    64M    |    Offline     |                                                                                                                                 |)


#### UniASR Models

[//]: # (|                                                               Model Name                                                               | Language |          Training Data           | Vocab Size | Parameter | Offline/Online | Notes                                                                                                                           |)

[//]: # (|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------:|:--------------------------------:|:----------:|:---------:|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------|)

[//]: # (|       [UniASR]&#40;https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-online/summary&#41;        | CN & EN  | Alibaba Speech Data &#40;60000hours&#41; |    8358    |   100M    |     Online     | UniASR streaming offline unifying models                                                                                                    |)

[//]: # (| [UniASR-large]&#40;https://modelscope.cn/models/damo/speech_UniASR-large_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-offline/summary&#41; | CN & EN  | Alibaba Speech Data &#40;60000hours&#41; |    8358    |   220M    |    Offline     | UniASR streaming offline unifying models                                                                                                    |)

[//]: # (|           [UniASR Burmese]&#40;https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-my-16k-common-vocab696-pytorch/summary&#41;           | Burmese  |  Alibaba Speech Data &#40;? hours&#41;   |    696     |    95M    |     Online     | UniASR streaming offline unifying models                                                                                                    |)

[//]: # (|           [UniASR Hebrew]&#40;https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-he-16k-common-vocab1085-pytorch/summary&#41;           |  Hebrew  |  Alibaba Speech Data &#40;? hours&#41;   |    1085    |    95M    |     Online     | UniASR streaming offline unifying models                                                                                                    |)

[//]: # (|       [UniASR Urdu]&#40;https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ur-16k-common-vocab877-pytorch/summary&#41;                  |   Urdu   |  Alibaba Speech Data &#40;? hours&#41;   |    877     |    95M    |     Online     | UniASR streaming offline unifying models                                                                                                    |)

#### Conformer Models

[//]: # (|                                                       Model Name                                                       | Language |     Training Data     | Vocab Size | Parameter | Offline/Online | Notes                                                                                                                           |)

[//]: # (|:----------------------------------------------------------------------------------------------------------------------:|:--------:|:---------------------:|:----------:|:---------:|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------|)

[//]: # (| [Conformer]&#40;https://modelscope.cn/models/damo/speech_conformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary&#41;   |   CN     |  AISHELL &#40;178hours&#41;   |    4234    |    44M    |    Offline     | Duration of input wav <= 20s                                                                                                    |)

[//]: # (| [Conformer]&#40;https://www.modelscope.cn/models/damo/speech_conformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary&#41;   |   CN     | AISHELL-2 &#40;1000hours&#41; |    5212    |    44M    |    Offline     | Duration of input wav <= 20s                                                                                                    |)


#### RNN-T Models

### Multi-talker Speech Recognition Models

#### MFCCA Models

[//]: # (|                                                  Model Name                                                   | Language |               Training Data                | Vocab Size | Parameter | Offline/Online | Notes                                                                                                                           |)

[//]: # (|:-------------------------------------------------------------------------------------------------------------:|:--------:|:------------------------------------------:|:----------:|:---------:|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------|)

[//]: # (| [MFCCA]&#40;https://www.modelscope.cn/models/NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950/summary&#41;    |   CN     | AliMeeting、AISHELL-4、Simudata &#40;917hours&#41;   |     4950   |    45M    |    Offline     | Duration of input wav <= 20s, channel of input wav <= 8 channel |)



### Voice Activity Detection Models

|                      Model Name                      |        Training Data         | Parameters | Sampling Rate | Notes |
|:----------------------------------------------------:|:----------------------------:|:----------:|:-------------:|:------|
| [FSMN-VAD](https://huggingface.co/funasr/FSMN-VAD)   | Alibaba Speech Data (5000hours) |    0.4M    |     16000     |       |

[//]: # (|   [FSMN-VAD]&#40;https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-8k-common/summary&#41;        | Alibaba Speech Data &#40;5000hours&#41; |    0.4M    |     8000      |       |)

### Punctuation Restoration Models

|                              Model Name                              |        Training Data         | Parameters | Vocab Size| Offline/Online | Notes |
|:--------------------------------------------------------------------:|:----------------------------:|:----------:|:----------:|:--------------:|:------|
| [CT-Transformer](https://huggingface.co/funasr/CT-Transformer-punc)  | Alibaba Text Data |    70M     |    272727     |    Offline     |   offline punctuation model    |

[//]: # (| [CT-Transformer]&#40;https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727/summary&#41;      | Alibaba Text Data |    70M     |    272727     |     Online     |  online punctuation model     |)

### Language Models

[//]: # (|                                                       Model Name                                                       |        Training Data         | Parameters | Vocab Size | Notes |)

[//]: # (|:----------------------------------------------------------------------------------------------------------------------:|:----------------------------:|:----------:|:----------:|:------|)

[//]: # (| [Transformer]&#40;https://www.modelscope.cn/models/damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch/summary&#41;      | Alibaba Speech Data &#40;?hours&#41; |    57M     |    8404    |       |)

### Speaker Verification Models

[//]: # (|                                                  Model Name                                                   |   Training Data   | Parameters | Number Speaker | Notes |)

[//]: # (|:-------------------------------------------------------------------------------------------------------------:|:-----------------:|:----------:|:----------:|:------|)

[//]: # (| [Xvector]&#40;https://www.modelscope.cn/models/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/summary&#41; | CNCeleb &#40;1,200 hours&#41;  |   17.5M    |    3465    |    Xvector, speaker verification, Chinese   |)

[//]: # (| [Xvector]&#40;https://www.modelscope.cn/models/damo/speech_xvector_sv-en-us-callhome-8k-spk6135-pytorch/summary&#41; | CallHome &#40;60 hours&#41; |    61M     |    6135    |   Xvector, speaker verification, English    |)

### Speaker diarization Models

[//]: # (|                                                    Model Name                                                    |    Training Data    | Parameters | Notes |)

[//]: # (|:----------------------------------------------------------------------------------------------------------------:|:-------------------:|:----------:|:------|)

[//]: # (| [SOND]&#40;https://www.modelscope.cn/models/damo/speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch/summary&#41; | AliMeeting &#40;120 hours&#41; |   40.5M    |    Speaker diarization, profiles and records, Chinese |)

[//]: # (| [SOND]&#40;https://www.modelscope.cn/models/damo/speech_diarization_sond-en-us-callhome-8k-n16k4-pytorch/summary&#41;    |  CallHome &#40;60 hours&#41;  |     12M     |    Speaker diarization, profiles and records, English   |)

### Timestamp Prediction Models

[//]: # (|                                                    Model Name                                     |  Language  |    Training Data    | Parameters | Notes |)

[//]: # (|:--------------------------------------------------------------------------------------------------:|:--------------:|:-------------------:|:----------:|:------|)

[//]: # (| [TP-Aligner]&#40;https://modelscope.cn/models/damo/speech_timestamp_prediction-v1-16k-offline/summary&#41; | CN | Alibaba Speech Data &#40;50000hours&#41; |   37.8M    |    Timestamp prediction, Mandarin, middle size |)
