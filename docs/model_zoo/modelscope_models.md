# Pretrained Models on ModelScope

## Model License
You are free to use, copy, modify, and share FunASR models under the conditions of this agreement. You should indicate the model source and author information when using, copying, modifying and sharing FunASR models. You should keep the relevant names of models in [FunASR software].. Full model license could see [license](https://github.com/alibaba-damo-academy/FunASR/blob/main/MODEL_LICENSE)

## Model Usage
Ref to [docs](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_pipeline/quick_start.html)

## Model Zoo
Here we provided several pretrained models on different datasets. The details of models and datasets can be found on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition).

### Speech Recognition Models
#### Paraformer Models

|                                                                     Model Name                                                                     | Language |          Training Data           | Vocab Size | Parameter | Offline/Online | Notes                                                                                                                           |
|:--------------------------------------------------------------------------------------------------------------------------------------------------:|:--------:|:--------------------------------:|:----------:|:---------:|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------|
|        [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)        | CN & EN  | Alibaba Speech Data (60000hours) |    8404    |   220M    |    Offline     | Duration of input wav <= 20s                                                                                                    |
| [Paraformer-large-long](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) | CN & EN  | Alibaba Speech Data (60000hours) |    8404    |   220M    |    Offline     | Which would deal with arbitrary length input wav                                                                                 |
| [Paraformer-large-contextual](https://www.modelscope.cn/models/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/summary) | CN & EN  | Alibaba Speech Data (60000hours) |    8404    |   220M    |    Offline     | Which supports the hotword customization based on the incentive enhancement, and improves the recall and precision of hotwords. |
|              [Paraformer](https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary)              | CN & EN  | Alibaba Speech Data (50000hours) |    8358    |    68M    |    Offline     | Duration of input wav <= 20s                                                                                                    |
|           [Paraformer-online](https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online/summary)           | CN & EN  | Alibaba Speech Data (50000hours) |    8404    |    68M    |     Online     | Which could deal with streaming input                                                                                           |
|  [Paraformer-large-online](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary)        | CN & EN  | Alibaba Speech Data (60000hours) |    8404    |   220M    |    Online     | Which could deal with streaming input                                                                                                    |
|       [Paraformer-tiny](https://www.modelscope.cn/models/damo/speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch/summary)       |    CN    |  Alibaba Speech Data (200hours)  |    544     |   5.2M    |    Offline     | Lightweight Paraformer model which supports Mandarin command words recognition                                                  |
|                   [Paraformer-aishell](https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-aishell1-pytorch/summary)                   |    CN    |        AISHELL (178hours)        |    4234    |    43M    |    Offline     |                                                                                                                                 |
|       [ParaformerBert-aishell](https://modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary)       |    CN    |        AISHELL (178hours)        |    4234    |    43M    |    Offline     |                                                                                                                                 |
|        [Paraformer-aishell2](https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary)         |    CN    |      AISHELL-2 (1000hours)       |    5212    |    64M    |    Offline     |                                                                                                                                 |
|    [ParaformerBert-aishell2](https://www.modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary)     |    CN    |      AISHELL-2 (1000hours)       |    5212    |    64M    |    Offline     |                                                                                                                                 |


#### UniASR Models

|                                                                    Model Name                                                                     |    Language     |           Training Data           | Vocab Size | Parameter | Offline/Online | Notes                                                                                                                           |
|:-------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------:|:---------------------------------:|:----------:|:---------:|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------|
|             [UniASR](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-online/summary)             |     CN & EN     | Alibaba Speech Data (60000 hours) |    8358    |   100M    |     Online     | UniASR streaming offline unifying models                                                                                                    |
|      [UniASR-large](https://modelscope.cn/models/damo/speech_UniASR-large_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-offline/summary)       |     CN & EN     | Alibaba Speech Data (60000 hours) |    8358    |   220M    |    Offline     | UniASR streaming offline unifying models                                                                                                    |
|          [UniASR English](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-en-16k-common-vocab1080-tensorflow1-online/summary)           |       EN        | Alibaba Speech Data (10000 hours) |    1080     |    95M    |     Online     | UniASR streaming online unifying models                                                                                                    |
|          [UniASR Russian](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ru-16k-common-vocab1664-tensorflow1-online/summary)           |       RU        | Alibaba Speech Data (5000 hours)  |    1664     |    95M    |     Online     | UniASR streaming online unifying models                                                                                                    |
|           [UniASR Japanese](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-online/summary)           |       JA        | Alibaba Speech Data (5000 hours)  |    5977     |    95M    |     Online     | UniASR streaming offline unifying models                                                                                                    |
|           [UniASR Korean](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-online/summary)           |       KO        | Alibaba Speech Data (2000 hours)  |    6400     |    95M    |     Online     | UniASR streaming online unifying models                                                                                                    |
| [UniASR Cantonese (CHS)](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online/summary) | Cantonese (CHS) | Alibaba Speech Data (5000 hours)  |    1468     |    95M    |     Online     | UniASR streaming online unifying models                                                                                                    |
|         [UniASR Indonesian](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-id-16k-common-vocab1067-tensorflow1-online/summary)         |       ID        | Alibaba Speech Data (1000 hours)  |    1067     |    95M    |     Online     | UniASR streaming offline unifying models                                                                                                    |
|           [UniASR Vietnamese](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-vi-16k-common-vocab1001-pytorch-online/summary)           |       VI        | Alibaba Speech Data (1000 hours)  |    1001     |    95M    |     Online     | UniASR streaming offline unifying models                                                                                                    |
|          [UniASR Spanish](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-es-16k-common-vocab3445-tensorflow1-online/summary)           |       ES        | Alibaba Speech Data (1000 hours)  |    3445     |    95M    |     Online     | UniASR streaming online unifying models                                                                                                    |
|         [UniASR Portuguese](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-pt-16k-common-vocab1617-tensorflow1-online/summary)         |       PT        | Alibaba Speech Data (1000 hours)  |    1617     |    95M    |     Online     | UniASR streaming offline unifying models                                                                                                    |
|           [UniASR French](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-fr-16k-common-vocab3472-tensorflow1-online/summary)           |       FR        | Alibaba Speech Data (1000 hours)  |    3472     |    95M    |     Online     | UniASR streaming online unifying models                                                                                                    |
|           [UniASR German](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-de-16k-common-vocab3690-tensorflow1-online/summary)           |       GE        | Alibaba Speech Data (1000 hours)  |    3690     |    95M    |     Online     | UniASR streaming online unifying models                                                                                                    |
|            [UniASR Persian](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-fa-16k-common-vocab1257-pytorch-online/summary)             |       FA        | Alibaba Speech Data (1000 hours)  |    1257     |    95M    |     Online     | UniASR streaming offline unifying models                                                                                                    |
|                [UniASR Burmese](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-my-16k-common-vocab696-pytorch/summary)                 |       MY        | Alibaba Speech Data (1000 hours)  |    696     |    95M    |     Online     | UniASR streaming offline unifying models                                                                                                    |
|                [UniASR Hebrew](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-he-16k-common-vocab1085-pytorch/summary)                 |       HE        | Alibaba Speech Data (1000 hours)  |    1085    |    95M    |     Online     | UniASR streaming offline unifying models                                                                                                    |
|              [UniASR Urdu](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ur-16k-common-vocab877-pytorch/summary)                      |       UR        | Alibaba Speech Data (1000 hours)  |    877     |    95M    |     Online     | UniASR streaming offline unifying models                                                                                                    |
|              [UniASR Turkish](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-tr-16k-common-vocab1582-pytorch/summary)                      |       TR        | Alibaba Speech Data (1000 hours)  |    1582     |    95M    |     Online     | UniASR streaming offline unifying models                                                                                                    |


#### Conformer Models

|                                                       Model Name                                                       | Language |     Training Data     | Vocab Size | Parameter | Offline/Online | Notes                                                                                                                           |
|:----------------------------------------------------------------------------------------------------------------------:|:--------:|:---------------------:|:----------:|:---------:|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------|
| [Conformer](https://modelscope.cn/models/damo/speech_conformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary)   |   CN     |  AISHELL (178hours)   |    4234    |    44M    |    Offline     | Duration of input wav <= 20s                                                                                                    |
| [Conformer](https://www.modelscope.cn/models/damo/speech_conformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary)   |   CN     | AISHELL-2 (1000hours) |    5212    |    44M    |    Offline     | Duration of input wav <= 20s                                                                                                    |
| [Conformer](https://modelscope.cn/models/damo/speech_conformer_asr-en-16k-vocab4199-pytorch/summary)   |   EN     | Alibaba Speech Data (10000hours) |    4199    |    220M    |    Offline     | Duration of input wav <= 20s                                                                                                    |


#### RNN-T Models

### Multi-talker Speech Recognition Models

#### MFCCA Models

|                                                  Model Name                                                   | Language |               Training Data                | Vocab Size | Parameter | Offline/Online | Notes                                                                                                                           |
|:-------------------------------------------------------------------------------------------------------------:|:--------:|:------------------------------------------:|:----------:|:---------:|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------|
| [MFCCA](https://www.modelscope.cn/models/NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950/summary)    |   CN     | AliMeeting、AISHELL-4、Simudata (917hours)   |     4950   |    45M    |    Offline     | Duration of input wav <= 20s, channel of input wav <= 8 channel |



### Voice Activity Detection Models

|                                           Model Name                                           |        Training Data         | Parameters | Sampling Rate | Notes |
|:----------------------------------------------------------------------------------------------:|:----------------------------:|:----------:|:-------------:|:------|
| [FSMN-VAD](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) | Alibaba Speech Data (5000hours) |    0.4M    |     16000     |       |
|   [FSMN-VAD](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-8k-common/summary)        | Alibaba Speech Data (5000hours) |    0.4M    |     8000      |       |

### Punctuation Restoration Models

|                                                         Model Name                                                         |        Training Data         | Parameters | Vocab Size| Offline/Online | Notes |
|:--------------------------------------------------------------------------------------------------------------------------:|:----------------------------:|:----------:|:----------:|:--------------:|:------|
|      [CT-Transformer](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary)      | Alibaba Text Data |    70M     |    272727     |    Offline     |   offline punctuation model    |
| [CT-Transformer](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727/summary)      | Alibaba Text Data |    70M     |    272727     |     Online     |  online punctuation model     |

### Language Models

|                                                       Model Name                                                       |        Training Data         | Parameters | Vocab Size | Notes |
|:----------------------------------------------------------------------------------------------------------------------:|:----------------------------:|:----------:|:----------:|:------|
| [Transformer](https://www.modelscope.cn/models/damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch/summary)      | Alibaba Speech Data (?hours) |    57M     |    8404    |       |

### Speaker Verification Models

|                                                  Model Name                                                   |   Training Data   | Parameters | Number Speaker | Notes |
|:-------------------------------------------------------------------------------------------------------------:|:-----------------:|:----------:|:----------:|:------|
| [Xvector](https://www.modelscope.cn/models/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/summary) | CNCeleb (1,200 hours)  |   17.5M    |    3465    |    Xvector, speaker verification, Chinese   |
| [Xvector](https://www.modelscope.cn/models/damo/speech_xvector_sv-en-us-callhome-8k-spk6135-pytorch/summary) | CallHome (60 hours) |    61M     |    6135    |   Xvector, speaker verification, English    |

### Speaker Diarization Models

|                                                    Model Name                                                    |    Training Data    | Parameters | Notes |
|:----------------------------------------------------------------------------------------------------------------:|:-------------------:|:----------:|:------|
| [SOND](https://www.modelscope.cn/models/damo/speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch/summary) | AliMeeting (120 hours) |   40.5M    |    Speaker diarization, profiles and records, Chinese |
| [SOND](https://www.modelscope.cn/models/damo/speech_diarization_sond-en-us-callhome-8k-n16k4-pytorch/summary)    |  CallHome (60 hours)  |     12M     |    Speaker diarization, profiles and records, English   |

### Timestamp Prediction Models

|                                                    Model Name                                     |  Language  |    Training Data    | Parameters | Notes |
|:--------------------------------------------------------------------------------------------------:|:--------------:|:-------------------:|:----------:|:------|
| [TP-Aligner](https://modelscope.cn/models/damo/speech_timestamp_prediction-v1-16k-offline/summary) | CN | Alibaba Speech Data (50000hours) |   37.8M    |    Timestamp prediction, Mandarin, middle size |

### Inverse Text Normalization (ITN) Models

|                                                    Model Name                                                    | Language | Parameters | Notes                    |
|:----------------------------------------------------------------------------------------------------------------:|:--------:|:----------:|:-------------------------|
| [English](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-en/summary) |    EN    |   1.54M    | ITN, ASR post-processing |
| [Russian](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-ru/summary) |    RU    |   17.79M   | ITN, ASR post-processing |
| [Japanese](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-ja/summary) |    JA    |    6.8M    | ITN, ASR post-processing |
| [Korean](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-ko/summary) |    KO    |   1.28M    | ITN, ASR post-processing |
| [Indonesian](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-id/summary) |    ID    |   2.06M    | ITN, ASR post-processing |
| [Vietnamese](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-vi/summary) |    VI    |   0.92M    | ITN, ASR post-processing |
| [Tagalog](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-tl/summary) |    TL    |    0.65M     | ITN, ASR post-processing |
| [Spanish](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-es/summary) |    ES    |   1.32M    | ITN, ASR post-processing |
| [Portuguese](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-pt/summary) |    PT    |   1.28M    | ITN, ASR post-processing |
| [French](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-fr/summary) |    FR    |   4.39M    | ITN, ASR post-processing |
| [German](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-de/summary)|    GE    |   3.95M    | ITN, ASR post-processing |
