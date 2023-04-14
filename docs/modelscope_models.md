# Pretrained models on ModelScope

## Model License
-  Apache License 2.0

## Model Zoo
Here we provided several pretrained models on different datasets. The details of models and datasets can be found on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition).

### Speech Recognition Models
#### Paraformer Models
|                                                                     Model Name                                                                     | Language |          Training Data           | Vocab Size | Parameter | Offline/Online | Notes                                                                                                                           |
|:--------------------------------------------------------------------------------------------------------------------------------------------------:|:--------:|:--------------------------------:|:----------:|:---------:|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------|
|        [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)        | CN & EN  | Alibaba Speech Data (60000hours) |    8404    |   220M    |    Offline     | Duration of input wav <= 20s                                                                                                    |
| [Paraformer-large-long](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) | CN & EN  | Alibaba Speech Data (60000hours) |    8404    |   220M    |    Offline     | Which ould deal with arbitrary length input wav                                                                                 |
| [paraformer-large-contextual](https://www.modelscope.cn/models/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/summary) | CN & EN  | Alibaba Speech Data (60000hours) |    8404    |   220M    |    Offline     | Which supports the hotword customization based on the incentive enhancement, and improves the recall and precision of hotwords. |
|              [Paraformer](https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary)              | CN & EN  | Alibaba Speech Data (50000hours) |    8358    |    68M    |    Offline     | Duration of input wav <= 20s                                                                                                    |
|          [Paraformer-online](https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary)           | CN & EN  | Alibaba Speech Data (50000hours) |    8404    |    68M    |     Online     | Which could deal with streaming input                                                                                           |
|       [Paraformer-tiny](https://www.modelscope.cn/models/damo/speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch/summary)       |    CN    |  Alibaba Speech Data (200hours)  |    544     |   5.2M    |    Offline     | Lightweight Paraformer model which supports Mandarin command words recognition                                                  |
|                   [Paraformer-aishell](https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-aishell1-pytorch/summary)                   |    CN    |        AISHELL (178hours)        |    4234    |    43M    |    Offline     |                                                                                                                                 |
|       [ParaformerBert-aishell](https://modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary)       |    CN    |        AISHELL (178hours)        |    4234    |    43M    |    Offline     |                                                                                                                                 |
|        [Paraformer-aishell2](https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary)         |    CN    |      AISHELL-2 (1000hours)       |    5212    |    64M    |    Offline     |                                                                                                                                 |
|    [ParaformerBert-aishell2](https://www.modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary)     |    CN    |      AISHELL-2 (1000hours)       |    5212    |    64M    |    Offline     |                                                                                                                                 |


#### UniASR Models
|                                                               Model Name                                                               | Language |          Training Data           | Vocab Size | Parameter | Offline/Online | Notes                                                                                                                           |
|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------:|:--------------------------------:|:----------:|:---------:|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------|
|       [UniASR](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-online/summary)        | CN & EN  | Alibaba Speech Data (60000hours) |    8358    |   100M    |     Online     | UniASR streaming offline unifying models                                                                                                    |
| [UniASR-large](https://modelscope.cn/models/damo/speech_UniASR-large_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-offline/summary) | CN & EN  | Alibaba Speech Data (60000hours) |    8358    |   220M    |    Offline     | UniASR streaming offline unifying models                                                                                                    |
|           [UniASR Burmese](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-my-16k-common-vocab696-pytorch/summary)           | Burmese  |  Alibaba Speech Data (? hours)   |    696     |    95M    |     Online     | UniASR streaming offline unifying models                                                                                                    |
|           [UniASR Hebrew](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-he-16k-common-vocab1085-pytorch/summary)           |  Hebrew  |  Alibaba Speech Data (? hours)   |    1085    |    95M    |     Online     | UniASR streaming offline unifying models                                                                                                    |
|       [UniASR Urdu](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ur-16k-common-vocab877-pytorch/summary)                  |   Urdu   |  Alibaba Speech Data (? hours)   |    877     |    95M    |     Online     | UniASR streaming offline unifying models                                                                                                    |

#### Conformer Models
#### Paraformer Models
|                                                       Model Name                                                       | Language |     Training Data     | Vocab Size | Parameter | Offline/Online | Notes                                                                                                                           |
|:----------------------------------------------------------------------------------------------------------------------:|:--------:|:---------------------:|:----------:|:---------:|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------|
| [Conformer](https://modelscope.cn/models/damo/speech_conformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary)   |   CN     |  AISHELL (178hours)   |    4234    |    44M    |    Offline     | Duration of input wav <= 20s                                                                                                    |
| [Conformer](https://www.modelscope.cn/models/damo/speech_conformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary)   |   CN     | AISHELL-2 (1000hours) |    5212    |    44M    |    Offline     | Duration of input wav <= 20s                                                                                                    |

#### RNN-T Models

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

|                                                  Model Name                                                   |   Training Data   | Parameters | Vocab Size | Notes |
|:-------------------------------------------------------------------------------------------------------------:|:-----------------:|:----------:|:----------:|:------|
| [Xvector](https://www.modelscope.cn/models/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/summary) | CNCeleb (?hours)  |   17.5M    |    3465    |       |
| [Xvector](https://www.modelscope.cn/models/damo/speech_xvector_sv-en-us-callhome-8k-spk6135-pytorch/summary) | CallHome (?hours) |    61M     |    6135    |       |

### Speaker diarization Models

|                                                    Model Name                                                    |    Training Data    | Parameters | Notes |
|:----------------------------------------------------------------------------------------------------------------:|:-------------------:|:----------:|:------|
| [SOND](https://www.modelscope.cn/models/damo/speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch/summary) | AliMeeting (?hours) |   40.5M    |       |
| [SOND](https://www.modelscope.cn/models/damo/speech_diarization_sond-en-us-callhome-8k-n16k4-pytorch/summary)    |  CallHome (?hours)  |     12M     |       |
