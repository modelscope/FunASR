(简体中文|[English](./modelscope_models.md))

# ModelScope上的预训练模型

## 模型许可协议
您可以在本协议的条件下自由使用、复制、修改和分享FunASR模型。在使用、复制、修改和分享FunASR模型时，您应当标明模型来源和作者信息。您应当在[FunASR软件]中保留相关模型的名称。完整的模型许可证请参见 [模型许可协议](https://github.com/alibaba-damo-academy/FunASR/blob/main/MODEL_LICENSE)

## 模型用法
模型用法参考[文档](funasr/quick_start_zh.md)

## 模型仓库
这里我们提供了在不同数据集上预训练的模型。模型和数据集的详细信息可在 [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition)中找到.

### 语音识别模型
#### Paraformer模型

|                                                                     模型名字                                                                     |    语言    |         训练数据          |       词典大小        | 参数量  | 非实时/实时  | 备注                         |
|:--------------------------------------------------------------------------------------------------------------------------------------------------:|:--------:|:---------------------:|:-----------------:|:----:|:-------:|:---------------------------|
|        [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)        |  中文和英文   |    阿里巴巴语音数据（60000小时）  |       8404        | 220M |   非实时   | 输入wav文件持续时间不超过20秒          |
| [Paraformer-large长音频版本](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) |  中文和英文   |   阿里巴巴语音数据（60000小时）   |       8404        | 220M |   非实时   || 能够处理任意长度的输入wav文件                                                                                |
|     [Paraformer-large热词](https://www.modelscope.cn/models/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/summary)      |         中文和英文         | 阿里巴巴语音数据（60000小时） | 8404 |  220M   | 非实时                        | 基于激励增强的热词定制支持，可以提高热词的召回率和准确率，输入wav文件持续时间不超过20秒  |
|       [Paraformer](https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary)                     |   中文和英文  |   阿里巴巴语音数据（50000小时）   |       8358        | 68M  |   离线    | 输入wav文件持续时间不超过20秒          |
|               [Paraformer实时](https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online/summary)                | 中文和英文  | 阿里巴巴语音数据 (50000hours) |       8404        | 68M  | 实时  | 能够处理流式输入                   |
|         [Paraformer-large实时](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary)          | 中文和英文  | 阿里巴巴语音数据 (60000hours) |       8404        | 220M | 实时  | 能够处理流式输入                   |
|       [Paraformer-tiny](https://www.modelscope.cn/models/damo/speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch/summary)       |   中文   |  阿里巴巴语音数据 (200hours)  |        544        | 5.2M | 非实时 | 轻量级Paraformer模型，支持普通话命令词识别 |
|                   [Paraformer-aishell](https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-aishell1-pytorch/summary)                   |   中文   |  AISHELL (178hours)   |       4234        | 43M  | 非实时 | 学术模型                       |
|       [ParaformerBert-aishell](https://modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary)       |   中文   |  AISHELL (178hours)   |       4234        | 43M  | 非实时 | 学术模型                       |
|        [Paraformer-aishell2](https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary)         |   中文   | AISHELL-2 (1000hours) |       5212        | 64M  | 非实时 | 学术模型                       |
|    [ParaformerBert-aishell2](https://www.modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary)     |   中文   | AISHELL-2 (1000hours) |       5212        | 64M  | 非实时 | 学术模型                       |


#### UniASR模型

|                                                                    模型名字                                                                     |    语言    |           训练数据           | Vocab Size | Parameter | 非实时/实时 | 备注                                                                                                                           |
|:-------------------------------------------------------------------------------------------------------------------------------------------------:|:--------:|:---------------------------------:|:----------:|:---------:|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------|
|             [UniASR](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-实时/summary)             |  中文和英文   | 阿里巴巴语音数据 (60000 小时) |    8358    |   100M    |     实时     | 流式离线一体化模型                                                                                                    |
|      [UniASR-large](https://modelscope.cn/models/damo/speech_UniASR-large_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-非实时/summary)       |  中文和英文   | 阿里巴巴语音数据 (60000 小时) |    8358    |   220M    |    非实时     | 流式离线一体化模型                                                                                                    |
|          [UniASR English](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-en-16k-common-vocab1080-tensorflow1-实时/summary)           |    英文    | 阿里巴巴语音数据 (10000 小时) |    1080     |    95M    |     实时     | 流式离线一体化模型                                                                                                    |
|          [UniASR Russian](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ru-16k-common-vocab1664-tensorflow1-实时/summary)           |    俄语    | 阿里巴巴语音数据 (5000 小时)  |    1664     |    95M    |     实时     | 流式离线一体化模型                                                                                                    |
|           [UniASR Japanese](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-实时/summary)           |    日语    | 阿里巴巴语音数据 (5000 小时)  |    5977     |    95M    |     实时     | 流式离线一体化模型                                                                                                    |
|           [UniASR Korean](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-实时/summary)           |    韩语    | 阿里巴巴语音数据 (2000 小时)  |    6400     |    95M    |     实时     | 流式离线一体化模型                                                                                                    |
| [UniASR Cantonese (CHS)](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-实时/summary) | 粤语（简体中文） | 阿里巴巴语音数据 (5000 小时)  |    1468     |    95M    |     实时     | 流式离线一体化模型                                                                                                    |
|         [UniASR Indonesian](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-id-16k-common-vocab1067-tensorflow1-实时/summary)         |   印尼语    | 阿里巴巴语音数据 (1000 小时)  |    1067     |    95M    |     实时     | 流式离线一体化模型                                                                                                    |
|           [UniASR Vietnamese](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-vi-16k-common-vocab1001-pytorch-实时/summary)           |   越南语    | 阿里巴巴语音数据 (1000 小时)  |    1001     |    95M    |     实时     | 流式离线一体化模型                                                                                                    |
|          [UniASR Spanish](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-es-16k-common-vocab3445-tensorflow1-实时/summary)           |   西班牙语   | 阿里巴巴语音数据 (1000 小时)  |    3445     |    95M    |     实时     | 流式离线一体化模型                                                                                                    |
|         [UniASR Portuguese](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-pt-16k-common-vocab1617-tensorflow1-实时/summary)         |   葡萄牙语   | 阿里巴巴语音数据 (1000 小时)  |    1617     |    95M    |     实时     | 流式离线一体化模型                                                                                                    |
|           [UniASR French](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-fr-16k-common-vocab3472-tensorflow1-实时/summary)           |    法语    | 阿里巴巴语音数据 (1000 小时)  |    3472     |    95M    |     实时     | 流式离线一体化模型                                                                                                    |
|           [UniASR German](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-de-16k-common-vocab3690-tensorflow1-实时/summary)           |    德语    | 阿里巴巴语音数据 (1000 小时)  |    3690     |    95M    |     实时     | 流式离线一体化模型                                                                                                    |
|            [UniASR Persian](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-fa-16k-common-vocab1257-pytorch-实时/summary)             |   波斯语    | 阿里巴巴语音数据 (1000 小时)  |    1257     |    95M    |     实时     | 流式离线一体化模型                                                                                                    |
|                [UniASR Burmese](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-my-16k-common-vocab696-pytorch/summary)                 |   缅甸语    | 阿里巴巴语音数据 (1000 小时)  |    696     |    95M    |     实时     | 流式离线一体化模型                                                                                                    |
|                [UniASR Hebrew](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-he-16k-common-vocab1085-pytorch/summary)                 |   希伯来语   | 阿里巴巴语音数据 (1000 小时)  |    1085    |    95M    |     实时     | 流式离线一体化模型                                                                                                    |
|              [UniASR Urdu](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ur-16k-common-vocab877-pytorch/summary)                      |   乌尔都语   | 阿里巴巴语音数据 (1000 小时)  |    877     |    95M    |     实时     | 流式离线一体化模型                                                                                                    |
|              [UniASR Turkish](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-tr-16k-common-vocab1582-pytorch/summary)                      |   土耳其语   | 阿里巴巴语音数据 (1000 小时)  |    1582     |    95M    |     实时     | 流式离线一体化模型                                                                                                    |


#### Conformer模型

|                                                       模型名字                                                       | 语言 |     训练数据     | Vocab Size | Parameter | 非实时/实时 | 备注                                                                                                                           |
|:----------------------------------------------------------------------------------------------------------------------:|:--------:|:---------------------:|:----------:|:---------:|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------|
| [Conformer](https://modelscope.cn/models/damo/speech_conformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary)   |  中文    |  AISHELL (178hours)   |    4234    |    44M    |    非实时     | 输入wav文件持续时间不超过20秒                                                                                                   |
| [Conformer](https://www.modelscope.cn/models/damo/speech_conformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary)   |  中文    | AISHELL-2 (1000hours) |    5212    |    44M    |    非实时     | 输入wav文件持续时间不超过20秒                                                                                                   |
| [Conformer](https://modelscope.cn/models/damo/speech_conformer_asr-en-16k-vocab4199-pytorch/summary)   | 英文    | 阿里巴巴语音数据 (10000hours) |    4199    |    220M    |    非实时     | 输入wav文件持续时间不超过20秒                                                                                                   |


#### RNN-T 模型

### 多说话人语音识别模型

#### MFCCA模型

|                                                  模型名字                                                   | 语言 |               训练数据                | Vocab Size | Parameter | 非实时/实时 | 备注                                                                                                                           |
|:-------------------------------------------------------------------------------------------------------------:|:--------:|:------------------------------------------:|:----------:|:---------:|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------|
| [MFCCA](https://www.modelscope.cn/models/NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950/summary)    |  中文    | AliMeeting、AISHELL-4、Simudata (917hours)   |     4950   |    45M    |    非实时     | 输入音频的持续时间不超过20秒，输入音频的通道数不超过8通道。 |



### 语音端点检测模型

|                                           模型名字                                           |        训练数据         | 模型参数 | Sampling Rate | 备注 |
|:----------------------------------------------------------------------------------------------:|:----------------------------:|:----------:|:-------------:|:------|
| [FSMN-VAD](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) | 阿里巴巴语音数据 (5000hours) |    0.4M    |     16000     |       |
|   [FSMN-VAD](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-8k-common/summary)        | 阿里巴巴语音数据 (5000hours) |    0.4M    |     8000      |       |

### 标点恢复模型

|                                                         模型名字                                                        | 语言  |        训练数据         | 模型参数 | Vocab Size| 非实时/实时 | 备注      |
|:--------------------------------------------------------------------------------------------------------------------------:|:----------:|:----------------------------:|:----------:|:----------:|:--------------:|:--------|
|      [CT-Transformer-Large](https://modelscope.cn/models/damo/punc_ct-transformer_cn-en-common-vocab471067-large/summary)     | 中文和英文 | Alibaba Text Data(100M) |    1.1G     |    471067     |    非实时     | 支持中英文标点大模型 |
|      [CT-Transformer](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary)     | 中文和英文 | Alibaba Text Data(70M) |    291M     |    272727     |    非实时     | 支持中英文标点 |
| [CT-Transformer-Realtime](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727/summary)      | 中文和英文 | Alibaba Text Data(70M) |    288M     |    272727     |     实时     | VAD点实时标点  |

### 语音模型

|                                                       模型名字                                                       |   训练数据    | 模型参数 | 词典大小 | 备注 |
|:----------------------------------------------------------------------------------------------------------------------:|:---------:|:----------:|:----:|:------|
| [Transformer](https://www.modelscope.cn/models/damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch/summary)      | 阿里巴巴语音数据  |    57M     | 8404 |       |

### 说话人确认模型

|                                                  模型名字                                                   |   训练数据   | 模型参数 | Number Speaker | 备注          |
|:-------------------------------------------------------------------------------------------------------------:|:-----------------:|:----------:|:----------:|:------------|
| [Xvector](https://www.modelscope.cn/models/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/summary) | CNCeleb (1,200 小时)  |   17.5M    |    3465    | Xvector， 中文 |
| [Xvector](https://www.modelscope.cn/models/damo/speech_xvector_sv-en-us-callhome-8k-spk6135-pytorch/summary) | CallHome (60 小时) |    61M     |    6135    | Xvector，英文  |

### 说话人日志模型

|                                                    模型名字                                                    |    训练数据    | 模型参数 | 备注  |
|:----------------------------------------------------------------------------------------------------------------:|:-------------------:|:----------:|:----|
| [SOND](https://www.modelscope.cn/models/damo/speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch/summary) | AliMeeting (120 小时) |   40.5M    | 中文  |
| [SOND](https://www.modelscope.cn/models/damo/speech_diarization_sond-en-us-callhome-8k-n16k4-pytorch/summary)    |  CallHome (60 小时)  |     12M     | 英文  |

### 时间戳预测模型

|                                                    模型名字                                     |  语言  |    训练数据    | 模型参数 | 备注       |
|:--------------------------------------------------------------------------------------------------:|:--------------:|:-------------------:|:----------:|:---------|
| [TP-Aligner](https://modelscope.cn/models/damo/speech_timestamp_prediction-v1-16k-非实时/summary) |中文| 阿里巴巴语音数据 (50000hours) |   37.8M    | 时间戳模型，中文 |

### 逆文本正则化

|                                                    模型名字                                                    | 语言  |  模型参数  | 备注            |
|:----------------------------------------------------------------------------------------------------------------:|:---:|:------:|:--------------|
| [English](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-en/summary) | EN  | 1.54M  | ITN，语音识别文本后处理 |
| [Russian](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-ru/summary) | RU  | 17.79M | ITN，语音识别文本后处理 |
| [Japanese](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-ja/summary) | JA  |  6.8M  | ITN，语音识别文本后处理 |
| [Korean](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-ko/summary) | KO  | 1.28M  | ITN，语音识别文本后处理 |
| [Indonesian](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-id/summary) | ID  | 2.06M  | ITN，语音识别文本后处理 |
| [Vietnamese](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-vi/summary) | VI  | 0.92M  | ITN，语音识别文本后处理 |
| [Tagalog](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-tl/summary) | TL  | 0.65M  | ITN，语音识别文本后处理 |
| [Spanish](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-es/summary) | ES  | 1.32M  | ITN，语音识别文本后处理 |
| [Portuguese](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-pt/summary) | PT  | 1.28M  | ITN，语音识别文本后处理 |
| [French](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-fr/summary) | FR  | 4.39M  | ITN，语音识别文本后处理 |
| [German](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-de/summary)| GE  | 3.95M  | ITN，语音识别文本后处理 |
