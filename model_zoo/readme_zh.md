(简体中文|[English](./readme.md))

# 模型仓库

## 模型许可协议
您可以在本协议的条件下自由使用、复制、修改和分享FunASR模型。在使用、复制、修改和分享FunASR模型时，您应当标明模型来源和作者信息。您应当在[FunASR软件]中保留相关模型的名称。完整的模型许可证请参见 [模型许可协议](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)

## 模型用法
模型用法参考[文档](../runtime/quick_start_zh.md)

## 模型仓库
这里我们提供了在不同数据集上预训练的模型。模型和数据集的详细信息可在 [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition)中找到.

### 语音识别模型
#### Paraformer模型

（注：[🤗]()表示Huggingface模型仓库链接，[⭐]()表示ModelScope模型仓库链接）

|                                                                              模型名字                                                                              |         任务详情          |     训练数据     | 参数量  |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------:|:------------:|:----:|
|      paraformer-zh <br> ([⭐](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)  [🤗]() )       |    语音识别，带时间戳输出，非实时    |  60000小时，中文  | 220M |
| SeACoParaformer-zh <br> ( [⭐](https://www.modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)  [🤗]() ) | 带热词功能的语音识别，带时间戳输出，非实时 |  60000小时，中文  | 220M |
|              paraformer-zh-spk <br> ( [⭐](https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn/summary)  [🤗]() )              |  分角色语音识别，带时间戳输出，非实时   |  60000小时，中文  | 220M |
|    paraformer-zh-streaming <br> ( [⭐](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary) [🤗]() )    |        语音识别，实时        |  60000小时，中文  | 220M |
| paraformer-zh-streaming-small <br> ( [⭐](https://www.modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online/summary) [🤗]() ) |        语音识别，实时        |  60000小时，中文  | 220M |
| paraformer-en <br> ( [⭐](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020/summary) [🤗]() )       |       语音识别，非实时        |  50000小时，英文  | 220M |
