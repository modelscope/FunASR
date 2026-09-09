# 模型仓库

简体中文 | [English](./readme.md)

先分别确定**模型、权重格式和运行时**。出现在模型列表中，不代表该模型可被
所有服务、导出工具或硬件后端直接使用。

## 按任务选择

| 任务 | 模型系列 | 关键边界 |
| --- | --- | --- |
| 需要上下文能力的文件转写 | [Fun-ASR-Nano](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) | 原始 checkpoint 与原生 vLLM 转换版是不同产物。 |
| 更广的多语言文件转写 | [Fun-ASR-MLT-Nano](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) | 独立 checkpoint，不能把它的语言覆盖写成基础 Nano 的能力。 |
| 带情感、音频事件标签的转写 | [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | 标签不等于说话人身份；说话人感知流水线需按文档组合配套组件。 |
| 中文转写与时间戳 | Paraformer | 离线模型与流式模型的推理契约不同。 |
| 一次请求返回转写、时间戳和匿名说话人 | [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) | OpenMOSS 第三方模型；统一离线路径不需要外部 VAD 或说话人模型，不是已知人物识别。 |

具体场景参见[模型选择](../docs/model_selection_zh.md)，参数与返回值参见
[SDK 参考](../docs/python_api_zh.md)，服务路径参见[部署矩阵](../docs/deployment_matrix_zh.md)。

## 模型用法

### 原生 Transformers checkpoint

[FunAudioLLM/Fun-ASR-Nano-2512-hf](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf)
使用 `AutoProcessor` 和 `AutoModelForSpeechSeq2Seq`，不是原始工具包、原生 vLLM 或
GGUF 的加载路径。[安装与推理指南](../docs/transformers_native_zh.md)固定了上游提交和官方
模型 revision。2026-09-09 核验的稳定版 Transformers 5.16.1 尚未包含该原生模型，
并且这条路径需要匹配的 torchaudio。

### FunASR 工具包

先完成[安装](../docs/installation/installation_zh.md)和
[Python 教程](../docs/tutorial/README_zh.md)。显式选择 hub，并记录实际下载的
checkpoint/revision、FunASR 版本、设备和推理选项。
[模型别名映射](../funasr/download/name_maps_from_hub.py)提供便捷入口，
但别名本身不是不可变的模型版本。

```python
from funasr import AutoModel

model = AutoModel(model="paraformer-zh", hub="ms", device="cpu")
result = model.generate(input="meeting.wav")
print(result[0]["text"])
```

将 `meeting.wav` 替换为实际录音。下载、预热和推理耗时分开记录；
验证时间戳、说话人标签和模型专有标签时，保留原始返回值。

## 语音识别模型

### Paraformer模型

| SDK 别名 | 用途 | ModelScope (`hub="ms"`) | Hugging Face (`hub="hf"`) |
| --- | --- | --- | --- |
| `paraformer-zh` | 中文离线转写；ModelScope 别名指向 SeACo | [SeACo 权重](https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) | [权重](https://huggingface.co/funasr/paraformer-zh) |
| `paraformer-zh-streaming` | 带会话 cache 的分块流式推理 | [权重](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary) | [权重](https://huggingface.co/funasr/paraformer-zh-streaming) |
| `paraformer-en` | 英文离线转写 | 通过 [hub 映射](../funasr/download/name_maps_from_hub.py)解析。 | 通过 hub 映射解析。 |

旧版 [Paraformer VAD/标点组合](https://modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
与 `paraformer-zh-spk` 描述的是组合流水线，不是 `paraformer-zh` 别名对应的权重。需要显式组合时，
按照 [SDK 文档](../docs/python_api_zh.md)设置 `vad_model`、`punc_model`
和 `spk_model`，不要把所有配套组件视为一个可互换的 ASR checkpoint。

## 流水线组件

声纹向量及 SenseVoice 原始／展示标签见[说话人与情感指南](../docs/speaker_emotion_zh.md)。
句末关键词检测见独立的 [KWS 指南](../docs/keyword_spotting_zh.md)。

| 组件 | 别名 | 模型卡 | 不包含的能力 |
| --- | --- | --- | --- |
| 语音活动检测 | `fsmn-vad` | [ModelScope](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) / [HF](https://huggingface.co/funasr/fsmn-vad) | 不转写语音，也不识别人名。 |
| 标点恢复 | `ct-punc` | [ModelScope](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/summary) / [HF](https://huggingface.co/funasr/ct-punc) | 不生成声学时间戳。 |
| 说话人向量 | `cam++` | [hub 映射](../funasr/download/name_maps_from_hub.py) | 没有另行设计的注册与匹配系统时，不识别已知人物。 |
| 时间戳预测 | `fa-zh` | [hub 映射](../funasr/download/name_maps_from_hub.py) | 需符合相应输入与模型路径，不代表所有识别器都支持时间戳。 |

更多 checkpoint 见 [ModelScope 清单](./modelscope_models_zh.md)和
[Hugging Face 清单](./huggingface_models.md)。清单保留部分历史模型，
新部署前应核查对应模型卡。

## 第三方统一转写与说话人分离

MOSS-Transcribe-Diarize 由 **OpenMOSS** 发布，不是 FunASR 团队的自有模型。
统一离线输出包含转写、时间戳和录音内匿名说话人标签；不是实时流式或已知人物身份识别。
适配器、上游原生服务、显存要求与返回边界见
[MOSS 部署指南](../docs/moss_transcribe_diarize_zh.md)。

## 模型许可协议

[FunASR 软件许可证](../LICENSE)不等于所有模型权重使用同一许可证。
请分别查看 checkpoint 的模型卡、许可证、发布方、训练数据说明，以及适用的
[模型许可协议](../MODEL_LICENSE)。分发权重或派生产物时保留上游归属。

## 验证与下一步

- [训练与微调](../docs/training_zh.md)：选择仓库实际支持的配方。
- [注册自定义模型](../docs/model_registration_zh.md)。
- [原生 vLLM 与 split-engine](../docs/vllm_guide_zh.md)：两条路径的权重和协议不能混用。
- [llama.cpp / GGUF](../runtime/llama.cpp/README.md)：ONNX 导出不等于 GGUF 转换。
- [可复现评测](../docs/benchmark/rtf_reproducibility.md)：分别记录质量、性能、硬件和版本。
