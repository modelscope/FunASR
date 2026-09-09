(简体中文 | [English](./README.md))

# Python SDK 教程

> **只需 Fun-ASR-Nano 原生推理？** 使用 [Transformers 5.17.0 快速开始](../transformers_native_zh.md)。它加载独立的 `-hf` 权重，不要求 FunASR 工具库。本页保留 `funasr.AutoModel` 工具库路径，两者的依赖、参数和输出不可混用。

先完成[安装与环境验证](../installation/installation_zh.md)。本页按首次转写、结果检查、VAD、批处理、模型专属选项的顺序介绍。模型选择、语言、依赖和模型卡请查阅[模型仓库](../../model_zoo/readme_zh.md)，不要将某个示例当成所有模型的通用能力。尚未配置本地环境时，可先看 [Colab 快速体验](../../examples/colab/README_zh.md)。

FunASR 软件采用 [MIT 许可](../../LICENSE)。每个模型权重都有各自的许可：请记录完整模型 ID 与 revision，并以对应模型卡为准。只有模型卡明确链接 [FunASR 模型许可协议](../../MODEL_LICENSE) 时，该协议才适用。第三方集成仍属于第三方模型，例如 MOSS-Transcribe-Diarize 来自 OpenMOSS，不是 FunASR 训练的权重。

<a id="模型推理" name="模型推理"></a>
## 1. 完成第一次转写

安装 SDK 及匹配的 PyTorch/torchaudio 环境后，运行下面的 Python 代码。首次执行需要联网下载模型和公开示例 WAV，并预留磁盘与内存。示例基于仓库中的 [Paraformer 示例](../../examples/industrial_data_pretraining/paraformer/demo.py)，不是本次文档审查中记录的推理结果。

```python
from funasr import AutoModel

audio = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
model = AutoModel(
    model="paraformer-zh", hub="ms", device="cpu", ncpu=4,
    disable_update=True, trust_remote_code=False,
)
results = model.generate(input=audio)
for item in results:
    print(item.get("key"), item.get("text", ""))
print("Model directory:", model.model_path)
```

识别自己的录音时，将 `audio` 替换为存在的本地 WAV 路径。建议先使用与模型采样率匹配的短单声道录音，本示例为 16 kHz。文件解码能力取决于音频后端，先检查可读 WAV 比直接尝试任意媒体容器更简单。NumPy 波形没有采样率文件头，需要通过 `fs=sample_rate` 传入真实采样率，参见[音频加载器](../../funasr/utils/load_utils.py)；不要仅修改采样率标签而不重采样。

后续 Python 代码块在同一会话中复用这里的 `AutoModel`、`audio` 和 `model`。断网使用时，请先按[离线检查清单](../installation/installation_zh.md)准备完整本地模型目录和本地输入文件。`disable_update=True` 仅跳过 SDK 启动版本检查。

## 2. 理解参数和结果

`AutoModel(...)` 构建模型与可选流水线组件，`model.generate(input=..., **options)` 执行推理。源码入口：[AutoModel](../../funasr/auto/auto_model.py)、[hub 别名映射](../../funasr/download/name_maps_from_hub.py)，以及具体模型的 `inference()` 实现。

| 参数 | 作用范围与含义 |
| --- | --- |
| `model`、`hub` | 模型 ID/别名或本地目录；默认 ModelScope (`ms`)，`hf` 选择 Hugging Face。同一别名可能对应不同 hub 仓库。 |
| `device`、`ncpu` | 入门显式使用 `cpu`；验证 PyTorch 构建与模型支持后再选择加速器。加载器存在 CPU 回退路径；`ncpu` 控制 PyTorch CPU 线程数。 |
| `vad_model`、`punc_model`、`spk_model` | 独立加载的可选模型；通过 `vad_kwargs`、`punc_kwargs`、`spk_kwargs` 配置。并非所有 ASR 后端都自动支持这些组合。 |
| `batch_size` | 普通非 VAD 路径每个解码 batch 的输入数量，仍受后端限制。 |
| `batch_size_s` | VAD 分段组 batch 的秒数预算，按填充后的片段长度计算，不是文件数。当前 CPU VAD 路径逐段解码。 |
| `batch_size_threshold_s` | VAD 组 batch 启发式使用的片段时长阈值，不是输入时长限制或通用内存上限。 |
| `output_dir` | 可选的后端输出目录；仍会返回 Python 结果。具体文件和格式取决于模型。 |

`generate()` 返回字典列表，通常每条输入录音对应一个结果。使用可选字段前先检查实际键名：

```python
for item in results:
    print(sorted(item.keys()))
    print(item.get("text", ""))
    print(item.get("timestamp", []))
```

Paraformer 路径在权重提供时间戳时，`timestamp` 为字/词元/词级的 `[start_ms, end_ms]` 区间。标点恢复或文本规范化后，不要假定每个可见字符都对应一组区间。其他后端可能返回不同结构的 `timestamps`，请按对应指南解析。无语音时可能返回空文本或空时间戳，不能当成有效转写。

## 3. 添加 VAD、标点与句级时间戳

VAD 检测语音区间，本身不做转写。下面独立检测同一音频：

```python
vad = AutoModel(model="fsmn-vad", device="cpu", disable_update=True)
vad_results = vad.generate(input=audio)
for item in vad_results:
    print(item["key"], item["value"])
```

`value` 为相对于录音起点的 `[start_ms, end_ms]` 语音区间列表。空列表表示没有检测到区间。参见 [FSMN VAD 示例](../../examples/industrial_data_pretraining/fsmn_vad_streaming/demo.py)。

组合分段 ASR 与标点：

```python
pipeline = AutoModel(
    model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu", disable_update=True, trust_remote_code=False,
)
segmented_results = pipeline.generate(
    input=audio, batch_size_s=60, batch_size_threshold_s=30,
    sentence_timestamp=True,
)
for item in segmented_results:
    print(item.get("text", ""))
    for sentence in item.get("sentence_info", []):
        print(sentence.get("start"), sentence.get("end"), sentence.get("text", ""))
```

`max_single_segment_time` 单位为**毫秒**，两个 batch 时长参数单位为**秒**。分段有助于处理较长文件，但不保证任意时长或固定内存占用；音频加载阶段和各模型仍会消耗资源。内存不足时，可缩短录音/片段或减小模型支持的 batch，再重新测量。句子边界依赖可用的时间戳和标点对齐，可能回退到 VAD 片段，应检查真实结果。

需要说话人分割时，在所选组件支持的前提下，为相同流水线添加 `spk_model="cam++"`。遍历每条输入的 `item.get("sentence_info", [])` 后，再读取 `sentence.get("spk")`；不要从外层结果列表直接读取 `spk`。聚类标签不等于已验证的真实身份。参见[说话人组合示例](../../examples/industrial_data_pretraining/sense_voice/demo_spk.py)。

## 4. 处理多条录音

下面故意重复同一示例，演示列表输入，不依赖额外文件。实际使用时替换为本地 WAV 路径：

```python
batch_results = model.generate(input=[audio, audio], batch_size=1)
for index, item in enumerate(batch_results):
    print(index, item.get("key"), item.get("text", ""))
```

仅在模型支持且内存允许时增大 `batch_size`。也可输入 `wav.scp` 文件列表，每行使用 `utterance_id path`，路径相对于进程工作目录解析，每条录音使用唯一 ID。需要模型写出文件时设置 `output_dir`，仅接收返回列表不强制要求该参数。参见[数据列表示例](../../data/list/train_wav.scp)。独立文件的批处理不等于同一句话的流式分块。

## 5. 热词与语言边界

本工作区中 ModelScope 的 `paraformer-zh` 别名映射到 SeACo Paraformer，其实现接受单数 `hotword`，值为以空格分隔的字符串：

```python
biased_results = model.generate(input=audio, hotword="魔搭 达摩院")
print([item.get("text", "") for item in biased_results])
```

这是模型级上下文偏置，不保证一定插入指定词，也不是确定性文本替换。请确认实际解析出的模型，并与不加热词的结果对比。参见 [SeACo 实现](../../funasr/models/seaco_paraformer/model.py)与[上下文 Paraformer 示例](../../examples/industrial_data_pretraining/contextual_paraformer/demo.py)。

本源码工作区的文本后处理是另一种操作：

```python
corrected_results = model.generate(
    input=audio,
    postprocess_hotwords={"科大迅飞": "科大讯飞"},
    return_postprocess_hotword_matches=True,
)
for item in corrected_results:
    print(item.get("text", ""), item.get("postprocess_hotword_matches", []))
```

显式映射替换输出文本中的匹配项。`postprocess_hotword_file` 还支持每行一个目标词或 `错误词=>目标词` 映射。模糊匹配额外需要 `pypinyin` 和 `rapidfuzz`，显式映射不需要。[后处理实现](../../funasr/utils/postprocess_hotwords.py)保留原有时间戳，不会重新对齐修改后的文本；生成字幕或宣称对齐准确前，请审查替换结果。

`hotword`、`hotwords` 和 `language` **不是可互换的通用 SDK 选项**。例如 [Fun-ASR-Nano](../../examples/industrial_data_pretraining/fun_asr_nano/README_zh.md)读取复数 `hotwords` 和模型专属语言提示。修改提示不能把单语言权重变成多语言权重。语言支持、提示取值、流式能力、对齐和依赖版本应按[模型仓库](../../model_zoo/readme_zh.md)及精确模型指南确认，不要跨模型照搬语言数量或安装版本。

## 6. 进入具体工作流

- **流式识别：**[Paraformer 流式示例](../../examples/industrial_data_pretraining/paraformer_streaming/demo.py)。每个音频流维护独立 `cache={}`，最后一块设置 `is_final=True`。16 kHz 下 `[0, 10, 5]` 的 600 ms 对应 **9600 个采样点**，不是 960；分块时长不代表端到端延时。流式 VAD 可能返回 `[start, -1]`、`[-1, end]`、完整区间或空区间，单位为毫秒。
- **标点与对齐：**[标点示例](../../examples/industrial_data_pretraining/ct_transformer/demo.py)与[时间戳预测示例](../../examples/industrial_data_pretraining/monotonic_aligner/demo.py)。对齐需要对应的文本输入，不等同于语音识别。
- **其他模型系列：**[SenseVoice](../../examples/industrial_data_pretraining/sense_voice/README_zh.md)、[Fun-ASR-Nano](../../examples/industrial_data_pretraining/fun_asr_nano/README_zh.md)、[第三方 OpenMOSS 集成](../moss_transcribe_diarize_zh.md)。更多入口统一在模型仓库查找。
- **命令行与服务：**[CLI 参考](../cli.md)、[运行时概览](../../runtime/readme_cn.md)和 [Docker](../installation/docker_zh.md)。Python 参数不代表服务请求具有相同结构。

<a id="模型训练与测试" name="模型训练与测试"></a>
## 训练与验证入口

请使用 [Paraformer 配方](../../examples/industrial_data_pretraining/paraformer/README.md)、[finetune.sh](../../examples/industrial_data_pretraining/paraformer/finetune.sh)和[训练数据示例](../../data/list/train.jsonl)。启动前检查数据集路径、标签对齐、模型许可、GPU 分配和输出目录；训练不是安装冒烟测试。测试训练后权重时，先检查 [infer_from_local.sh](../../examples/industrial_data_pretraining/paraformer/infer_from_local.sh)，确保配置、分词器/前端资源和 checkpoint 路径一致。[验证数据](../../data/list/val.jsonl)应与训练数据分开。

<a id="模型导出与测试" name="模型导出与测试"></a>
## 导出与运行时验证

按照模型的 [Paraformer 导出示例](../../examples/industrial_data_pretraining/paraformer/export.py)和 [ONNX Runtime 指南](../../runtime/python/onnxruntime/README.md)操作。导出支持与额外依赖由具体模型/后端决定。导出成功不表示结果等价；部署前应使用有代表性的输入测试导出制品，并与原模型比较。

<a id="新模型注册教程" name="新模型注册教程"></a>
## 注册自定义模型

参见[注册表教程](./Tables_zh.md)和 [SenseVoice 实现](../../funasr/models/sense_voice/model.py)。完成注册不代表 `generate()` 契约已经满足，模型推理结果必须与计划使用的下游组件匹配。

遇到问题时，带上解释器/依赖版本、实际模型 ID/revision、输入格式和最小复现，回到[常见问题](../troubleshooting_zh.md)。不要在反馈中附带私密音频或凭据。
