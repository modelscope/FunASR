# Python SDK：AutoModel

> **只需 Fun-ASR-Nano 原生推理？** 使用 [Transformers 5.17.0 快速开始](./transformers_native_zh.md)。它加载独立的 `-hf` 权重，不要求 FunASR 工具库。本页保留 `funasr.AutoModel` 工具库路径，两者的依赖、参数和输出不可混用。

[English](python_api.md) | [安装](installation/installation_zh.md) | [模型选择](model_selection_zh.md)

`from funasr import AutoModel` 在当前 Python 进程内运行模型。它不是 HTTP 客户端，也不实现完整的 OpenAI API。本文说明当前代码版本的实现，不代表所有历史 FunASR 版本或上游 checkpoint 的行为。

## 模型构建与推理调用

`AutoModel(**kwargs)` 解析模型配置、查找注册的实现、加载权重，并按需构建 VAD、标点和说话人模型。`model.generate(input, input_len=None, progress_callback=None, **cfg)` 使用已经加载的组件进行推理。在 `generate()` 中传入新的 `model`、`vad_model` 或 `device`，不是重新构建或迁移模型的受支持方式；需要另建实例。

包装层接收 `**kwargs`，但没有覆盖所有模型的统一参数校验。参数能传进去，不等于选定模型会使用它。配置文件可能覆盖下表中的代码回退默认值。请结合 [AutoModel 源码](../funasr/auto/auto_model.py)、[注册表](../funasr/register.py)、[模型下载与解析](../funasr/download/download_model_from_hub.py) 和 [别名映射](../funasr/download/name_maps_from_hub.py) 阅读。

### 构建参数

| 参数 | 代码回退默认值 | 含义 |
|---|---|---|
| `model` | 必填 | 模型平台别名、完整模型 ID 或完整本地模型目录。别名依赖平台，不是版本锁定。 |
| `hub` | `"ms"` | `"ms"`/`"modelscope"` 或 `"hf"`/`"huggingface"`。特殊的 `"openai"` 分支用于 Whisper 加载，不代表 HTTP API 兼容性。 |
| `model_revision` | 平台加载器中为 `"master"` | ModelScope 会向下载器转发此版本。通用 Hugging Face 下载助手当前会忽略它，详见复现说明。 |
| `device` | `"cuda"` | 例如 `"cuda:0"` 或 `"cpu"`。经检查不可用的加速后端会回退到 CPU；显式 `ngpu=0` 也会如此，并设置 `batch_size=1`。 |
| `ngpu` | `1` | 设为零选择 CPU。它不是多 GPU 服务或模型分片配置。 |
| `ncpu` | `4` | 正整数 CPU 线程数；非法值使用回退值，小于 1 的值限制为 1。修改进程级 PyTorch 线程数。 |
| `vad_model`, `punc_model`, `spk_model` | `None` | 可选组件，在构建时加载。正确名称是 `vad_model`，拼错为 `vda_model` 会被拒绝。 |
| `vad_kwargs`, `punc_kwargs`, `spk_kwargs` | `{}` | 子组件配置字典。设备继承主模型；平台和 CPU 线程数在子字典未指定时继承。 |
| `vad_model_revision`, `punc_model_revision`, `spk_model_revision` | 各自为 `"master"` | 分别指定子组件版本，不继承 ASR 版本。这些顶层参数会覆盖子字典中的 `model_revision`。 |
| `spk_mode` | `"punc_segment"` | 使用 `"punc_segment"` 或 `"vad_segment"`。校验还接受历史值 `"default"`，但后续句子构造分支未实现它，请勿选择。 |
| `disable_update` | `False` | 仅禁用 FunASR 包版本检查，不会禁用模型下载。 |
| `disable_pbar` | `False` | 隐藏包装层进度条。 |
| `trust_remote_code` | 平台加载器中为 `False` | 允许模型附带的依赖安装及代码路径；启用前检查实际文件和依赖。 |

### 单次推理参数

| 参数 | 代码回退默认值 | 作用范围与限制 |
|---|---|---|
| `input` | 必填 | 音频路径、URL、波形、受支持的音频 bytes，或输入列表。标点等文本模型接收文本。 |
| `input_len` | `None` | 直接推理包装层仅对单条 `data_type="fbank"` 输入将它转发为 `data_lengths`，不是统一的波形长度控制参数。 |
| `progress_callback` | `None` | 直接推理每批结束后调用 `callback(current, total)`。VAD 流程可能向子组件调用转发它，因此不能当作整个流水线单调递增的全局进度。 |
| `batch_size` | `1` | 直接推理每批输入数量。是否支持批量取决于模型。 |
| `batch_size_s` | `300` | VAD 路径的秒级批处理预算，按最长补齐片段长度乘以片段数量估算，不是文件时长硬上限。CPU 的 VAD 路径按单片段处理。 |
| `batch_size_threshold_s` | `60` | VAD 片段参与组批的时长阈值，不是 VAD 切分参数。 |
| `merge_vad` | `False` | 为本次调用启用 VAD 区域合并。 |
| `merge_length_s` | `15` | VAD 合并目标，在本次参数合并进 ASR 配置之前读取。当前代码版本应在构建时设置它。 |
| `cache` | 包装层不维护会话 | 每次流式调用都传入调用方持有的字典。它不是模型下载缓存。 |
| `sentence_timestamp` | `False` | 请求 VAD 流程输出句子记录，依赖实际时间戳与标点，部分情形可回退到 VAD 区域。 |
| `return_spk_res` | `True` | 配置 VAD 与说话人模型时输出聚类结果。设为 `False` 并不能省去模型构建或片段说话人特征计算。 |
| `preset_spk_num` | `None` | 向聚类后端提供可选的说话人数提示。 |
| `return_spk_center` | `False` | 聚类运行时增加 `spk_embedding_center`。数组类型输出在 JSON 序列化前可能需要转换。 |
| `return_raw_text` | `False` | 在实际应用标点的路径上保留标点前文本，不保证所有路径都返回该字段。 |

`generate()` 在合并本次选项前恢复构建时保存的配置。每次调用都应明确传入本次需要的语言、批处理、热词等选项，不要依赖上一次调用遗留的值。配置重置不代表线程安全，也不重置每个模型属性，例如说话人模式回退会修改 `self.spk_mode`。除非自行验证过并发行为，否则应串行访问共享实例。

## 本地文件与批处理

先按[安装指南](installation/installation_zh.md) 安装 SDK。下面三个示例都是带命令行参数的独立脚本，需要提前准备完整本地模型快照和真实音频。这里只在不下载权重的条件下检查语法与包装层契约，不宣称完成了真实模型推理测试。

第一个示例依次接收本地 ASR 模型目录和一个或多个音频路径，例如 `python transcribe.py /models/paraformer recording.wav recording2.wav`。路径不存在时会在模型构建前失败。

```python
import argparse
from pathlib import Path
from funasr import AutoModel

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("audio", nargs="+")
args = parser.parse_args()
model_dir = Path(args.model_dir).expanduser().resolve(strict=True)
if not model_dir.is_dir():
    raise ValueError("model_dir must be a complete local model directory")
audio = [str(Path(path).expanduser().resolve(strict=True)) for path in args.audio]
model = AutoModel(model=str(model_dir), device="cpu", disable_update=True)
results = model.generate(input=audio, batch_size=1)
for result in results:
    print(result.get("key"), result.get("text", ""))
```

`batch_size=1` 在直接推理路径中逐条处理列表；只有确认模型支持批量并测量过内存占用后再增大它。`batch_size_s` 的含义不同，只控制包装层的 VAD 路径。VAD 切分不是麦克风增量流式识别，也不保证无限录音长度或固定的总内存占用。

波形应为采样率已知的一维单声道 float32 数组。使用通用音频加载器的模型，在 `generate()` 中通过 `fs` 指定数组的原始采样率，回退值为 16000；目标采样率由模型前端决定。不要把立体声矩阵或数字列表默认当作单条波形，Python 列表通常被解释为多条输入。文件解码能力取决于已安装的音频后端。顶层 bytes 由 `load_bytes` 处理：识别出的容器格式会被解码并重采样为 16 kHz，否则按没有采样率元数据的 int16 PCM 解释。输入格式有歧义时，优先使用解码后的数组并明确采样率。参见[音频加载代码](../funasr/utils/load_utils.py) 和 [bytes 输入测试](../tests/test_load_audio_bytes.py)。

`prepare_data_iterator` 也接受清单文件。`.scp` 的一行可以是 `utterance_id /path/to/audio.wav`；`.jsonl` 使用 `{"source": "/path/to/audio.wav", "key": "utterance_id"}`。该迭代器不会把 `.json` 文件当作任意 JSON 数组解析。不要向不可信调用方直接开放任意路径或 URL 输入。

## VAD、时间戳与说话人

下面示例依次接收 **ASR、VAD、标点、说话人模型目录，最后是音频路径**。它面向兼容的本地 Paraformer/SeACo-Paraformer、FSMN-VAD、CT-Transformer 标点与 CAM++ 组合，不适用于任意模型组合。

```python
import argparse
from pathlib import Path
from funasr import AutoModel

parser = argparse.ArgumentParser()
for name in ("asr_dir", "vad_dir", "punc_dir", "spk_dir", "audio"):
    parser.add_argument(name)
args = parser.parse_args()
paths = {name: Path(value).expanduser().resolve(strict=True)
         for name, value in vars(args).items()}
if not all(paths[name].is_dir() for name in ("asr_dir", "vad_dir", "punc_dir", "spk_dir")):
    raise ValueError("All model arguments must be complete local directories")
model = AutoModel(
    model=str(paths["asr_dir"]),
    vad_model=str(paths["vad_dir"]),
    punc_model=str(paths["punc_dir"]),
    spk_model=str(paths["spk_dir"]),
    spk_mode="punc_segment",
    device="cpu",
    disable_update=True,
)
for result in model.generate(input=str(paths["audio"]), return_spk_res=True):
    print(result.get("text", ""))
    for sentence in result.get("sentence_info", []):
        print(sentence.get("start"), sentence.get("end"),
              sentence.get("spk"), sentence.get("text", ""))
```

配置 `vad_model` 后，包装层检测语音区域、按时长排序组批识别、恢复原始顺序，再将时间戳偏移到原始录音坐标。通用说话人聚类在这条 VAD 路径运行；只设置 `spk_model` 不会给直接推理增加说话人分离。`punc_segment` 依赖可用的标点和时间戳，缺少标点或时间戳字段可能使模式变为 `vad_segment`。VAD 边界是语音区域边界，不一定是说话人切换点。`spk=0` 等标签是本次结果内的匿名标签，不是已验证的身份，也不是跨录音稳定的 ID。

没有说话人模型而设置 `sentence_timestamp=True` 时，当前包装层可在标点和 token 时间戳均缺失，或标点与时间戳长度无法对齐时，返回按 VAD 区域对齐的句子记录。其他组合可能返回空的 `sentence_info`，包括已有时间戳但没有标点结果的情形。该开关不会创建强制对齐模型。

### 返回字段

`generate()` 返回 Python `list`，其中是模型结果字典，而不是 OpenAI 响应对象。不要假设所有模型都有全部字段，也不要未经检查直接使用 `results[0]`：VAD 路径中的部分空文本分支会跳过一条结果。没有检测到语音时，VAD 流程通常返回 `{"key": ..., "text": "", "timestamp": []}`。

| 字段 | 含义 |
|---|---|
| `key` | 输入标识，通常是文件名主干或清单中的 key。其他情况下会生成随机 key，不保证全局唯一。 |
| `text` | 解码文本，可能包含模型特有的富文本标签，不一定可直接作为纯文本展示。 |
| `timestamp` | 本文 ASR 路径产生该字段时，通常为 token/词/字的区间对 `[[start_ms, end_ms], ...]`，单位为毫秒。粒度与可用性取决于模型和 checkpoint。 |
| `timestamps` | Nano 的 CTC 对齐可返回包含 `token`、`start_time`、`end_time` 的字典，单位是**秒**。VAD 合并会偏移这些时间，并可额外生成毫秒单位的 `timestamp`。 |
| `sentence_info` | 流水线句子字典，`start`/`end` 单位为**毫秒**，包含 `text`、可选 `timestamp`；完成说话人分配时包含 `spk`。部分路径也有 `sentence`。 |
| `raw_text` | 可选的标点前文本，不保证它是所有可能处理前完全未改动的副本。 |
| `value` | 独立 VAD 模型的语音区间，而非 ASR 文本。在线 VAD 可能输出尚未闭合的边界，应遵循对应模型的流式协议。 |

`words`、`ctc_timestamps`、说话人特征等额外字段属于模型特有输出。不能不经换算就把 SDK 毫秒值当作服务端秒值。参见[时间戳回归测试](../tests/test_paraformer_timestamp_contract.py) 及下列模型源码。

### 使用 SenseVoice 对齐结果

在本工具包的 SenseVoice 路径中，通过 `generate()` 的 `output_timestamp=True` 请求对齐。应将返回的 `words` 与**毫秒**单位的 `timestamp` 配对。这些是模型的对齐单元：英文子词可能合并成词，其他语言则可能保留字符单元。原始富标签文本和 `rich_transcription_postprocess(text)` 都不能作为时间戳的逐字符索引；显示后处理和文本替换不会重新对齐音频。

对含对齐字段的每个结果使用下面的辅助函数。长度检查避免静默截断；缺少字段时应由应用显式处理，不能编造时间戳。

```python
def sensevoice_word_intervals(result):
    words, timestamps = result.get("words"), result.get("timestamp")
    if words is None or timestamps is None:
        raise ValueError("This result has no word/timestamp alignment")
    if len(words) != len(timestamps):
        raise ValueError("Word/timestamp count mismatch")
    return [
        {"word": word, "start_ms": start_ms, "end_ms": end_ms}
        for word, (start_ms, end_ms) in zip(words, timestamps)
    ]
```

将 `model.generate(..., output_timestamp=True)` 返回的结果字典传给 `sensevoice_word_intervals(result)`。两份对齐列表均为空时返回空列表；无语音结果若缺少 `words`，则明确报错。显示文本清理应与对齐序列分开处理。

在 FunASR 1.4.15 的固定 checkpoint 检查中，公开中文样例开启 ITN 得到 242 组配对单元，关闭 ITN 得到 217 组；英文样例有 84 个显示字符，但只有 16 组配对单元。这些结果用于说明消费契约，不是时间戳精度、转写准确率评测，也不代表所有特殊符号问题已修复。参见[可复现输入与验证边界](https://github.com/QwenAudio/SenseVoice/issues/215#issuecomment-5615363096)。这不是原生 Nano Transformers 的输出，也不提供说话人身份识别。

## 语言、热词和对齐取决于模型

| 实现 | 当前代码版本的运行参数 |
|---|---|
| [SeACo-Paraformer](../funasr/models/seaco_paraformer/model.py) | `hotword` 为单数，回退值 `None`：空白分隔的字符串、本地 `.txt` 路径或受支持的 URL。此解析器要求字符串，不是 Nano 的列表形式。ModelScope 的 `paraformer-zh` 别名映射到该实现的 checkpoint。 |
| [Paraformer](../funasr/models/paraformer/model.py) | 指定 `pred_timestamp` 时优先使用它，否则读取默认 `False` 的 `output_timestamp`。checkpoint 保存的配置可能启用时间戳。不能把某一 Paraformer 变体的热词或时间戳行为推广到所有变体。 |
| [SenseVoice](../funasr/models/sense_voice/model.py) | `language="auto"`，语言 token 包含 `zh`、`en`、`yue`、`ja`、`ko`。未知提示回退到 auto token。`use_itn=False`，显式 `text_norm` 优先；`output_timestamp=False`，启用后使用该实现的 CTC 对齐路径。此处不读取通用解码热词参数。 |
| [Fun-ASR-Nano](../funasr/models/fun_asr_nano/model.py) | `hotwords=[]` 为复数，接收字符串列表；`language=None`、`itn=True`。语言直接插入文本提示，不经过 SenseVoice 的语言 token 表归一化。CTC 时间戳取决于已加载的组件和完整权重，而不只是一个输出开关。 |
| [Qwen3-ASR 适配器](../funasr/models/qwen3_asr/model.py) | `language=None`；`auto` 转为 `None`，实现中已有的 `zh`/`en` 等 ISO 别名转为完整语言名。上下文提示为 `context=""`，不是 `hotword`。`return_time_stamps=False` 或 `output_timestamp=False`，任一启用都会请求时间戳，但必须在构建时配置 `forced_aligner`。未配置时适配器会警告并跳过时间戳。 |

这些是参数约定，不是语言覆盖、准确率或不同平台 checkpoint 内容相同的承诺。Nano 缺少所需张量时会禁用 CTC 输出，见 [checkpoint 校验](../funasr/models/fun_asr_nano/checkpoint_utils.py)。MOSS-Transcribe-Diarize 是第三方 **OpenMOSS** 模型，原生联合完成转写与说话人分离，拥有独立的输出结构和依赖路径。请使用 [MOSS 指南](moss_transcribe_diarize_zh.md)，而不是照搬上面的外部 VAD/CAM++ 组合。

### 文本级热词纠正

包装层还支持在 `generate()` 中传入 `postprocess_hotwords`（字符串、列表或字典）和 `postprocess_hotword_file`。显式映射可以写成 `{"wrong": "right"}`，或文件中一行 `wrong=>right`。回退默认值为 `postprocess_hotword_threshold=0.85`、`postprocess_hotword_fuzzy=True`、`return_postprocess_hotword_matches=False`。模糊匹配可能需要 `pypinyin` 和 `rapidfuzz`；设为 `postprocess_hotword_fuzzy=False` 时只做显式替换。

此处理发生在**解码之后**，会更新文本和句子文本，并有意保留与原始识别对齐的时间戳。它既不影响声学解码，也不对替换后的内容重新对齐。将纠正结果用于逐词字幕前，应检查替换明细。这些参数必须按调用传入：`generate()` 向后处理器传递的是本次 `cfg`，不是构建默认配置。参见[实现](../funasr/utils/postprocess_hotwords.py) 和[测试](../tests/test_postprocess_hotwords.py)。

## 流式 Cache 生命周期

下面示例只面向本地 **Paraformer 流式** checkpoint，不代表给离线模型传入 `cache={}` 就能变成流式。运行时依次传入模型目录和非空的单声道 16 kHz 音频文件。示例依据[仓库原有示例](../examples/industrial_data_pretraining/paraformer_streaming/demo.py) 和[流式实现](../funasr/models/paraformer_streaming/model.py)。

```python
import argparse
from pathlib import Path
import soundfile as sf
from funasr import AutoModel

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("audio")
args = parser.parse_args()
model_dir = Path(args.model_dir).expanduser().resolve(strict=True)
if not model_dir.is_dir():
    raise ValueError("model_dir must be a complete local streaming model directory")
speech, sample_rate = sf.read(args.audio, dtype="float32")
if sample_rate != 16000 or speech.ndim != 1 or len(speech) == 0:
    raise ValueError("Expected nonempty mono 16 kHz audio")
model = AutoModel(model=str(model_dir), device="cpu", disable_update=True)
chunk_size = [0, 10, 5]
stride = chunk_size[1] * 960
cache = {}
for start in range(0, len(speech), stride):
    end = min(start + stride, len(speech))
    result = model.generate(
        input=speech[start:end], cache=cache, is_final=end == len(speech),
        chunk_size=chunk_size, encoder_chunk_look_back=4,
        decoder_chunk_look_back=1, batch_size=1,
    )
    print(result)
```

该实现的 `chunk_size` 回退值为 `[0, 10, 5]`，`is_final` 回退值为 `False`。中间值乘以 960 得到输入步长：16 kHz 下 9600 个采样点为 600 ms。源码中 `encoder_chunk_look_back` 和 `decoder_chunk_look_back` 的默认值都是 **0**，示例中的 **4** 和 **1** 是显式示例设置，不是包装层默认值。

每条流创建一个新的 `cache={}`，按顺序处理各块时始终传入同一个字典，保持 chunk/look-back 设置固定，并在最后一块传入 `is_final=True`，冲刷缓冲状态。当前 Paraformer 流式实现在结束时会重新初始化 cache，但结束、取消或开始其他流时仍应丢弃旧字典。不要在不同录音或并发用户之间共享 cache。保持 `batch_size=1`，因为该实现断言每次只有一条波形。返回文本对应本次调用解码的块，不保证是累计全文；应用侧需要保留各次输出。WebSocket 会话协议及服务端缓冲机制与这里的字典生命周期是不同层次。

## 版本复现与离线使用

1. SDK 版本/commit、依赖版本与模型文件必须分别锁定。记录 ASR 及每个辅助模型的平台、完整模型 ID、解析后的上游版本、配置和权重哈希。别名或持续变化的 `master` 不是可复现的版本锁定。
2. 通用 ModelScope 加载器会向快照下载器转发 `model_revision`。通用 Hugging Face 助手虽然接收该参数，当前实际调用却是 **`snapshot_download(model)`**，没有传入 `revision`、`local_files_only` 或 `check_latest`。不能宣称 `hub="hf", model_revision=...` 能锁定这条路径。有些适配器使用自己的加载逻辑，也要检查实际选用的适配器。
3. 离线运行前，应在联网环境准备并验证完整本地快照，包括 tokenizer/前端文件、配置、权重、嵌套编码器/LLM 以及可选对齐器。为所有组件传入已存在的本地目录。本地路径绕过通用平台下载解析，但嵌套模型代码仍可能联网或缺少依赖，必须在实际网络限制下验证。
4. `disable_update=True` 只跳过包更新检查。`check_latest=False` 不保证离线，这些通用下载助手也没有统一的 `local_files_only` 开关。URL 音频和 URL 热词仍需联网；已有缓存的模型别名也可能触发平台请求。
5. 除非审查过的模型代码确实需要，否则保持 `trust_remote_code=False`。ModelScope 通用加载器在信任开启时可以导入 `remote_code`，默认值为 `"model"`；两个通用平台加载器都可能在信任开启时安装模型的 `requirements.txt`，但代码导入行为并不相同。本地快照不等于安全代码。

FunASR 软件采用 [MIT 许可证](../LICENSE)；模型权重、数据集、第三方适配器和依赖可能采用不同许可证。再分发或部署前，请检查具体模型自身的许可证和使用条件。

## SDK 与服务接口边界

| 接口 | 应遵循的契约 |
|---|---|
| Python `AutoModel` | 本文描述的进程内模型配置、波形输入、流式字典及模型特有的 Python 结果。没有 `base_url`、API key 或 HTTP `response_format` 契约。 |
| `funasr-server` | 打包的 [CLI](../funasr/bin/server.py) 创建[服务应用](../funasr/bin/_server_app.py)。`/v1/audio/transcriptions` 是 OpenAI 风格的语音接口子集；`/asr` 是另一个服务特有接口。表单参数、默认值、单位及说话人行为属于该服务，不属于 `generate()`。 |
| OpenAI 兼容示例服务 | [示例服务](../examples/openai_api/server.py) 与打包服务不是同一实现。可查阅它的 [OpenAPI 说明](../examples/openai_api/OPENAPI_zh.md)、[客户端示例](../examples/openai_api/CLIENTS.md) 和[安全/网关指南](../examples/openai_api/SECURITY_zh.md)，再核对实际部署服务的 `/openapi.json`。不要假设二者默认值或响应字段相同。 |
| vLLM | `AutoModelVLLM`、FunASR 模型专用服务与原生 `vllm serve` 是不同入口。SDK 的 `cache` 或一个 HTTP 转写端点，都不能证明兼容实时会话协议。按所选路线与 checkpoint 格式阅读 [vLLM 指南](vllm_guide_zh.md)。 |
| llama.cpp / GGUF | 独立的 C++ 可执行程序与 GGUF 文件，不使用 Python `AutoModel` 的 kwargs 或结果结构。请参考[运行时指南](../runtime/llama.cpp/README.md)。仅有 VAD 不等于说话人分离。 |

[Hydra 推理入口](../funasr/bin/inference.py) 从配置构建 `AutoModel` 后调用 `generate()`，不会把所有 CLI/配置项变成 HTTP 表单字段。开放服务前，应在相应边界落实鉴权、TLS、上传限制、超时及访问限制。OpenAI 客户端的占位 API key 不是访问控制。
