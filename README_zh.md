([English](./README.md)|简体中文|[日本語](./README_ja.md)|[한국어](./README_ko.md))

<p align="center">
<a href="https://github.com/modelscope/FunASR"><img src="https://svg-banners.vercel.app/api?type=origin&text1=FunASR🤠&text2=💖%20A%20Fundamental%20End-to-End%20Speech%20Recognition%20Toolkit&width=800&height=210" alt="FunASR"></a>
</p>

<p align="center">
  <strong>面向离线、流式与边缘部署的工业级语音识别工具箱。</strong><br>
  <em>ASR · VAD · 标点 · 说话人 pipeline · 情感与音频事件模型 · OpenAI 兼容服务</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/funasr/"><img src="https://img.shields.io/pypi/v/funasr" alt="PyPI"></a>
  <a href="https://github.com/modelscope/FunASR"><img src="https://img.shields.io/github/stars/modelscope/FunASR?style=social" alt="Stars"></a>
  <a href="https://pypi.org/project/funasr/"><img src="https://img.shields.io/pypi/dm/funasr" alt="Downloads"></a>
  <a href="https://modelscope.github.io/FunASR/zh/"><img src="https://img.shields.io/badge/文档-在线-blue" alt="Docs"></a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/10479" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10479" alt="modelscope%2FFunASR | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> · <a href="./docs/model_selection_zh.md">模型选择</a> · <a href="#模型列表">模型列表</a> · <a href="./docs/deployment_matrix_zh.md">部署选型</a> · <a href="https://www.funasr.com/">部署中心</a> · <a href="https://www.funasr.com/docs/">文档中心</a> · <a href="#性能评测">性能评测</a> · <a href="./CONTRIBUTING.md">贡献</a>
</p>

---

## 快速开始

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modelscope/FunASR/blob/main/examples/colab/funasr_quickstart.ipynb)

不想先配置本地环境？可以打开 [Colab 快速体验](./examples/colab/README_zh.md) 在浏览器里转写公开样例或上传自己的音频。

FunASR 对你有帮助？欢迎 [Star 项目](https://github.com/modelscope/FunASR)，让更多开发者找到它。

```bash
pip install torch torchaudio
pip install funasr
```

如果要运行 GPU quickstart，请先按 [pytorch.org](https://pytorch.org/get-started/locally/)
选择与你的 NVIDIA driver 匹配的 PyTorch / torchaudio CUDA wheel，再安装 FunASR。
安装后先确认 GPU 可见：

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

只有这里输出 `True` 时才使用 `device="cuda"`；否则请先使用
`device="cpu"`，或重新安装匹配 CUDA 的 PyTorch wheel。

以下 SenseVoiceSmall 示例先使用 CPU，并组合 FSMN-VAD 与 CAM++。

```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad", spk_model="cam++", device="cpu")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")

# AutoModel pipeline 返回带说话人 id 和时间戳的 VAD 分段：
for seg in result[0]["sentence_info"]:
    print(f"[{seg['start']/1000:.1f}s] 说话人{seg['spk']}: {rich_transcription_postprocess(seg['sentence'])}")
```

代码打印实际返回的 VAD 分段起点（秒）、匿名说话人编号和去除 SenseVoice
标签后的文本。文本与分段边界取决于音频和 checkpoint，这里不预设转写结果。

CAM++ 提取 `spk_embedding` 说话人嵌入，`AutoModel` 再聚类并为 VAD 分段分配
说话人编号；编号仅在当前录音内有效，不是已知人物身份，也不是 SenseVoiceSmall
checkpoint 的内置输出。组件和返回字段见 [SDK 契约](./docs/python_api_zh.md)。
需要 GPU 时，先按上文验证环境，再将 `device` 改为 `"cuda"`。
SenseVoice 论文见 [arXiv:2407.04051](https://arxiv.org/abs/2407.04051)，
模型见 [Hugging Face checkpoint](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)，
边缘部署可用 [GGUF checkpoint](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF)。

### LLM 语音识别：Fun-ASR-Nano

Fun-ASR-Nano 是基于 SenseVoice 编码器 + Qwen3-0.6B 解码器的 LLM-ASR，
支持中文、英语、日语，以及 7 种中文方言和 26 种地域口音：

```python
from funasr import AutoModel

model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512", vad_model="fsmn-vad", device="cuda")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")
```

需要 31 语种时，请改用独立的
[Fun-ASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512)
checkpoint。Nano 与 MLT-Nano 的语言范围不同，使用时请按 checkpoint 选择。

使用 vLLM 做高吞吐批处理：

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

model = AutoModelVLLM(model="FunAudioLLM/Fun-ASR-Nano-2512", tensor_parallel_size=1)
results = model.generate(["audio1.wav", "audio2.wav"], language="auto")
```

> **部署为 API 服务：** [本地 SenseVoice CPU 配方](#部署) · [Nano GPU 服务与固定版本 vLLM 环境](./docs/vllm_guide_zh.md)
>
> **接入 AI Agent：** [MCP 服务](examples/mcp_server/) 支持 Claude/Cursor · [OpenAI API](examples/openai_api/README_zh.md) 支持 LangChain/Dify/AutoGen
>
> **接入语音 Agent：** [OpenClaw 实时转写插件](integrations/openclaw/) 支持私有部署的 Talk 与 Voice Call 转写

### 为什么选 FunASR？

FunASR 是工具箱，需要分别选择任务、checkpoint 和运行时。
一个模型或适配器支持的能力，不代表所有服务后端都支持。

| 任务 | Checkpoint 或 pipeline | 运行时入口 | 主要边界 |
|---|---|---|---|
| 文件转写及情感/事件标签 | SenseVoiceSmall | Python `AutoModel`，CPU 或 GPU | 五语种 checkpoint；标签不代表说话人身份。 |
| LLM 文件转写 | Fun-ASR-Nano | `AutoModel`；文档指定的 GPU 拆分引擎 `AutoModelVLLM` | 基础 Nano 覆盖中/英/日及中文方言/口音；时间戳取决于 checkpoint 和路径。 |
| 更多语种的文件转写 | Fun-ASR-MLT-Nano | Python `AutoModel` | 独立的 31 语种 checkpoint，不能把其覆盖范围归给基础 Nano。 |
| 分块实时转写 | Paraformer-zh-streaming | 流式 SDK 或 runtime WebSocket 服务 | 使用流式 checkpoint 和会话独立 cache，不能替换为离线 checkpoint。 |
| 带说话人的文件转写 | SenseVoiceSmall + FSMN-VAD + CAM++ | `AutoModel` 的 VAD 与嵌入聚类 | 编号仅在当前录音内有效，不是已注册人物身份识别。 |
| 联合文本、时间戳与说话人 | 第三方 OpenMOSS 的 MOSS-Transcribe-Diarize | MOSS 指南中的 FunASR adapter 或上游后端 | 离线、录音内匿名标签；统一路径不外接 VAD/说话人 pipeline。 |
| 原生 CPU/端侧转写 | Fun-ASR-Nano 或 SenseVoiceSmall GGUF | llama.cpp runtime | 需要匹配的转换权重，GGUF 不能作为 Python `AutoModel` checkpoint 加载。 |

Checkpoint、接口与协议边界见 [Model Zoo](./model_zoo/readme_zh.md) 和
[部署矩阵](./docs/deployment_matrix_zh.md)。请用目标音频与硬件评测后再选运行时。

第一次试用 FunASR？可以先跑 [Colab 快速体验](./examples/colab/README_zh.md)，再配置本地环境。还不确定先用哪个模型？先看 [模型选择指南](./docs/model_selection_zh.md)。计划从 Whisper 或云端 ASR 切换？请按 [迁移指南](./docs/migration_from_whisper_zh.md) 和 [评测示例](./examples/migration/) 用代表性音频评测、映射功能并安全上线。

---

<a name="性能评测"></a>

## 性能评测

[历史评测报告](https://modelscope.github.io/FunASR/zh/benchmark.html) 与
[拆分引擎测量](./docs/vllm_guide_zh.md#benchmark) 保留原始结果。
两者是独立记录，不能合并成通用速度排名或生产容量承诺。

请按 [RTFx 与可复现评测说明](./docs/benchmark/rtf_reproducibility.md) 对齐
checkpoint/revision、音频集、硬件、批量大小、预热、计时范围与 CER/WER。
离线吞吐量不等于流式延迟；可使用 [迁移评测示例](./examples/migration/)
以相同口径测量自己的录音。

---

## 最新动态

- **MOSS-Transcribe-Diarize** 已接入 FunASR 服务、Docker、Kubernetes、vLLM/SGLang 工作流和 FunClip，一次完成长音频转写、时间戳与匿名说话人标注。[部署 MOSS ->](./docs/moss_transcribe_diarize_zh.md)
- **FunASR 1.4.15** 新增经过测试的 NumPy 2 兼容支持，修复流式 KWS/VAD 边界处理和 checkpoint 排序。升级命令：`python -m pip install -U "funasr==1.4.15"`。[发布说明与验证范围 ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.15)
- **工业部署** 新增更快、更稳定的实时服务，以及覆盖 Linux、macOS、Windows 十种目标的 llama.cpp 预编译包。[GPU 服务 ->](./docs/vllm_guide_zh.md) · [CPU/端侧包 ->](https://www.funasr.com/deploy/llama-cpp.html)

> 完整改动记录和可下载资产请查看 [GitHub Releases](https://github.com/modelscope/FunASR/releases)。

---

## 安装

```bash
pip install funasr
```

<details><summary>从源码安装</summary>

```bash
git clone https://github.com/modelscope/FunASR.git && cd FunASR
pip install -e ./
```
环境要求：Python ≥ 3.8、PyTorch ≥ 1.13、torchaudio

</details>

---

<a name="模型列表"></a>

## 模型列表

列表包含第三方模型。MOSS-Transcribe-Diarize 由 **OpenMOSS** 发布，FunASR
提供适配器，并不拥有其权重。统一路径为离线处理，匿名说话人标签仅在当前录音
内有效，不是实时流式或已知人物身份识别。模型协议与工具箱 MIT 协议分别适用。

| 模型 | 任务 | 语言 | 参数量 | 链接 |
|------|------|------|--------|------|
| **Fun-ASR-Nano** | 识别 | 中/英/日 + 中文方言 | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) [GGUF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF) |
| **Fun-ASR-MLT-Nano** | 识别 | 31 种语言 | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) |
| **SenseVoiceSmall** | 识别 + 情感 + 事件 | 中/英/日/韩/粤 | 234M | [⭐](https://www.modelscope.cn/models/iic/SenseVoiceSmall) [🤗](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) [GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF) [论文](https://arxiv.org/abs/2407.04051) |
| **MOSS-Transcribe-Diarize** | 第三方 OpenMOSS：离线识别 + 时间戳 + 匿名说话人 | 以官方模型卡为准 | 以官方模型卡为准 | [🤗](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) [指南](./docs/moss_transcribe_diarize_zh.md) |
| **Paraformer-zh** | 识别 + 时间戳 | 中/英 | 220M | [⭐](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) [🤗](https://huggingface.co/funasr/paraformer-zh) |
| Paraformer-zh-streaming | 流式识别 | 中/英 | 220M | [⭐](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary) [🤗](https://huggingface.co/funasr/paraformer-zh-streaming) |
| Qwen3-ASR | 识别，52 种语言 | 多语言 | 1.7B | [使用](examples/industrial_data_pretraining/qwen3_asr) |
| GLM-ASR-Nano | 识别，17 种语言 | 多语言 | 1.5B | [使用](examples/industrial_data_pretraining/glm_asr) |
| Whisper-large-v3 | 识别 + 翻译 | 多语言 | 1550M | [使用](examples/industrial_data_pretraining/whisper) |
| Whisper-large-v3-turbo | 识别 + 翻译 | 多语言 | 809M | [使用](examples/industrial_data_pretraining/whisper) |
| ct-punc | 标点恢复 | 中/英 | 290M | [⭐](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/summary) [🤗](https://huggingface.co/funasr/ct-punc) |
| fsmn-vad | 语音检测 | 中/英 | 0.4M | [⭐](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) [🤗](https://huggingface.co/funasr/fsmn-vad) |
| cam++ | 说话人嵌入（speaker embeddings，pipeline 组件） | — | 7.2M | [⭐](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/summary) [🤗](https://huggingface.co/funasr/campplus) |
| emotion2vec+large | 情感识别 | — | 300M | [⭐](https://modelscope.cn/models/iic/emotion2vec_plus_large/summary) [🤗](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |

---

## 使用示例

> [Python 教程](./docs/tutorial/README_zh.md) · [SDK 参数与输出](./docs/python_api_zh.md) · [训练与微调](./docs/training_zh.md) · [模型注册](./docs/model_registration_zh.md)

```python
from funasr import AutoModel

# 中文生产级（VAD + 识别 + 标点 + 说话人）
model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", spk_model="cam++", device="cuda")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav", hotword="关键词 20")

# 使用 Silero VAD（先安装：python -m pip install "funasr[silero]"）
model = AutoModel(
    model="paraformer-zh", vad_model="silero-vad", device="cpu",
    vad_kwargs={"silero_threshold": 0.5, "silero_min_silence_duration_ms": 100},
)
result = model.generate(input="audio.wav")

# 中/英/日 + 中文方言
model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512", hub="hf", trust_remote_code=True,
                  vad_model="fsmn-vad", vad_kwargs={"max_single_segment_time": 30000}, device="cuda")
result = model.generate(input="audio.wav", batch_size=1)

# 流式实时识别(逐块喂音频)
import soundfile as sf
model = AutoModel(model="paraformer-zh-streaming", device="cuda")
audio, sr = sf.read("speech.wav", dtype="float32")   # 16 kHz 单声道
chunk_size = [0, 10, 5]                               # 每块 600ms
chunk_stride = chunk_size[1] * 960
cache = {}
n_chunks = (len(audio) - 1) // chunk_stride + 1
for i in range(n_chunks):
    chunk = audio[i * chunk_stride : (i + 1) * chunk_stride]
    res = model.generate(input=chunk, cache=cache, is_final=(i == n_chunks - 1),
                         chunk_size=chunk_size, encoder_chunk_look_back=4, decoder_chunk_look_back=1)
    if res[0]["text"]:
        print(res[0]["text"], end="", flush=True)

# 情感识别
model = AutoModel(model="emotion2vec_plus_large", device="cuda")
result = model.generate(input="audio.wav", granularity="utterance")
```

### 命令行工具（Agent 友好）

```bash
# 转写音频（最简用法）
funasr audio.wav

# JSON 输出（适合 AI Agent 调用）
funasr audio.wav --output-format json

# 生成 SRT 字幕
funasr audio.wav --output-format srt --output-dir ./subs

# 说话人分离 + 时间戳
funasr audio.wav --spk --timestamps -f json

# 指定模型和语言
funasr audio.wav --model paraformer --language zh

# 批量转写
funasr *.wav --output-format srt --output-dir ./output
```

可用模型：`sensevoice`（默认）、`paraformer`、`paraformer-en`、`fun-asr-nano`


---

## 部署

在新目录中使用 POSIX shell 和 Python 3.11，启动本地 SenseVoice CPU 服务。
下面将 PyPI 发布包安装到独立环境，不是安装当前源码 checkout。服务不内置鉴权，
请保留 loopback 监听；向其他客户端开放前先阅读[安全指南](./examples/openai_api/SECURITY_zh.md)。

```bash
python3.11 -m venv .venv-funasr-http
. .venv-funasr-http/bin/activate
python -m pip install torch torchaudio
python -m pip install funasr fastapi uvicorn python-multipart
python -m pip check
funasr-server --host 127.0.0.1 --port 8000 --model sensevoice --device cpu
```

等待模型下载和服务启动。在第二个终端进入同一目录，使用 curl 7.76+ 下载并转写
公开的中文样例音频。请求使用已预加载的模型，不保证固定文本或说话人标签。

```bash
curl --fail --location https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav && \
curl --fail-with-body http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

需要离线联合 ASR 与匿名说话人标签（`moss-transcribe-diarize`）时，请按
[MOSS 服务、Docker、Kubernetes、vLLM、SGLang、LocalAI 与 FunClip 部署指南 →](./docs/moss_transcribe_diarize_zh.md)
准备独立环境。这是替代服务，不是在上述 CPU 环境中再执行一条命令；复用 8000
端口前先停止 CPU 服务。Nano GPU 服务请遵循[固定版本的分离引擎指南](./docs/vllm_guide_zh.md)，
并检查实际后端加载日志，不能仅凭模型选择就认定已启用 vLLM。

```bash
# Docker 流式服务
docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.12
```

### CPU / 边缘部署 - llama.cpp / GGUF（无需 GPU、无需 Python）

在 CPU 和边缘设备上用单个自包含二进制运行 **SenseVoice / Paraformer / Fun-ASR-Nano**，无需 Python 运行环境，并内置 FSMN-VAD。

```bash
# Linux / macOS：在解压后的发布目录中执行
bash download-funasr-model.sh sensevoice ./gguf        # 也可使用 paraformer 或 nano
./llama-funasr-sensevoice -m ./gguf/sensevoice-small-q8.gguf --vad ./gguf/fsmn-vad.gguf -a audio.wav
# -> 欢迎大家来体验达摩院推出的语音识别模型
```

```powershell
# Windows PowerShell：在解压根目录执行（需已安装 `hf` CLI）
hf download FunAudioLLM/SenseVoiceSmall-GGUF sensevoice-small-q8.gguf --local-dir .\gguf
hf download FunAudioLLM/fsmn-vad-GGUF fsmn-vad.gguf --local-dir .\gguf
.\llama-funasr-sensevoice.exe -m .\gguf\sensevoice-small-q8.gguf --vad .\gguf\fsmn-vad.gguf -a audio.wav
# 使用 windows-x64-vulkan 包，并安装 AMD、Intel 或 NVIDIA 的当前 Vulkan 显卡驱动：
.\llama-funasr-sensevoice.exe -m .\gguf\sensevoice-small-q8.gguf --vad .\gguf\fsmn-vad.gguf -a audio.wav --backend vulkan
# RTX 30 系列等架构 86 GPU 可使用 windows-x64-cuda 包：
.\llama-funasr-sensevoice.exe -m .\gguf\sensevoice-small-q8.gguf --vad .\gguf\fsmn-vad.gguf -a audio.wav --backend cuda
```

Linux GPU 用户可下载 `funasr-llamacpp-linux-x64-vulkan.tar.gz`，在已安装可用
Vulkan driver/ICD 的机器上运行：

```bash
./llama-funasr-sensevoice -m ./gguf/sensevoice-small-q8.gguf --vad ./gguf/fsmn-vad.gguf -a audio.wav --backend vulkan
```

Windows Vulkan ZIP 使用显卡驱动提供的系统 Vulkan loader，不需要另外安装 Vulkan SDK；当前与 Linux Vulkan 包一样，仅加速 SenseVoiceSmall。

带 tag 的发布提供两个 Windows CUDA 包：标准 `windows-x64-cuda` ZIP 面向 CUDA
architecture 86，`windows-x64-cuda-blackwell` 面向 RTX 50 / Blackwell 的 architecture
120（`sm_120`）。两个 ZIP 都包含所需的 cuBLAS DLL，并使用静态 MSVC runtime；用户只需
安装兼容的 NVIDIA 驱动，无需另装 CUDA Toolkit。CI 验证架构与打包边界，但不代表已经在
Blackwell 实机上完成推理验证。

**预编译二进制：** [Releases](https://github.com/modelscope/FunASR/releases) · [v0.2.6](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.6) · [Linux Vulkan tarball](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-linux-x64-vulkan.tar.gz) · [Windows Vulkan zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-vulkan.zip) · [Windows CUDA zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-cuda.zip) · [Windows Blackwell CUDA zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-cuda-blackwell.zip) · **下载与快速开始：** [funasr.com/deploy/llama-cpp](https://www.funasr.com/deploy/llama-cpp.html) · **GGUF 模型：** [Hugging Face](https://huggingface.co/FunAudioLLM) · **文档与评测：** [runtime/llama.cpp/](./runtime/llama.cpp/)

[OpenAI API 示例 →](./examples/openai_api/README_zh.md) · [Gradio Demo →](./examples/openai_api/GRADIO_zh.md) · [客户端配方 →](./examples/openai_api/CLIENTS.md) · [JavaScript/TypeScript 配方 →](./examples/openai_api/JAVASCRIPT_zh.md) · [Kubernetes 模板 →](./examples/openai_api/kubernetes/README_zh.md) · [工作流配方 →](./examples/openai_api/WORKFLOWS_zh.md) · [Postman 集合 →](./examples/openai_api/POSTMAN_zh.md) · [OpenAPI 规范 →](./examples/openai_api/OPENAPI_zh.md) · [安全指南 →](./examples/openai_api/SECURITY_zh.md) · [部署选型 →](./docs/deployment_matrix_zh.md) · [部署文档 →](./runtime/readme_cn.md) · [Agent 集成 →](https://modelscope.github.io/FunASR/agent.html)

---

## 社区

反馈问题前先查阅[故障排查](./docs/troubleshooting_zh.md)，并附上确切模型、运行时、环境与最小复现。

|  |  |
|---|---|
| 📖 [文档](https://modelscope.github.io/FunASR/zh/) | 🐛 [问题反馈](https://github.com/modelscope/FunASR/issues) |
| 💬 [讨论](https://github.com/modelscope/FunASR/discussions) | 🤗 [HuggingFace](https://huggingface.co/funasr) |
| 🤝 [贡献指南](./CONTRIBUTING.md) | 📈 [20k 增长计划](./docs/community_growth_20k.md) |
| 🗺️ [仓库职责与路线图](./docs/repository_roles_zh.md) | 🌐 [funasr.com](https://www.funasr.com) |
| 🧩 [社区集成](./docs/community_projects_zh.md) | 💡 [使用案例](./docs/use_case_showcase_zh.md) |

## Star 趋势

<a href="https://star-history.com/#modelscope/FunASR&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date" width="600" />
 </picture>
</a>

## 许可证

- 本仓库的 FunASR 工具箱源码：[MIT License](./LICENSE)。
- 预训练模型权重单独授权，请以各模型卡标注的协议为准；模型卡若链接本仓库的 [FunASR 模型开源协议](./MODEL_LICENSE)，则适用该协议。

## 引用

```bibtex
@inproceedings{gao2023funasr,
  author={Zhifu Gao and others},
  title={FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
  booktitle={INTERSPEECH},
  year={2023}
}
```
