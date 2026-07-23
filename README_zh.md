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
  <a href="#快速开始">快速开始</a> · <a href="./examples/colab/README_zh.md">Colab</a> · <a href="#性能评测">性能评测</a> · <a href="./docs/model_selection_zh.md">模型选择</a> · <a href="./docs/migration_from_whisper_zh.md">迁移指南</a> · <a href="./docs/use_case_showcase_zh.md">场景速览</a> · <a href="./docs/community_projects_zh.md">社区集成</a> · <a href="./docs/deployment_matrix_zh.md">部署选型</a> · <a href="./docs/troubleshooting_zh.md">排障 FAQ</a> · <a href="#模型列表">模型列表</a> · <a href="https://modelscope.github.io/FunASR/agent.html">Agent 集成</a> · <a href="https://modelscope.github.io/FunASR/zh/">文档</a> · <a href="./CONTRIBUTING.md">贡献</a>
</p>

---

## 快速开始

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modelscope/FunASR/blob/main/examples/colab/funasr_quickstart.ipynb)

不想先配置本地环境？可以打开 [Colab 快速体验](./examples/colab/README_zh.md) 在浏览器里转写公开样例或上传自己的音频。

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

```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad", spk_model="cam++", device="cuda")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")

# AutoModel pipeline 返回带说话人 id 和时间戳的 VAD 分段：
for seg in result[0]["sentence_info"]:
    print(f"[{seg['start']/1000:.1f}s] 说话人{seg['spk']}: {rich_transcription_postprocess(seg['sentence'])}")
```

**输出** — 带说话人标签、时间戳和标点的结构化文本：
```
[0.6s] 说话人0: 欢迎大家来体验达摩院推出的语音识别模型
```

这是一次 `AutoModel` pipeline 调用，实际组合了 SenseVoiceSmall、FSMN-VAD
和 CAM++ 三个独立模型；说话人分离由 CAM++ 提供，并非 SenseVoiceSmall
checkpoint 的内置输出。
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

> **部署为 API 服务：** `funasr-server --device cuda` → 本地 OpenAI 兼容接口 localhost:8000
>
> **接入 AI Agent：** [MCP 服务](examples/mcp_server/) 支持 Claude/Cursor · [OpenAI API](examples/openai_api/README_zh.md) 支持 LangChain/Dify/AutoGen

### 为什么选 FunASR？

Whisper 是单个模型，**FunASR 是一个工具箱**——按场景挑模型：
**Fun-ASR-Nano**（中/英/日及中文方言，需 GPU）、
**Fun-ASR-MLT-Nano**（31 语种）、**SenseVoiceSmall**（五语种 ASR，
并返回情感与音频事件标签）、**Paraformer**（低延迟流式）。下表展示的是
工具箱级能力，并标明由哪个模型或 pipeline 提供：

| | FunASR（工具箱） | Whisper | 云端 API |
|---|---|---|---|
| 最高速度 | **340 倍实时**（Fun-ASR-Nano + vLLM） | 13 倍实时 | ~1 倍实时 |
| 说话人识别 | ✅ 由 VAD + CAM++ pipeline 提供 | ❌ 需要 pyannote | ✅ 额外付费 |
| 情感识别 | ✅ 由 SenseVoice 提供 | ❌ | ❌ |
| 语言数 | 取决于 checkpoint（例如 Qwen3-ASR 52、MLT-Nano 31、Nano 中/英/日） | 57 | 因服务而异 |
| 流式识别 | ✅ WebSocket（Paraformer） | ❌ | ✅ |
| CPU 可用 | ✅ 17 倍实时（SenseVoice） | ❌ 太慢 | 不适用 |
| 私有部署 | ✅ 支持（工具箱 MIT；模型协议各异） | ✅ MIT 开源 | ❌ 仅云端 |
| 费用 | 免费 | 免费 | ¥0.04/分钟起 |

第一次试用 FunASR？可以先跑 [Colab 快速体验](./examples/colab/README_zh.md)，再配置本地环境。还不确定先用哪个模型？先看 [模型选择指南](./docs/model_selection_zh.md)。计划从 Whisper 或云端 ASR 切换？请按 [迁移指南](./docs/migration_from_whisper_zh.md) 和 [评测示例](./examples/migration/) 用代表性音频评测、映射功能并安全上线。

---

<a name="性能评测"></a>

## 性能评测

> 184 条长音频（共 192 分钟）。[完整报告 →](https://modelscope.github.io/FunASR/zh/benchmark.html)

| 模型 | 中文 CER ↓ | GPU 速度 | CPU 速度 | 对比 Whisper-large-v3 |
|------|------|----------|----------|---------------------|
| **Fun-ASR-Nano**（vLLM） | **8.20%** | **340 倍**实时 | — | 🚀 **快 26 倍** |
| **SenseVoice-Small** | **7.81%** | **170 倍**实时 | **17 倍**实时 | 🚀 **快 13 倍** |
| **Paraformer-Large** | 10.18% | **120 倍**实时 | **15 倍**实时 | 🚀 **快 9 倍** |
| Whisper-large-v3-turbo | 21.71% | 46 倍实时 | ❌ | 快 3.4 倍 |
| Whisper-large-v3 | 20.02% | 13 倍实时 | ❌ | 基准 |

> **一句话：** FunASR 在 CPU 上的速度，比 Whisper 在 GPU 上还快。

---

## 最新动态

- 2026/07/24：**v1.3.27 已发布到 PyPI** — OpenAI 兼容服务现在会在 `verbose_json` 中返回 SenseVoice 检测到的语言，并在 vLLM 降级后复用已缓存的 Fun-ASR-Nano `AutoModel`。当 vLLM/VAD 初始化及其 fallback 均失败时，不会残留半初始化的 engine 状态，后续请求可以重试。安装命令：`python -m pip install -U "funasr==1.3.27"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.27)
- 2026/07/23：**llama.cpp runtime v0.1.9** — 新增独立的 Windows Vulkan 包 `funasr-llamacpp-windows-x64-vulkan.zip`，支持在安装当前 AMD、Intel 或 NVIDIA Vulkan 驱动的 Windows 机器上运行 SenseVoiceSmall；Linux Vulkan、Windows CUDA、CPU/AVX2、Linux arm64 和 macOS arm64 包继续提供。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.1.9)
- 2026/07/23：**v1.3.26 已发布到 PyPI** — `funasr-server --model fun-asr-nano --hub ms` 现在会在默认 Fun-ASR-Nano 的 vLLM 路径和 AutoModel fallback 路径中都尊重 ModelScope hub 选择，避免用户指定 ModelScope 时仍误走 Hugging Face 下载。安装命令：`python -m pip install -U "funasr==1.3.26"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.26)
- 2026/07/23：**v1.3.25 已发布到 PyPI** — 实时 WebSocket 服务新增 `POSTPROCESS_HOTWORDS:错词=>正确词` 与 `--postprocess-hotword-file`，可在 final 文本阶段做确定性热词纠正，避免把固定错词修正误用成模型层 `HOTWORDS:` 解码偏置；源码目录下的实时服务入口也可直接运行。安装命令：`python -m pip install -U "funasr==1.3.25"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.25)
- 2026/07/23：**v1.3.24 已发布到 PyPI** — OpenAI 兼容服务现在支持自定义模型路径和 hub 选择，llama.cpp/GGUF 文档补充 HTTP 转写 wrapper 与 Linux Vulkan 包，公开文档链接也已刷新，便于新用户顺利上手。安装命令：`python -m pip install -U "funasr==1.3.24"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.24)
- 2026/07/19：**v1.3.22 已发布到 PyPI** — `funasr-server` 现在会为 SenseVoice/Paraformer fallback 的纯文本结果补齐 OpenAI 兼容 `verbose_json.segments`，避免字幕类客户端在 `text` 已有内容时仍拿到空 `segments` 数组。安装命令：`python -m pip install -U "funasr==1.3.22"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.22)
- 2026/07/19：**v1.3.21 已发布到 PyPI** — 修复全新环境里先安装 `funasr`、尚未选择平台对应 PyTorch 版本时的首次导入阻塞。现在 `import funasr` 和 `funasr.__version__` 不再因为缺少 torch 失败；真正访问 `AutoModel` 时仍会要求安装 PyTorch，并给出明确安装提示。安装命令：`python -m pip install -U "funasr==1.3.21"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.21)
- 2026/07/19：**v1.3.20 已发布到 PyPI** — PyPI 项目页和安装引导已同步到当前 FunASR 文档、社区集成列表，以及 Fun-ASR-Nano 部署路径中带引号的 `python -m pip install -U "funasr>=1.3.19"` 命令。本版本是文档/打包元数据同步，运行时代码与 v1.3.19 保持一致。安装命令：`python -m pip install -U "funasr==1.3.20"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.20)
- 2026/07/19：**v1.3.19 已发布到 PyPI** — 实时 WebSocket 长会话排障文档已随包发布。启动服务时加上 `--enable-spk --log-session-stats-interval 30`，如果仍遇到断连或内存增长，请在 issue 中附上输出的 `Session stats:` 日志。安装命令：`python -m pip install -U "funasr==1.3.19"`。[长会话诊断 →](docs/vllm_guide_zh.md#长会话诊断) · [发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.19)
- 2026/07/19：**v1.3.18 已发布到 PyPI** — CLI 的 SRT/TSV 字幕输出现在会请求句级时间戳，并在需要时加载标点模型；`funasr audio.wav --output-format srt --output-dir ./subs` 会输出分句字幕，不再退化成一个全文字幕块。安装命令：`python -m pip install -U "funasr==1.3.18"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.18)
- 2026/07/18：**v1.3.16 已发布到 PyPI** — Fun-ASR-Nano 实时服务新增客户端分句模式。一个 WebSocket 会话可连续发送 PCM，并用 `COMMIT` 提交每个句子；无需加载服务端 VAD，短句可正常结束，多轮时间戳保持递增。执行 `pip install --upgrade funasr` 后，可用 `funasr-realtime-server --endpoint-mode client` 启动。[使用文档 →](examples/industrial_data_pretraining/fun_asr_nano/docs/realtime_demo.md)
- 2026/07/22：**llama.cpp runtime v0.1.8** — 新增 Linux Vulkan 预编译包 `funasr-llamacpp-linux-x64-vulkan.tar.gz`，可在支持 Vulkan driver/ICD 的 Linux GPU 上运行 `llama-funasr-sensevoice ... --backend vulkan`；CPU、AVX2、macOS arm64、Windows CPU/AVX2、Windows CUDA 包继续保留。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.1.8)
- 2026/07/18：**llama.cpp runtime v0.1.7** — 新增 SenseVoiceSmall 的 Windows CUDA 预编译包 `funasr-llamacpp-windows-x64-cuda.zip`，并保留 Linux / macOS / Windows CPU 包。下载 GGUF 模型后，可在支持的 NVIDIA GPU 上运行 `llama-funasr-sensevoice ... --backend cuda`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.1.7)
- 2026/05/24：**vLLM 推理引擎** — Fun-ASR-Nano 解码加速 2-3 倍。支持流式 WebSocket 服务（VAD + 说话人分离 + 热词）。[文档 →](docs/vllm_guide_zh.md) · [实时 WS 调优 →](docs/vllm_guide_zh.md#67-生产并发与多进程部署) · [API 稳定性清单 →](docs/vllm_guide_zh.md#生产-api-稳定性清单)
- 2026/05/24：**动态 VAD** — 自适应静音阈值（默认开启），短句不切碎、长句自动切分。[详情 →](docs/vllm_guide_zh.md#7-动态-vad)
- 2026/05/24：**v1.3.3** — `funasr-server` 命令行工具、OpenAI 兼容 API、MCP 服务。`pip install --upgrade funasr`
- 2026/05/20：新增 Qwen3-ASR (0.6B/1.7B)，52 种语言自动检测。[使用方法](examples/industrial_data_pretraining/qwen3_asr)
- 2026/05/20：新增 GLM-ASR-Nano (1.5B)，17 种语言，方言优化。[使用方法](examples/industrial_data_pretraining/glm_asr)
- 2026/05/19：Fun-ASR-Nano 和 SenseVoice 可与 VAD、CAM++ 组合为说话人分离 pipeline。
- 2025/12/15：[Fun-ASR-Nano-2512](https://github.com/FunAudioLLM/Fun-ASR) 上线，支持中/英/日及中文方言。

<details><summary>更早</summary>

- 2024/10/10：支持 Whisper-large-v3-turbo。
- 2024/07/04：[SenseVoice](https://github.com/FunAudioLLM/SenseVoice) 发布。
- 2024/01/30：FunASR 1.0 发布。

</details>

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

| 模型 | 任务 | 语言 | 参数量 | 链接 |
|------|------|------|--------|------|
| **Fun-ASR-Nano** | 识别 | 中/英/日 + 中文方言 | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) [GGUF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF) |
| **Fun-ASR-MLT-Nano** | 识别 | 31 种语言 | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) |
| **SenseVoiceSmall** | 识别 + 情感 + 事件 | 中/英/日/韩/粤 | 234M | [⭐](https://www.modelscope.cn/models/iic/SenseVoiceSmall) [🤗](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) [GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF) [论文](https://arxiv.org/abs/2407.04051) |
| **Paraformer-zh** | 识别 + 时间戳 | 中/英 | 220M | [⭐](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) [🤗](https://huggingface.co/funasr/paraformer-zh) |
| Paraformer-zh-streaming | 流式识别 | 中/英 | 220M | [⭐](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary) [🤗](https://huggingface.co/funasr/paraformer-zh-streaming) |
| Qwen3-ASR | 识别，52 种语言 | 多语言 | 1.7B | [使用](examples/industrial_data_pretraining/qwen3_asr) |
| GLM-ASR-Nano | 识别，17 种语言 | 多语言 | 1.5B | [使用](examples/industrial_data_pretraining/glm_asr) |
| Whisper-large-v3 | 识别 + 翻译 | 多语言 | 1550M | [使用](examples/industrial_data_pretraining/whisper) |
| Whisper-large-v3-turbo | 识别 + 翻译 | 多语言 | 809M | [使用](examples/industrial_data_pretraining/whisper) |
| ct-punc | 标点恢复 | 中/英 | 290M | [⭐](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/summary) [🤗](https://huggingface.co/funasr/ct-punc) |
| fsmn-vad | 语音检测 | 中/英 | 0.4M | [⭐](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) [🤗](https://huggingface.co/funasr/fsmn-vad) |
| cam++ | 说话人分离 | — | 7.2M | [⭐](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/summary) [🤗](https://huggingface.co/funasr/campplus) |
| emotion2vec+large | 情感识别 | — | 300M | [⭐](https://modelscope.cn/models/iic/emotion2vec_plus_large/summary) [🤗](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |

---

## 使用示例

> 完整参数文档：[教程 →](https://modelscope.github.io/FunASR/zh/tutorial.html)

```python
from funasr import AutoModel

# 中文生产级（VAD + 识别 + 标点 + 说话人）
model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", spk_model="cam++", device="cuda")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav", hotword="关键词 20")

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

```bash
# OpenAI 兼容 API（推荐）
pip install funasr fastapi uvicorn python-multipart
funasr-server --model sensevoice --device cuda
# → POST /v1/audio/transcriptions，地址 localhost:8000
```

使用公开样例音频验证服务：

```bash
curl -L https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

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

当前 Windows CUDA 包面向 CUDA architecture 86。RTX 50 / Blackwell GPU 会报告
compute capability 12.0（`sm_120`），在专用 CUDA 产物发布前，请使用 CPU 包，或从
源码构建并设置 `-DCMAKE_CUDA_ARCHITECTURES=120`。

**预编译二进制：** [Releases](https://github.com/modelscope/FunASR/releases) · [v0.1.9](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.1.9) · [Linux Vulkan tarball](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.1.9/funasr-llamacpp-linux-x64-vulkan.tar.gz) · [Windows Vulkan zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.1.9/funasr-llamacpp-windows-x64-vulkan.zip) · [Windows CUDA zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.1.9/funasr-llamacpp-windows-x64-cuda.zip) · **下载与快速开始：** [funasr.com/llama-cpp](https://www.funasr.com/llama-cpp.html) · **GGUF 模型：** [Hugging Face](https://huggingface.co/FunAudioLLM) · **文档与评测：** [runtime/llama.cpp/](./runtime/llama.cpp/)

[OpenAI API 示例 →](./examples/openai_api/README_zh.md) · [Gradio Demo →](./examples/openai_api/GRADIO_zh.md) · [客户端配方 →](./examples/openai_api/CLIENTS.md) · [JavaScript/TypeScript 配方 →](./examples/openai_api/JAVASCRIPT_zh.md) · [Kubernetes 模板 →](./examples/openai_api/kubernetes/README_zh.md) · [工作流配方 →](./examples/openai_api/WORKFLOWS_zh.md) · [Postman 集合 →](./examples/openai_api/POSTMAN_zh.md) · [OpenAPI 规范 →](./examples/openai_api/OPENAPI_zh.md) · [安全指南 →](./examples/openai_api/SECURITY_zh.md) · [部署选型 →](./docs/deployment_matrix_zh.md) · [部署文档 →](./runtime/readme_cn.md) · [Agent 集成 →](https://modelscope.github.io/FunASR/agent.html)

---

## 社区

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
