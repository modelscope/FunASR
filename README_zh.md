([English](./README.md)|简体中文|[日本語](./README_ja.md)|[한국어](./README_ko.md))

<p align="center">
<a href="https://github.com/modelscope/FunASR"><img src="https://svg-banners.vercel.app/api?type=origin&text1=FunASR🤠&text2=💖%20A%20Fundamental%20End-to-End%20Speech%20Recognition%20Toolkit&width=800&height=210" alt="FunASR"></a>
</p>

<p align="center">
  <strong>工业级语音识别。比 Whisper 快 170 倍。支持 50+ 语言。</strong><br>
  <em>说话人分离 · 情感识别 · 流式转写 · 一次调用搞定</em>
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
  <a href="#快速开始">快速开始</a> · <a href="./examples/colab/README_zh.md">Colab</a> · <a href="#性能评测">性能评测</a> · <a href="./docs/model_selection_zh.md">模型选择</a> · <a href="./docs/migration_from_whisper_zh.md">迁移指南</a> · <a href="./docs/use_case_showcase_zh.md">场景速览</a> · <a href="./docs/deployment_matrix_zh.md">部署选型</a> · <a href="#模型列表">模型列表</a> · <a href="https://modelscope.github.io/FunASR/agent.html">Agent 集成</a> · <a href="https://modelscope.github.io/FunASR/zh/">文档</a> · <a href="./CONTRIBUTING.md">贡献</a>
</p>

---

## 快速开始

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modelscope/FunASR/blob/main/examples/colab/funasr_quickstart.ipynb)

不想先配置本地环境？可以打开 [Colab 快速体验](./examples/colab/README_zh.md) 在浏览器里转写公开样例或上传自己的音频。

```bash
pip install funasr
```

```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad", spk_model="cam++", device="cuda")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")

# 一次调用即返回带说话人 id 和时间戳的 VAD 分段，可自由渲染：
for seg in result[0]["sentence_info"]:
    print(f"[{seg['start']/1000:.1f}s] 说话人{seg['spk']}: {rich_transcription_postprocess(seg['sentence'])}")
```

**输出** — 带说话人标签、时间戳和标点的结构化文本：
```
[0.6s] 说话人0: 欢迎大家来体验达摩院推出的语音识别模型
```

一个模型、一次调用 — VAD 分段、语音识别、标点恢复、说话人分离全部自动完成。

### LLM 语音识别：Fun-ASR-Nano

追求更高精度、支持 31 种语言（含中文方言），使用 [Fun-ASR-Nano](https://github.com/FunAudioLLM/Fun-ASR) — SenseVoice 编码器 + Qwen3-0.6B 解码器的 LLM-based ASR：

```python
from funasr import AutoModel

model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512", vad_model="fsmn-vad", device="cuda")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")
```

使用 vLLM 加速（批量处理快 16 倍）：

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

model = AutoModelVLLM(model="FunAudioLLM/Fun-ASR-Nano-2512", tensor_parallel_size=1)
results = model.generate(["audio1.wav", "audio2.wav"], language="auto")
```

> **部署为 API 服务：** `funasr-server --device cuda` → 本地 OpenAI 兼容接口 localhost:8000
>
> **接入 AI Agent：** [MCP 服务](examples/mcp_server/) 支持 Claude/Cursor · [OpenAI API](examples/openai_api/README_zh.md) 支持 LangChain/Dify/AutoGen

### 为什么选 FunASR？

| | FunASR | Whisper | 云端 API |
|---|---|---|---|
| 速度 | **170 倍实时** | 13 倍实时 | ~1 倍实时 |
| 说话人识别 | ✅ 内置 | ❌ 需要 pyannote | ✅ 额外付费 |
| 情感识别 | ✅ 开心/悲伤/愤怒 | ❌ | ❌ |
| 语言数 | 50+ | 57 | 因服务而异 |
| 流式识别 | ✅ WebSocket | ❌ | ✅ |
| 私有部署 | ✅ MIT 开源 | ✅ MIT 开源 | ❌ 仅云端 |
| 费用 | 免费 | 免费 | ¥0.04/分钟起 |
| CPU 可用 | ✅ 17 倍实时 | ❌ 太慢 | 不适用 |

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

- 2026/05/24：**vLLM 推理引擎** — Fun-ASR-Nano 解码加速 2-3 倍。支持流式 WebSocket 服务（VAD + 说话人分离 + 热词）。[文档 →](docs/vllm_guide.md)
- 2026/05/24：**动态 VAD** — 自适应静音阈值（默认开启），短句不切碎、长句自动切分。[详情 →](docs/vllm_guide.md#附录dynamicstreamingvad)
- 2026/05/24：**v1.3.3** — `funasr-server` 命令行工具、OpenAI 兼容 API、MCP 服务。`pip install --upgrade funasr`
- 2026/05/20：新增 Qwen3-ASR (0.6B/1.7B)，52 种语言自动检测。[使用方法](examples/industrial_data_pretraining/qwen3_asr)
- 2026/05/20：新增 GLM-ASR-Nano (1.5B)，17 种语言，方言优化。[使用方法](examples/industrial_data_pretraining/glm_asr)
- 2026/05/19：Fun-ASR-Nano 和 SenseVoice 支持说话人分离。
- 2025/12/15：[Fun-ASR-Nano-2512](https://github.com/FunAudioLLM/Fun-ASR) 上线。

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
| **Fun-ASR-Nano** | 识别 + 时间戳 | 31 种语言 | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) |
| **SenseVoiceSmall** | 识别 + 情感 + 事件 | 中/英/日/韩/粤 | 234M | [⭐](https://www.modelscope.cn/models/iic/SenseVoiceSmall) [🤗](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) |
| **Paraformer-zh** | 识别 + 时间戳 | 中/英 | 220M | [⭐](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) [🤗](https://huggingface.co/funasr/paraformer-zh) |
| Paraformer-zh-streaming | 流式识别 | 中/英 | 220M | [⭐](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary) [🤗](https://huggingface.co/funasr/paraformer-zh-streaming) |
| Qwen3-ASR | 识别，52 种语言 | 多语言 | 1.7B | [使用](examples/industrial_data_pretraining/qwen3_asr) |
| GLM-ASR-Nano | 识别，17 种语言 | 多语言 | 1.5B | [使用](examples/industrial_data_pretraining/glm_asr) |
| Whisper-large-v3 | 识别 + 翻译 | 多语言 | 1550M | [使用](examples/industrial_data_pretraining/whisper) |
| Whisper-large-v3-turbo | 识别 + 翻译 | 多语言 | 809M | [使用](examples/industrial_data_pretraining/whisper) |
| ct-punc | 标点恢复 | 中/英 | 290M | [⭐](https://modelscope.cn/models/damo/punc_ct-transformer_cn-en-common-vocab471067-large/summary) [🤗](https://huggingface.co/funasr/ct-punc) |
| fsmn-vad | 语音检测 | 中/英 | 0.4M | [⭐](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) [🤗](https://huggingface.co/funasr/fsmn-vad) |
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

# 31 种语言 + 时间戳
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

> **CPU / 边缘部署(无需 GPU、无需 Python):** 用 **llama.cpp / GGUF** 跑 Fun-ASR-Nano / SenseVoice / Paraformer —— 单个自包含二进制,对标 whisper.cpp。详见 [runtime/llama.cpp/](./runtime/llama.cpp/)。

[OpenAI API 示例 →](./examples/openai_api/README_zh.md) · [Gradio Demo →](./examples/openai_api/GRADIO_zh.md) · [客户端配方 →](./examples/openai_api/CLIENTS.md) · [JavaScript/TypeScript 配方 →](./examples/openai_api/JAVASCRIPT_zh.md) · [Kubernetes 模板 →](./examples/openai_api/kubernetes/README_zh.md) · [工作流配方 →](./examples/openai_api/WORKFLOWS_zh.md) · [Postman 集合 →](./examples/openai_api/POSTMAN_zh.md) · [OpenAPI 规范 →](./examples/openai_api/OPENAPI_zh.md) · [安全指南 →](./examples/openai_api/SECURITY_zh.md) · [部署选型 →](./docs/deployment_matrix_zh.md) · [部署文档 →](./runtime/readme_cn.md) · [Agent 集成 →](https://modelscope.github.io/FunASR/agent.html)

---

## 社区

|  |  |
|---|---|
| 📖 [文档](https://modelscope.github.io/FunASR/zh/) | 🐛 [问题反馈](https://github.com/modelscope/FunASR/issues) |
| 💬 [讨论](https://github.com/modelscope/FunASR/discussions) | 🤗 [HuggingFace](https://huggingface.co/funasr) |
| 🤝 [贡献指南](./CONTRIBUTING.md) | 📈 [20k 增长计划](./docs/community_growth_20k.md) |

## Star 趋势

<a href="https://star-history.com/#modelscope/FunASR&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date" width="600" />
 </picture>
</a>

## 许可证

[MIT License](./LICENSE)

## 引用

```bibtex
@inproceedings{gao2023funasr,
  author={Zhifu Gao and others},
  title={FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
  booktitle={INTERSPEECH},
  year={2023}
}
```
