([简体中文](./README_zh.md)|English)

[![SVG Banners](https://svg-banners.vercel.app/api?type=origin&text1=FunASR🤠&text2=💖%20A%20Fundamental%20End-to-End%20Speech%20Recognition%20Toolkit&width=800&height=210)](https://github.com/Akshay090/svg-banners)

<p align="center">
  <strong>Industrial speech recognition. 170x faster than Whisper. 50+ languages.</strong><br>
  <em>Speaker diarization · Emotion detection · Streaming · One API call</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/funasr/"><img src="https://img.shields.io/pypi/v/funasr" alt="PyPI"></a>
  <a href="https://github.com/modelscope/FunASR"><img src="https://img.shields.io/github/stars/modelscope/FunASR?style=social" alt="Stars"></a>
  <a href="https://pypi.org/project/funasr/"><img src="https://img.shields.io/pypi/dm/funasr" alt="Downloads"></a>
  <a href="https://modelscope.github.io/FunASR/"><img src="https://img.shields.io/badge/docs-online-blue" alt="Docs"></a>
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/3839" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3839" alt="Trendshift" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> · <a href="#benchmark">Benchmark</a> · <a href="#model-zoo">Models</a> · <a href="https://modelscope.github.io/FunASR/agent.html">Agent Integration</a> · <a href="https://modelscope.github.io/FunASR/">Docs</a>
</p>

---

## Quick Start

```bash
pip install funasr
```

```python
from funasr import AutoModel

# One line: ASR + VAD + punctuation + speaker diarization
model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    spk_model="cam++",
    device="cuda",
)
result = model.generate(input="meeting.wav")

# result[0]["sentence_info"]:
# [{"text": "Let's discuss the Q3 plan.", "start": 420, "end": 3800, "spk": 0},
#  {"text": "Sounds good. I have three points.", "start": 4200, "end": 7100, "spk": 1}]
```

> **Deploy as API server:** `funasr-server --device cuda` → OpenAI-compatible endpoint at localhost:8000
>
> **Use with AI agents:** [MCP Server](examples/mcp_server/) for Claude/Cursor · [OpenAI API](examples/openai_api/) for LangChain/Dify/AutoGen

---

<a name="benchmark"></a>

## Benchmark

> 7 models on GPU, 3 on CPU. 184 long-form Chinese audio files (192 min total). [Full report →](https://modelscope.github.io/FunASR/benchmark.html)

### GPU Results (PyTorch)

| Model | Type | Speed | Highlights |
|-------|------|-------|-----------|
| **SenseVoice-Small** | NAR | **170x** realtime | Fastest + most accurate |
| **Paraformer-Large** | NAR | **120x** realtime | Best speed/accuracy tradeoff |
| Whisper-large-v3-turbo (OpenAI) | AR | 46x realtime | Fastest Whisper variant |
| Whisper-large-v3-turbo (FunASR) | AR | 26x realtime | With VAD pipeline |
| faster-whisper-large-v3 | AR | 21.5x realtime | CTranslate2 optimized |
| **Fun-ASR-Nano** | LLM | 17x realtime | Best LLM-based accuracy |
| Whisper-large-v3 (OpenAI) | AR | 13.4x realtime | Baseline |

### CPU Results

| Model | Speed | Note |
|-------|-------|------|
| **SenseVoice-Small** | **17.2x** realtime | Production-ready on CPU |
| **Paraformer-Large** | **15.6x** realtime | Production-ready on CPU |
| **Fun-ASR-Nano** | 3.6x realtime | Still faster than realtime |
| Whisper models | - | Impractical on CPU (>2h for 192min audio) |

---

## What's new

- 2026/05/24: **v1.3.3** — `funasr-server` CLI, OpenAI-compatible API, MCP Server for AI agents. `pip install --upgrade funasr`
- 2026/05/20: Added Qwen3-ASR (0.6B/1.7B) — 52 languages, auto detection. [usage](examples/industrial_data_pretraining/qwen3_asr)
- 2026/05/20: Added GLM-ASR-Nano (1.5B) — 17 languages, dialect support. [usage](examples/industrial_data_pretraining/glm_asr)
- 2026/05/19: Fun-ASR-Nano and SenseVoice now support speaker diarization.
- 2025/12/15: [Fun-ASR-Nano-2512](https://github.com/FunAudioLLM/Fun-ASR) — 31 languages, tens of millions of hours training.

<details><summary>Older</summary>

- 2024/10/10: Whisper-large-v3-turbo support added.
- 2024/07/04: [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — ASR + emotion + audio events.
- 2024/01/30: FunASR 1.0 released.

</details>

---

## Installation

```bash
pip install funasr
```

<details><summary>From source / Requirements</summary>

```bash
git clone https://github.com/modelscope/FunASR.git && cd FunASR
pip install -e ./
```
Requirements: Python ≥ 3.8, PyTorch ≥ 1.13, torchaudio

</details>

---

<a name="model-zoo"></a>

## Model Zoo

| Model | Task | Languages | Params | Links |
|-------|------|-----------|--------|-------|
| **Fun-ASR-Nano** | ASR + timestamps | 31 languages | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) |
| **SenseVoiceSmall** | ASR + emotion + events | zh/en/ja/ko/yue | 234M | [⭐](https://www.modelscope.cn/models/iic/SenseVoiceSmall) [🤗](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) |
| **Paraformer-zh** | ASR + timestamps | zh/en | 220M | [⭐](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) [🤗](https://huggingface.co/funasr/paraformer-zh) |
| Paraformer-zh-streaming | Streaming ASR | zh/en | 220M | [⭐](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary) [🤗](https://huggingface.co/funasr/paraformer-zh-streaming) |
| Qwen3-ASR | ASR, 52 languages | multilingual | 1.7B | [usage](examples/industrial_data_pretraining/qwen3_asr) |
| GLM-ASR-Nano | ASR, 17 languages | multilingual | 1.5B | [usage](examples/industrial_data_pretraining/glm_asr) |
| ct-punc | Punctuation | zh/en | 290M | [⭐](https://modelscope.cn/models/damo/punc_ct-transformer_cn-en-common-vocab471067-large/summary) [🤗](https://huggingface.co/funasr/ct-punc) |
| fsmn-vad | VAD | zh/en | 0.4M | [⭐](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) [🤗](https://huggingface.co/funasr/fsmn-vad) |
| cam++ | Speaker diarization | — | 7.2M | [⭐](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/summary) [🤗](https://huggingface.co/funasr/campplus) |
| emotion2vec+large | Emotion recognition | — | 300M | [⭐](https://modelscope.cn/models/iic/emotion2vec_plus_large/summary) [🤗](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |

---

## Usage

> Full examples with parameter docs: [Tutorial →](https://modelscope.github.io/FunASR/tutorial.html)

```python
from funasr import AutoModel

# Chinese production (VAD + ASR + punctuation + speaker)
model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", spk_model="cam++", device="cuda")
result = model.generate(input="meeting.wav", hotword="关键词 20")

# 31 languages with timestamps
model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512", hub="hf", trust_remote_code=True,
                  vad_model="fsmn-vad", vad_kwargs={"max_single_segment_time": 30000}, device="cuda")
result = model.generate(input="audio.wav", batch_size=1)

# Streaming real-time
model = AutoModel(model="paraformer-zh-streaming", device="cuda")
result = model.generate(input="chunk.wav", cache={}, chunk_size=[0, 10, 5])

# Emotion recognition
model = AutoModel(model="emotion2vec_plus_large", device="cuda")
result = model.generate(input="audio.wav", granularity="utterance")
```

---

## Deploy

```bash
# OpenAI-compatible API (recommended)
pip install funasr fastapi uvicorn python-multipart
funasr-server --device cuda
# → POST /v1/audio/transcriptions at localhost:8000

# Docker streaming service
docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.12
```

[Deployment docs →](./runtime/readme.md) · [Agent integration →](https://modelscope.github.io/FunASR/agent.html)

---

## Community

|  |  |
|---|---|
| 📖 [Documentation](https://modelscope.github.io/FunASR/) | 🐛 [Issues](https://github.com/modelscope/FunASR/issues) |
| 💬 [Discussions](https://github.com/modelscope/FunASR/discussions) | 🤗 [HuggingFace](https://huggingface.co/funasr) |

## Contributors

<a href="https://github.com/modelscope/FunASR/graphs/contributors"><img src="https://contrib.rocks/image?repo=modelscope/FunASR" /></a>

## License

[MIT License](./LICENSE)

## Citations

```bibtex
@inproceedings{gao2023funasr,
  author={Zhifu Gao and others},
  title={FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
  booktitle={INTERSPEECH},
  year={2023}
}
```
