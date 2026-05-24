([简体中文](./README_zh.md)|English|[日本語](./README_ja.md)|[한국어](./README_ko.md))

<p align="center">
<a href="https://github.com/modelscope/FunASR"><img src="https://svg-banners.vercel.app/api?type=origin&text1=FunASR🤠&text2=💖%20A%20Fundamental%20End-to-End%20Speech%20Recognition%20Toolkit&width=800&height=210" alt="FunASR"></a>
</p>

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
  <a href="#quick-start">Quick Start</a> · <a href="#benchmark">Benchmark</a> · <a href="#model-zoo">Models</a> · <a href="https://modelscope.github.io/FunASR/agent.html">Agent Integration</a> · <a href="https://modelscope.github.io/FunASR/">Docs</a> · <a href="./CONTRIBUTING.md">Contribute</a>
</p>

---

## Quick Start

```bash
pip install funasr
```

```python
from funasr import AutoModel

model = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad", spk_model="cam++", device="cuda")
result = model.generate(input="meeting.wav")
```

**Output** — structured text with speaker labels, timestamps, and punctuation:
```
[00:00.4 → 00:03.8] Speaker 0: Let's discuss the Q3 plan.
[00:04.2 → 00:07.1] Speaker 1: Sounds good. I have three points.
[00:07.5 → 00:12.3] Speaker 0: Go ahead. We have 30 minutes.
```

That's it. **One model, one call** — VAD segmentation, speech recognition, punctuation, speaker diarization all happen automatically.

> **Deploy as API server:** `funasr-server --device cuda` → OpenAI-compatible endpoint at localhost:8000
>
> **Use with AI agents:** [MCP Server](examples/mcp_server/) for Claude/Cursor · [OpenAI API](examples/openai_api/) for LangChain/Dify/AutoGen

### Why FunASR?

| | FunASR | Whisper | Cloud APIs |
|---|---|---|---|
| Speed | **170x realtime** | 13x realtime | ~1x realtime |
| Speaker ID | ✅ Built-in | ❌ Needs pyannote | ✅ Extra cost |
| Emotion | ✅ Happy/Sad/Angry | ❌ | ❌ |
| Languages | 50+ | 57 | Varies |
| Streaming | ✅ WebSocket | ❌ | ✅ |
| vLLM Acceleration | ✅ 2-3x faster | ❌ | N/A |
| Self-hosted | ✅ MIT license | ✅ MIT license | ❌ Cloud only |
| Cost | Free | Free | $0.006/min+ |
| CPU viable | ✅ 17x realtime | ❌ Too slow | N/A |

---

<a name="benchmark"></a>

## Benchmark

> 184 long-form audio files (192 min). [Full report →](https://modelscope.github.io/FunASR/benchmark.html)

| Model | GPU Speed | CPU Speed | vs Whisper-large-v3 |
|-------|-----------|-----------|-------------------|
| **SenseVoice-Small** | **170x** realtime | **17x** realtime | 🚀 **13x faster** |
| **Paraformer-Large** | **120x** realtime | **15x** realtime | 🚀 **9x faster** |
| Whisper-large-v3-turbo | 46x realtime | ❌ | 3.4x faster |
| **Fun-ASR-Nano** | 17x realtime | 3.6x realtime | 1.3x faster |
| Whisper-large-v3 | 13x realtime | ❌ | baseline |

> **Key takeaway:** FunASR models run on CPU faster than Whisper runs on GPU.

---

## What's new

- 2026/05/24: **vLLM Inference Engine** — 2-3x faster LLM decoding for Fun-ASR-Nano. Streaming WebSocket service with VAD + Speaker Diarization. [Guide →](docs/vllm_guide.md)
- 2026/05/24: **Dynamic VAD** — adaptive silence threshold (default on). Short sentences stay intact, long segments get auto-split. [Details →](docs/vllm_guide.md#附录dynamicstreamingvad)
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
| Whisper-large-v3 | ASR + translation | multilingual | 1550M | [usage](examples/industrial_data_pretraining/whisper) |
| Whisper-large-v3-turbo | ASR + translation | multilingual | 809M | [usage](examples/industrial_data_pretraining/whisper) |
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
funasr-server --model sensevoice --device cuda
# → POST /v1/audio/transcriptions at localhost:8000
```

Verify it with a public sample:

```bash
curl -L https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

```bash
# Docker streaming service
docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.12
```

[OpenAI API example →](./examples/openai_api/) · [Deployment docs →](./runtime/readme.md) · [Agent integration →](https://modelscope.github.io/FunASR/agent.html)

---

## Community

|  |  |
|---|---|
| 📖 [Documentation](https://modelscope.github.io/FunASR/) | 🐛 [Issues](https://github.com/modelscope/FunASR/issues) |
| 💬 [Discussions](https://github.com/modelscope/FunASR/discussions) | 🤗 [HuggingFace](https://huggingface.co/funasr) |
| 🤝 [Contributing](./CONTRIBUTING.md) | 📈 [20k growth plan](./docs/community_growth_20k.md) |

## Star History

<a href="https://star-history.com/#modelscope/FunASR&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date" width="600" />
 </picture>
</a>

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
