<div align="center">
<img src="docs/images/funasr_logo.jpg" width="400"/>

**A Fundamental End-to-End Speech Recognition Toolkit**

[![PyPI](https://img.shields.io/pypi/v/funasr)](https://pypi.org/project/funasr/)
[![GitHub Stars](https://img.shields.io/github/stars/modelscope/FunASR?style=social)](https://github.com/modelscope/FunASR/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/modelscope/FunASR/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/funasr?color=blue)](https://pypi.org/project/funasr/)

[English](./README.md) | [简体中文](./README_zh.md) | [Documentation](https://modelscope.github.io/FunASR) | [Paper](https://arxiv.org/abs/2305.11013)

</div>

---

FunASR is an industrial-grade speech recognition toolkit developed by [Tongyi Lab, Alibaba Group](https://tongyi.aliyun.com). It provides a unified Python interface for the complete speech understanding pipeline — from raw audio to speaker-attributed, punctuated transcripts — in a single API call.

```python
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh",       # speech recognition
    vad_model="fsmn-vad",        # voice activity detection
    punc_model="ct-punc",        # punctuation restoration
    spk_model="cam++",           # speaker diarization
)
res = model.generate(input="meeting.wav", batch_size_s=300)

for s in res[0]["sentence_info"]:
    print(f"[Speaker {s['spk']}] {s['text']}")
```

**Output:**

> \[Speaker 0\] 欢迎大家来体验达摩院推出的语音识别模型。  
> \[Speaker 1\] 非常感谢，今天我们主要讨论三个议题。  
> \[Speaker 0\] 好的，请开始吧。

---

## Key Features

- **Unified Pipeline** — VAD, ASR, punctuation, speaker diarization, emotion detection composed in one `AutoModel()` call
- **50+ Languages** — Fun-ASR-Nano (31 languages incl. Chinese dialects), Qwen3-ASR (52 languages with auto-detection)
- **High Performance** — Non-autoregressive inference; SenseVoice achieves 70ms RTF for 10s audio (15× faster than Whisper)
- **Speaker Diarization** — Per-sentence speaker labels, compatible with Paraformer, Fun-ASR-Nano, and SenseVoice
- **Emotion & Audio Events** — SenseVoice classifies emotions (happy/sad/angry/neutral) and detects BGM, laughter, applause
- **Production Ready** — Fine-tune with DeepSpeed, export to ONNX, deploy via Docker runtime or Python SDK

## Installation

```bash
pip install -U funasr

# Or install from source for latest models
git clone https://github.com/modelscope/FunASR.git && pip install -e ./FunASR
```

> Models auto-download from [ModelScope](https://modelscope.cn) (fast in China). Add `hub="hf"` for [HuggingFace](https://huggingface.co/FunASR).

## Model Zoo

| Model | Type | Languages | Params | Inference Speed | Links |
|:------|:-----|:---------:|:------:|:---------------:|:-----:|
| [Fun-ASR-Nano](https://github.com/FunAudioLLM/Fun-ASR) | ASR | 31 | 800M | 中 | [MS](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) · [HF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) |
| [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh) | ASR | 2 | 220M | 快 | [MS](https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch) · [HF](https://huggingface.co/funasr/paraformer-zh) |
| [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) | ASR+SER+AED | 5 | 234M | 极快 (70ms/10s) | [MS](https://modelscope.cn/models/iic/SenseVoiceSmall) · [HF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) |
| [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | ASR (LLM) | 52 | 1.7B | 慢 | [MS](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B) · [HF](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| [GLM-ASR-Nano](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) | ASR (LLM) | 17 | 1.5B | 慢 | [MS](https://modelscope.cn/models/ZhipuAI/GLM-ASR-Nano-2512) · [HF](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) |
| Paraformer-Streaming | ASR | 1 | 220M | 实时 | [MS](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online) · [HF](https://huggingface.co/funasr/paraformer-zh-streaming) |
| fsmn-vad | VAD | 2 | 0.4M | — | [MS](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch) · [HF](https://huggingface.co/funasr/fsmn-vad) |
| ct-punc | Punctuation | 2 | 290M | — | [MS](https://modelscope.cn/models/damo/punc_ct-transformer_cn-en-common-vocab471067-large) · [HF](https://huggingface.co/funasr/ct-punc) |
| cam++ | Speaker | — | 7.2M | — | [MS](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common) · [HF](https://huggingface.co/funasr/campplus) |
| emotion2vec+ | Emotion | — | 300M | — | [MS](https://modelscope.cn/models/iic/emotion2vec_plus_large) · [HF](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |

Full model list → [model_zoo/](./model_zoo) &nbsp;|&nbsp; Detailed usage → [Documentation](https://modelscope.github.io/FunASR/tutorial.html)

## What's New

| Date | Update |
|:-----|:-------|
| 2026.05 | [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) — 52 languages, LLM-based, auto language detection |
| 2026.05 | [GLM-ASR-Nano](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) — 17 languages, dialect & low-volume optimization |
| 2026.05 | Speaker diarization support added for Fun-ASR-Nano and SenseVoice |
| 2025.12 | [Fun-ASR-Nano](https://github.com/FunAudioLLM/Fun-ASR) — 31-language end-to-end ASR, trained on tens of millions of hours |
| 2024.07 | [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — multi-task speech understanding (ASR + emotion + events) |

<details><summary>View full changelog</summary>

See [CHANGELOG.md](./CHANGELOG.md) for complete release history.

</details>

## Learn More

| Resource | Description |
|:---------|:------------|
| [Tutorial](https://modelscope.github.io/FunASR/tutorial.html) | Install, choose a model, run ASR/VAD/speaker diarization |
| [Training Guide](https://modelscope.github.io/FunASR/training.html) | Fine-tune Paraformer, SenseVoice, Fun-ASR-Nano on custom data |
| [Developer Guide](https://modelscope.github.io/FunASR/model-registration.html) | Add a new model, understand the registry, test & contribute |
| [API Reference](https://modelscope.github.io/FunASR/api.html) | Auto-generated class & method docs with source links |
| [Runtime / Deployment](./runtime/readme.md) | File transcription service, real-time streaming service (CPU/GPU) |

## Ecosystem

| Project | Description |
|:--------|:------------|
| [Fun-ASR-Nano](https://github.com/FunAudioLLM/Fun-ASR) | Multi-language ASR large model — 31 languages, timestamps, hotwords, speaker diarization |
| [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) | Multi-task speech understanding — ASR, language ID, emotion, audio events |
| [FunClip](https://github.com/modelscope/FunClip) | AI video clipping powered by FunASR and LLM-assisted editing |
| [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | Natural speech generation with multi-language, timbre, and emotion control |

## Community

- Issues & feature requests → [GitHub Issues](https://github.com/modelscope/FunASR/issues)
- Join our DingTalk discussion group:

<img src="docs/images/dingding.png" width="150"/>

## Citation

```bibtex
@inproceedings{gao2023funasr,
  author    = {Zhifu Gao and Zerui Li and Jiaming Wang and Haoneng Luo and Xian Shi and Mengzhe Chen and Yabin Li and Lingyun Zuo and Zhihao Du and Zhangyu Xiao and Shiliang Zhang},
  title     = {FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
  booktitle = {INTERSPEECH},
  year      = {2023},
}
```

## License

Code: [MIT License](./LICENSE) &nbsp;·&nbsp; Model weights: [FunASR Model License](./MODEL_LICENSE) (commercial use permitted with attribution)
