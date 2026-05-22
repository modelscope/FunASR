<p align="center">
  <h1 align="center">FunASR</h1>
  <p align="center"><b>Speech → Structured Text, in 3 Lines of Python</b></p>
</p>

<p align="center">
  <a href="https://pypi.org/project/funasr/"><img src="https://img.shields.io/pypi/v/funasr?style=flat&color=2563eb" alt="PyPI"/></a>
  <a href="https://github.com/modelscope/FunASR/stargazers"><img src="https://img.shields.io/github/stars/modelscope/FunASR?style=flat&color=b45309" alt="Stars"/></a>
  <a href="https://github.com/modelscope/FunASR/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-059669?style=flat" alt="License"/></a>
  <a href="https://modelscope.github.io/FunASR"><img src="https://img.shields.io/badge/docs-online-blue?style=flat" alt="Docs"/></a>
</p>

<p align="center">
  <a href="./README_zh.md">简体中文</a> ·
  <a href="https://modelscope.github.io/FunASR"><b>Documentation</b></a> ·
  <a href="https://modelscope.github.io/FunASR/tutorial.html">Tutorial</a> ·
  <a href="https://modelscope.github.io/FunASR/training.html">Training</a> ·
  <a href="https://modelscope.github.io/FunASR/api.html">API</a> ·
  <a href="https://arxiv.org/abs/2305.11013">Paper</a>
</p>

---

```python
from funasr import AutoModel

model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad",
                  punc_model="ct-punc", spk_model="cam++")

res = model.generate(input="meeting.wav")
```
```
[Speaker 0]  欢迎大家来体验达摩院推出的语音识别模型。
[Speaker 1]  非常感谢，今天我们主要讨论三个议题。
[Speaker 0]  好的，请开始吧。
```

One API call. Long audio in, speaker-labeled sentences out. VAD segmentation, speech recognition, punctuation, and speaker diarization — handled.

---

## Install

```bash
pip install funasr
```

> **China users**: models auto-download from ModelScope. International users: add `hub="hf"` for HuggingFace.

## Why FunASR

| | |
|---|---|
| **50+ languages** | Fun-ASR-Nano covers 31 languages; Qwen3-ASR covers 52 with auto-detection |
| **15× faster than Whisper** | Non-autoregressive architecture; SenseVoice processes 10s audio in 70ms |
| **Speaker diarization** | "Who said what" — works with Paraformer, Fun-ASR-Nano, and SenseVoice |
| **Emotion & audio events** | SenseVoice detects happy/sad/angry + BGM, laughter, applause |
| **Full lifecycle** | Fine-tune with DeepSpeed → export ONNX → deploy with Docker |

## Models

| Model | Languages | Speed | Best for | Links |
|:------|:---------:|:-----:|:---------|:-----:|
| **[Fun-ASR-Nano](https://github.com/FunAudioLLM/Fun-ASR)** | 31 | ●●●○ | Multi-language, dialects, lyrics | [ModelScope](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) · [HF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) |
| **Paraformer** | 2 | ●●●● | Chinese production ASR | [ModelScope](https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch) · [HF](https://huggingface.co/funasr/paraformer-zh) |
| **SenseVoice** | 5 | ●●●●● | ASR + emotion + audio events | [ModelScope](https://modelscope.cn/models/iic/SenseVoiceSmall) · [HF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) |
| **[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)** | 52 | ●●○○ | Highest accuracy, context-aware | [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B) · [HF](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| **Paraformer-Streaming** | 1 | real-time | Live transcription | [ModelScope](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online) · [HF](https://huggingface.co/funasr/paraformer-zh-streaming) |
| **cam++** | — | — | Speaker diarization | [ModelScope](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common) · [HF](https://huggingface.co/funasr/campplus) |
| **emotion2vec+** | — | — | Speech emotion recognition | [ModelScope](https://modelscope.cn/models/iic/emotion2vec_plus_large) · [HF](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |

Speed: ●●●●● = ultra-fast (70ms/10s audio), ●●○○ = slower (LLM-based). Full list → [model_zoo/](./model_zoo)

## What's New

- **2026.05** — [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B): 52-language LLM-based ASR with auto language detection
- **2026.05** — [GLM-ASR-Nano](https://huggingface.co/zai-org/GLM-ASR-Nano-2512): 17-language robust ASR with dialect optimization
- **2026.05** — Speaker diarization support for Fun-ASR-Nano and SenseVoice
- **2025.12** — [Fun-ASR-Nano](https://github.com/FunAudioLLM/Fun-ASR): 31-language ASR trained on tens of millions of hours
- **2024.07** — [SenseVoice](https://github.com/FunAudioLLM/SenseVoice): multi-task speech understanding (ASR + emotion + events)

<details><summary>Earlier releases</summary>

- 2024.10 — Whisper-large-v3-turbo support
- 2024.09 — Keyword spotting models
- 2024.03 — Qwen-Audio, Whisper-large-v3 support
- 2024.01 — FunASR 1.0 released
- 2023.11 — File transcription service 3.0 (CPU)
- 2023.10 — Paraformer-VAD-SPK combined pipeline
- 2023.08 — Real-time transcription service

</details>

## Ecosystem

| | |
|:--|:--|
| [**Fun-ASR-Nano**](https://github.com/FunAudioLLM/Fun-ASR) | Multi-language ASR large model — 31 languages, timestamps, hotwords |
| [**SenseVoice**](https://github.com/FunAudioLLM/SenseVoice) | Multi-task speech understanding — ASR, emotion, audio events |
| [**FunClip**](https://github.com/modelscope/FunClip) | AI video clipping powered by FunASR + LLM editing |
| [**CosyVoice**](https://github.com/FunAudioLLM/CosyVoice) | Natural speech generation with voice cloning |

## Community

Questions → [GitHub Issues](https://github.com/modelscope/FunASR/issues)  
Discussion → DingTalk group (scan below)

<img src="docs/images/dingding.png" width="150"/>

## Citation

```bibtex
@inproceedings{gao2023funasr,
  author={Zhifu Gao and Zerui Li and Jiaming Wang and Haoneng Luo and Xian Shi and Mengzhe Chen and Yabin Li and Lingyun Zuo and Zhihao Du and Zhangyu Xiao and Shiliang Zhang},
  title={FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
  year={2023},
  booktitle={INTERSPEECH},
}
```

## License

Code: [MIT](./LICENSE) · Model weights: [FunASR Model License](./MODEL_LICENSE)
