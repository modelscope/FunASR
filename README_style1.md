# FunASR

[![PyPI](https://img.shields.io/pypi/v/funasr)](https://pypi.org/project/funasr/)
[![Stars](https://img.shields.io/github/stars/modelscope/FunASR?style=flat-square)](https://github.com/modelscope/FunASR/stargazers)
[![License](https://img.shields.io/badge/license-MIT-059669)](https://github.com/modelscope/FunASR/blob/main/LICENSE)

Production-ready speech recognition, VAD, punctuation, speaker diarization, emotion detection, and audio event recognition — one unified Python interface.

**50+ languages · 15x faster than Whisper · Single API for the full pipeline**

<p align="center">
  <a href="https://modelscope.github.io/FunASR"><b>📖 Documentation</b></a> ·
  <a href="https://modelscope.github.io/FunASR/zh/"><b>中文文档</b></a> ·
  <a href="https://modelscope.github.io/FunASR/ja/"><b>日本語ドキュメント</b></a>
</p>

## Install

```bash
pip install funasr
```

## Quick Start

```python
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++",
)
res = model.generate(input="meeting.wav")

for sent in res[0]["sentence_info"]:
    print(f"[Speaker {sent['spk']}] {sent['text']}")
```

→ More examples: [Tutorial](https://modelscope.github.io/FunASR/tutorial.html) · [Training](https://modelscope.github.io/FunASR/training.html) · [API Reference](https://modelscope.github.io/FunASR/api.html)

## Models

| Model | Languages | Speed | Use Case |
|-------|-----------|-------|----------|
| [Fun-ASR-Nano](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) | 31 | Medium | Multi-language, dialects, lyrics |
| [Paraformer](https://huggingface.co/funasr/paraformer-zh) | zh/en | Fast | Chinese production ASR |
| [SenseVoice](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | 5 | Ultra-fast | ASR + emotion + audio events |
| [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 52 | Slow | Highest accuracy, context-aware |
| [Paraformer-Streaming](https://huggingface.co/funasr/paraformer-zh-streaming) | zh | Real-time | Live transcription |

Full model zoo: [Model Zoo](./model_zoo)

## What's New

- **2026/05** — Qwen3-ASR (52 languages), GLM-ASR-Nano (17 languages)
- **2026/05** — Fun-ASR-Nano & SenseVoice speaker diarization support
- **2025/12** — Fun-ASR-Nano-2512: 31 languages, tens of millions of hours training data

[Full Changelog →](./CHANGELOG.md)

## Ecosystem

| Project | Description |
|---------|-------------|
| [Fun-ASR-Nano](https://github.com/FunAudioLLM/Fun-ASR) | Multi-language ASR large model |
| [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) | Multi-task speech understanding |
| [FunClip](https://github.com/modelscope/FunClip) | AI video clipping with FunASR |
| [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | Natural speech generation |

## Community

DingTalk group for questions and discussion:

<img src="docs/images/dingding.png" width="180"/>

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
