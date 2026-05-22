# FunASR: A Fundamental End-to-End Speech Recognition Toolkit

[![PyPI](https://img.shields.io/pypi/v/funasr)](https://pypi.org/project/funasr/)
[![Stars](https://img.shields.io/github/stars/modelscope/FunASR?style=flat-square)](https://github.com/modelscope/FunASR/stargazers)
[![License](https://img.shields.io/badge/license-MIT-059669)](https://github.com/modelscope/FunASR/blob/main/LICENSE)

([简体中文](./README_zh.md) | English)

FunASR is an open-source speech understanding toolkit providing production-ready ASR, VAD, punctuation restoration, speaker diarization, emotion detection, and audio event recognition with one unified Python interface. It supports 50+ languages at 15x Whisper speed.

<p align="center">
  📖 <a href="https://modelscope.github.io/FunASR"><b>Full Documentation</b></a> &nbsp;|&nbsp;
  <a href="https://modelscope.github.io/FunASR/zh/">中文文档</a> &nbsp;|&nbsp;
  <a href="https://modelscope.github.io/FunASR/ja/">日本語</a>
</p>

---

## Highlights

- **One API for everything**: ASR, VAD, punctuation, speaker diarization, emotion — compose in a single `AutoModel()` call
- **50+ languages**: Fun-ASR-Nano (31 languages), Qwen3-ASR (52 languages with auto-detection)
- **Ultra-fast**: SenseVoice processes 10s audio in 70ms; non-autoregressive architecture
- **Speaker diarization**: "Who said what" for Paraformer, Fun-ASR-Nano, and SenseVoice
- **Train & deploy**: Fine-tune with DeepSpeed, export to ONNX, deploy via Docker or Python SDK
- **Production-proven**: Powers Alibaba Cloud's speech services

## What's New

- **2026/05/20**: Qwen3-ASR (0.6B/1.7B) — 52 languages, auto language detection. [Usage](examples/industrial_data_pretraining/qwen3_asr)
- **2026/05/20**: GLM-ASR-Nano (1.5B) — 17 languages, dialect & low-volume optimization. [Usage](examples/industrial_data_pretraining/glm_asr)
- **2026/05/19**: Fun-ASR-Nano & SenseVoice now support speaker diarization. [Demo](examples/industrial_data_pretraining/fun_asr_nano/demo_spk.py)
- **2025/12/15**: Fun-ASR-Nano-2512 — 31 languages, trained on tens of millions of hours. [Repo](https://github.com/FunAudioLLM/Fun-ASR)
- **2024/10/29**: Real-time transcription service 1.12, SenseVoice support in 2pass-offline

<details><summary>Older releases</summary>

- 2024/10/10: Whisper-large-v3-turbo support
- 2024/09/25: Keyword spotting models (fsmn_kws, sanm_kws)
- 2024/07/04: SenseVoice released
- 2024/03/05: Qwen-Audio support
- 2024/01/30: FunASR 1.0 released

</details>

---

## Installation

```bash
pip install funasr

# Or install from source (latest models & fixes)
pip install git+https://github.com/modelscope/FunASR.git
```

> **China users**: Models download from ModelScope by default (fast in China).  
> **International users**: Add `hub="hf"` to download from HuggingFace.

## Quick Start

### Offline ASR with speaker diarization

```python
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++",
)
res = model.generate(input="meeting.wav", batch_size_s=300)

for sent in res[0]["sentence_info"]:
    print(f"[Speaker {sent['spk']}] [{sent['start']}-{sent['end']}ms] {sent['text']}")
```

### Multi-language (31 languages)

```python
model = AutoModel(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    trust_remote_code=True, remote_code="./model.py",
    vad_model="fsmn-vad", device="cuda:0", hub="hf",
)
res = model.generate(input=["audio.wav"], cache={}, language="auto")
print(res[0]["text"])
```

### Streaming (real-time)

```python
model = AutoModel(model="paraformer-zh-streaming")
cache = {}
for chunk in audio_chunks:
    res = model.generate(input=chunk, cache=cache, is_final=is_last,
                         chunk_size=[0, 10, 5])
    print(res[0]["text"], end="", flush=True)
```

### Emotion detection

```python
model = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad")
res = model.generate(input="audio.wav", language="auto")
# Output includes emotion tags: <|HAPPY|>, <|SAD|>, <|ANGRY|>, <|NEUTRAL|>
```

→ **Full tutorial with all models & scenarios**: [Documentation](https://modelscope.github.io/FunASR/tutorial.html)

---

## Model Zoo

| Model | Task | Languages | Params | Links |
|-------|------|-----------|--------|-------|
| Fun-ASR-Nano | ASR + timestamps + hotwords | 31 | 800M | [⭐](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) |
| SenseVoice | ASR + emotion + audio events | 5 | 234M | [⭐](https://modelscope.cn/models/iic/SenseVoiceSmall) [🤗](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) |
| Paraformer | ASR (non-autoregressive) | zh/en | 220M | [⭐](https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch) [🤗](https://huggingface.co/funasr/paraformer-zh) |
| Paraformer-Streaming | Real-time ASR | zh | 220M | [⭐](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online) [🤗](https://huggingface.co/funasr/paraformer-zh-streaming) |
| Qwen3-ASR | ASR (LLM-based) | 52 | 0.6B/1.7B | [🤗](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| GLM-ASR-Nano | ASR (robust) | 17 | 1.5B | [🤗](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) |
| fsmn-vad | Voice activity detection | zh/en | 0.4M | [⭐](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch) [🤗](https://huggingface.co/funasr/fsmn-vad) |
| ct-punc | Punctuation restoration | zh/en | 290M | [⭐](https://modelscope.cn/models/damo/punc_ct-transformer_cn-en-common-vocab471067-large) [🤗](https://huggingface.co/funasr/ct-punc) |
| cam++ | Speaker verification/diarization | zh | 7.2M | [⭐](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common) [🤗](https://huggingface.co/funasr/campplus) |
| emotion2vec+ | Emotion recognition | multi | 300M | [⭐](https://modelscope.cn/models/iic/emotion2vec_plus_large) [🤗](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |
| Whisper-large-v3 | ASR (multi-task) | multi | 1550M | [⭐](https://modelscope.cn/models/iic/Whisper-large-v3) |

→ Full list: [model_zoo/](./model_zoo)

---

## Training & Fine-tuning

```bash
cd examples/industrial_data_pretraining/paraformer
bash finetune.sh
```

Supports: Multi-GPU (DDP), DeepSpeed ZeRO 1/2/3, dynamic batching, checkpoint averaging.

→ **Full training guide**: [Documentation](https://modelscope.github.io/FunASR/training.html)

---

## ONNX Export

```python
from funasr import AutoModel
model = AutoModel(model="paraformer", device="cpu")
model.export(quantize=False)
```

## Deployment

FunASR supports service deployment for file transcription (CPU/GPU) and real-time streaming.

→ **Deployment docs**: [runtime/readme.md](./runtime/readme.md)

---

## Ecosystem

| Project | Description |
|---------|-------------|
| [Fun-ASR-Nano](https://github.com/FunAudioLLM/Fun-ASR) | Latest multi-language ASR large model |
| [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) | Multi-task speech understanding |
| [FunClip](https://github.com/modelscope/FunClip) | AI video clipping powered by FunASR |
| [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | Natural speech generation |

## Community

DingTalk group:

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
