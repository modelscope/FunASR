([简体中文](./README_zh.md)|English)

[![SVG Banners](https://svg-banners.vercel.app/api?type=origin&text1=FunASR🤠&text2=💖%20A%20Fundamental%20End-to-End%20Speech%20Recognition%20Toolkit&width=800&height=210)](https://github.com/modelscope/FunASR)

[![PyPI](https://img.shields.io/pypi/v/funasr)](https://pypi.org/project/funasr/)

<strong>FunASR</strong> hopes to build a bridge between academic research and industrial applications on speech recognition. By supporting the training & finetuning of the industrial-grade speech recognition model, researchers and developers can conduct research and production of speech recognition models more conveniently, and promote the development of speech recognition ecology. ASR for Fun！

<p align="center">
  📖 <b><a href="https://modelscope.github.io/FunASR">Documentation Site</a></b> &nbsp;·&nbsp;
  <a href="https://modelscope.github.io/FunASR/zh/">中文文档</a> &nbsp;·&nbsp;
  <a href="https://modelscope.github.io/FunASR/ja/">日本語</a>
</p>

[**Highlights**](#highlights)
| [**News**](#whats-new)
| [**Installation**](#installation)
| [**Quick Start**](#quick-start)
| [**Tutorial**](https://modelscope.github.io/FunASR/tutorial.html)
| [**Runtime**](./runtime/readme.md)
| [**Model Zoo**](#model-zoo)
| [**Contact**](#contact)

<a name="highlights"></a>

## Highlights

- FunASR is a fundamental speech recognition toolkit that offers a variety of features, including speech recognition (ASR), Voice Activity Detection (VAD), Punctuation Restoration, Language Models, Speaker Verification, Speaker Diarization and multi-talker ASR. FunASR provides convenient scripts and tutorials, supporting inference and fine-tuning of pre-trained models.
- We have released a vast collection of academic and industrial pretrained models on the [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition) and [HuggingFace](https://huggingface.co/FunASR), which can be accessed through our [Model Zoo](https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/model_zoo/modelscope_models.md).

<a name="whats-new"></a>

## What's New

- 2026/05/20: Added Qwen3-ASR (0.6B/1.7B) multi-language speech recognition models, supporting 52 languages with auto language detection. [usage](examples/industrial_data_pretraining/qwen3_asr).
- 2026/05/20: Added GLM-ASR-Nano (1.5B) robust speech recognition model, supporting 17 languages with dialect and low-volume speech optimization. [usage](examples/industrial_data_pretraining/glm_asr).
- 2026/05/19: Fun-ASR-Nano and SenseVoice now support speaker diarization. Use with `vad_model` + `spk_model` to get per-sentence speaker labels. See [Fun-ASR-Nano demo](examples/industrial_data_pretraining/fun_asr_nano/demo_spk.py), [SenseVoice demo](examples/industrial_data_pretraining/sense_voice/demo_spk.py).
- 2025/12/15: [Fun-ASR-Nano-2512](https://github.com/FunAudioLLM/Fun-ASR) is an end-to-end speech recognition large model trained on tens of millions of hours real speech data. Supports 31 languages.
- 2024/10/29: Real-time Transcription Service 1.12 released, 2pass-offline mode supports SenseVoiceSmall model. ([docs](runtime/readme.md))

<details><summary>Full Changelog</summary>

- 2024/10/10: Added support for Whisper-large-v3-turbo model.
- 2024/09/25: Keyword spotting models are new supported (fsmn_kws, sanm_kws).
- 2024/07/04: [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) released.
- 2024/03/05: Qwen-Audio and Whisper-large-v3 support.
- 2024/01/30: FunASR 1.0 released.
- 2023/11/08: File Transcription Service 3.0 released.
- 2023/10/10: Paraformer-VAD-SPK combined pipeline released.
- 2023/09/01: File Transcription Service 2.0 released.
- 2023/08/07: Real-time Transcription Service released.

</details>

<a name="Installation"></a>

## Installation

```bash
pip install funasr

# Or install from source (recommended for latest models)
pip install git+https://github.com/modelscope/FunASR.git
```

<a name="quick-start"></a>

## Quick Start

```shell
funasr ++model=paraformer-zh ++vad_model="fsmn-vad" ++punc_model="ct-punc" ++input=asr_example_zh.wav
```

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
    print(f"[Speaker {sent['spk']}] {sent['text']}")
```

> **China users:** Models download from ModelScope by default.  
> **International users:** Add `hub="hf"` to download from HuggingFace.

→ Full tutorial with all models and usage scenarios: **[Documentation](https://modelscope.github.io/FunASR/tutorial.html)**

## Model Zoo

FunASR has open-sourced a large number of pre-trained models on industrial data. You are free to use, copy, modify, and share FunASR models under the [Model License Agreement](./MODEL_LICENSE).

(⭐ = ModelScope, 🤗 = HuggingFace)

| Model | Task | Training Data | Params |
|-------|------|---------------|--------|
| Fun-ASR-Nano ([⭐](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)) | ASR, 31 languages, timestamps, hotwords | Tens of millions of hours | 800M |
| SenseVoiceSmall ([⭐](https://modelscope.cn/models/iic/SenseVoiceSmall) [🤗](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)) | ASR + emotion + audio events, 5 languages | 300K hours | 234M |
| paraformer-zh ([⭐](https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch) [🤗](https://huggingface.co/funasr/paraformer-zh)) | Chinese ASR, timestamps, non-streaming | 60K hours | 220M |
| paraformer-zh-streaming ([⭐](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online) [🤗](https://huggingface.co/funasr/paraformer-zh-streaming)) | Chinese ASR, streaming | 60K hours | 220M |
| Qwen3-ASR-1.7B ([🤗](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)) | ASR, 52 languages, auto language detection | Multilingual | 1.7B |
| GLM-ASR-Nano ([🤗](https://huggingface.co/zai-org/GLM-ASR-Nano-2512)) | ASR, 17 languages, dialect support | Multilingual | 1.5B |
| ct-punc ([⭐](https://modelscope.cn/models/damo/punc_ct-transformer_cn-en-common-vocab471067-large) [🤗](https://huggingface.co/funasr/ct-punc)) | Punctuation restoration | 100M sentences | 290M |
| fsmn-vad ([⭐](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch) [🤗](https://huggingface.co/funasr/fsmn-vad)) | Voice activity detection | 5K hours | 0.4M |
| cam++ ([⭐](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common) [🤗](https://huggingface.co/funasr/campplus)) | Speaker verification/diarization | 5K hours | 7.2M |
| emotion2vec+large ([⭐](https://modelscope.cn/models/iic/emotion2vec_plus_large) [🤗](https://huggingface.co/emotion2vec/emotion2vec_plus_large)) | Emotion recognition | 40K hours | 300M |
| Whisper-large-v3 ([⭐](https://modelscope.cn/models/iic/Whisper-large-v3)) | ASR, multilingual, timestamps | Multilingual | 1550M |

## Training & Fine-tuning

```bash
cd examples/industrial_data_pretraining/paraformer
bash finetune.sh
```

→ Full training guide: **[Training Documentation](https://modelscope.github.io/FunASR/training.html)**

## ONNX Export

```python
from funasr import AutoModel
model = AutoModel(model="paraformer", device="cpu")
model.export(quantize=False)
```

## Deployment Service

FunASR supports deploying pre-trained or fine-tuned models for service:
- File transcription service (CPU/GPU)
- Real-time transcription service (CPU)

→ [Deployment Documentation](runtime/readme.md)

## Ecosystem

| Project | Description |
|---------|-------------|
| [Fun-ASR-Nano](https://github.com/FunAudioLLM/Fun-ASR) | Latest multi-language ASR large model |
| [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) | Multi-task speech understanding |
| [FunClip](https://github.com/modelscope/FunClip) | AI video clipping powered by FunASR |
| [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | Natural speech generation |

<a name="contact"></a>

## Community

If you encounter problems, raise Issues on the GitHub page. You can also scan the DingTalk QR code to join the community:

<img src="docs/images/dingding.png" width="180"/>

## Contributors

[Contributor list](./Acknowledge.md)

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
