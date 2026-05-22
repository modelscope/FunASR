# Changelog

## 2026

- **2026/05/20**: Added Qwen3-ASR (0.6B/1.7B) multi-language speech recognition models, supporting 52 languages with auto language detection. [usage](examples/industrial_data_pretraining/qwen3_asr).
- **2026/05/20**: Added GLM-ASR-Nano (1.5B) robust speech recognition model, supporting 17 languages with dialect and low-volume speech optimization. [usage](examples/industrial_data_pretraining/glm_asr).
- **2026/05/19**: Fun-ASR-Nano and SenseVoice now support speaker diarization. Use with `vad_model` + `spk_model` to get per-sentence speaker labels. See [Fun-ASR-Nano demo](examples/industrial_data_pretraining/fun_asr_nano/demo_spk.py), [SenseVoice demo](examples/industrial_data_pretraining/sense_voice/demo_spk.py).

## 2025

- **2025/12/15**: [Fun-ASR-Nano-2512](https://github.com/FunAudioLLM/Fun-ASR) is an end-to-end speech recognition large model trained on tens of millions of hours real speech data. It supports low-latency real-time transcription and covers 31 languages.

## 2024

- **2024/10/29**: Real-time Transcription Service 1.12 released, 2pass-offline mode supports SenseVoiceSmall model. ([docs](runtime/readme.md))
- **2024/10/10**: Added support for Whisper-large-v3-turbo model, multilingual speech recognition, speech translation, and language identification. Download from [ModelScope](examples/industrial_data_pretraining/whisper/demo.py) or [OpenAI](examples/industrial_data_pretraining/whisper/demo_from_openai.py).
- **2024/09/26**: Offline File Transcription Service 4.6, Real-time Transcription Service 1.11 released, fix memory leak & Support SenseVoiceSmall ONNX model. GPU File Transcription Service 2.0, fix GPU memory leak. ([docs](runtime/readme.md))
- **2024/09/25**: Keyword spotting models are new supported. Supports fine-tuning and inference for fsmn_kws, fsmn_kws_mt, sanm_kws, sanm_kws_streaming.
- **2024/07/04**: [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) is a speech foundation model with multiple speech understanding capabilities, including ASR, LID, SER, and AED.
- **2024/07/01**: Offline File Transcription Service GPU 1.1 released, optimize BladeDISC model compatibility. ([docs](runtime/readme.md))
- **2024/06/27**: Offline File Transcription Service GPU 1.0 released, supporting dynamic batch and multi-threading concurrency. Long audio RTF is 0.0076, multi-thread speedup 1200+. ([docs](runtime/readme.md))
- **2024/05/15**: Emotion recognition models released: emotion2vec+large, emotion2vec+base, emotion2vec+seed.
- **2024/05/15**: Offline File Transcription Service 4.5, Real-time Service 1.10, adapting to FunASR 1.0 model structure. ([docs](runtime/readme.md))
- **2024/03/05**: Added Qwen-Audio and Qwen-Audio-Chat audio-text multimodal models. [usage](examples/industrial_data_pretraining/qwen_audio).
- **2024/03/05**: Added Whisper-large-v3 model support.
- **2024/03/05**: Offline Service 4.4, English Service 1.5, Real-time Service 1.9, Docker ARM64 support. ([docs](runtime/readme.md))
- **2024/01/30**: FunASR 1.0 released. ([docs](https://github.com/alibaba-damo-academy/FunASR/discussions/1319))
- **2024/01/30**: Emotion recognition model (emotion2vec_base_finetuned).
- **2024/01/25**: Offline Service 4.2, English Service 1.3, optimized VAD data processing, reduced peak memory usage.
- **2024/01/09**: FunASR SDK for Windows 2.0 released.
- **2024/01/03**: File Transcription Service 4.0, added 8k model support, sentence-level timestamps, improved hotwords.
- **2024/01/03**: Real-time Service 1.6, Ngram language model and WFST hotwords in 2pass-offline mode.

## 2023

- **2023/12/04**: FunASR SDK for Windows 1.0 released.
- **2023/11/08**: Offline File Transcription Service 3.0 (CPU), added punctuation large model, Ngram language model, WFST hotwords. ([docs](runtime/readme.md))
- **2023/10/17**: English offline file transcription service released. ([docs](runtime/readme.md))
- **2023/10/13**: [SlideSpeech](https://slidespeech.github.io/): large scale multi-modal audio-visual corpus.
- **2023/10/10**: Paraformer-VAD-SPK combined pipeline released.
- **2023/10/07**: [FunCodec](https://github.com/alibaba-damo-academy/FunCodec): Neural Speech Codec toolkit.
- **2023/09/01**: Offline File Transcription Service 2.0 (CPU), added ffmpeg, timestamp, hotword support. ([docs](runtime/readme.md))
- **2023/08/07**: Real-time Transcription Service (CPU) released. ([docs](runtime/readme.md))
- **2023/07/17**: BAT: low-latency RNN-T model released. ([BAT](egs/aishell/bat))
- **2023/06/26**: ASRU2023 M2MeT Challenge 2.0 results announced. ([M2MeT2.0](https://alibaba-damo-academy.github.io/FunASR/m2met2/index.html))
