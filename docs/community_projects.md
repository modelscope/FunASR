[English](./community_projects.md) | [简体中文](./community_projects_zh.md)

# FunASR Community Integrations

This page lists maintained projects where FunASR, Fun-ASR-Nano, SenseVoice, or an official FunASR model is already integrated upstream. Each entry below was checked against the project's default branch and, where available, a linked merged change.

These integrations are community-maintained. Their release cadence, hardware support, and API stability are controlled by the upstream project, not by the FunASR maintainers.

## Voice agents and applications

| Project | What is integrated | Start here |
|---|---|---|
| [Pipecat](https://github.com/pipecat-ai/pipecat) | A local `FunASRSTTService` backed by SenseVoice for voice and multimodal agent pipelines, with transcription and voice-agent examples. | [FunASR service docs](https://docs.pipecat.ai/api-reference/server/services/stt/funasr), [transcription example](https://github.com/pipecat-ai/pipecat/blob/main/examples/transcription/transcription-funasr.py), [voice-agent example](https://github.com/pipecat-ai/pipecat/blob/main/examples/voice/voice-funasr.py), and merged [#4844](https://github.com/pipecat-ai/pipecat/pull/4844). |
| [xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server) | The default local ASR provider uses FunASR with SenseVoiceSmall; recognition language can be set to `auto`, `zh`, `en`, `ja`, `ko`, or `yue`. A separate FunASR runtime-server provider is also available. | [Provider implementation](https://github.com/xinnan-tech/xiaozhi-esp32-server/blob/main/main/xiaozhi-server/core/providers/asr/fun_local.py), [configuration](https://github.com/xinnan-tech/xiaozhi-esp32-server/blob/main/main/xiaozhi-server/config.yaml), and merged [#3255](https://github.com/xinnan-tech/xiaozhi-esp32-server/pull/3255). |
| [AutoSubs](https://github.com/tmoroney/auto-subs) | An on-device int8 ONNX SenseVoice engine for subtitle generation in DaVinci Resolve, Premiere, and After Effects. | [SenseVoice model notes](https://github.com/tmoroney/auto-subs#sensevoice), [engine source](https://github.com/tmoroney/auto-subs/blob/main/AutoSubs-App/src-tauri/crates/transcription-engine/src/engines/sense_voice.rs), and merged [#629](https://github.com/tmoroney/auto-subs/pull/629). |
| [TMSpeech](https://github.com/jxlpzqc/TMSpeech) | A Windows meeting subtitle and translation app that can run Fun-ASR-Nano INT8 through its external-command recognizer. Silero VAD completes each utterance before Nano decoding, while the existing low-cost C# streaming recognizer remains available. | [Fun-ASR-Nano setup](https://github.com/jxlpzqc/TMSpeech/blob/master/README.md#%E4%BD%BF%E7%94%A8-fun-asr-nano), [recognizer source](https://github.com/jxlpzqc/TMSpeech/blob/master/external_recognizer/simulate-streaming-funasr-nano.py), and merged [#110](https://github.com/jxlpzqc/TMSpeech/pull/110). |
| [OmniVoice Studio](https://github.com/debpalash/OmniVoice-Studio) | Its OpenAI-compatible remote ASR backend can point dictation and dubbing workflows at a self-hosted FunASR or SenseVoice server. | [OpenAI-compatible ASR guide](https://github.com/debpalash/OmniVoice-Studio/blob/main/docs/engines/openai-compatible-asr.md) and merged [#1003](https://github.com/debpalash/OmniVoice-Studio/pull/1003). |
| [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | Dataset preparation and WebUI transcription with Fun-ASR-Nano, SenseVoice, and classic FunASR models. | [`funasr_asr.py`](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/tools/asr/funasr_asr.py), runtime fallback [#2801](https://github.com/RVC-Boss/GPT-SoVITS/pull/2801), and backend documentation [#2803](https://github.com/RVC-Boss/GPT-SoVITS/pull/2803). |
| [AudioNotes](https://github.com/harry0703/AudioNotes) | Audio and video note extraction to structured Markdown with Fun-ASR-MLT-Nano routed through the Fun-ASR Nano inference profile, including cache, batch size, and list-based hotwords. | [Project README](https://github.com/harry0703/AudioNotes#readme), [FunASR service](https://github.com/harry0703/AudioNotes/blob/main/app/services/asr_funasr.py), and merged [#65](https://github.com/harry0703/AudioNotes/pull/65). |
| [LiveTalking](https://github.com/lipku/LiveTalking) | A real-time interactive digital-human server with a local FunASR/SenseVoice ASR server path; the merged fix serializes shared model access so concurrent requests do not race during lazy model loading or `generate()`. | [ASR server](https://github.com/lipku/LiveTalking/blob/main/server/asr_server.py), [project README](https://github.com/lipku/LiveTalking#readme), and merged [#611](https://github.com/lipku/LiveTalking/pull/611). |

## Model serving and runtimes

| Project | What is integrated | Start here |
|---|---|---|
| [Xinference](https://github.com/xorbitsai/inference) | Built-in audio model specifications for SenseVoiceSmall, Fun-ASR-Nano-2512, and Fun-ASR-MLT-Nano-2512 through Xinference's unified inference API. | [Audio model specifications](https://github.com/xorbitsai/inference/blob/main/xinference/model/audio/model_spec.json) and the FunASR 1.3 compatibility update in merged [#5140](https://github.com/xorbitsai/inference/pull/5140). |
| [Fun-ASR-vLLM](https://github.com/yuekaizhang/Fun-ASR-vllm) | Community vLLM inference for Fun-ASR-Nano and Fun-ASR-MLT-Nano, including batch evaluation and NVIDIA Triton deployment. | [Setup and benchmarks](https://github.com/yuekaizhang/Fun-ASR-vllm#readme) and the deterministic ASR decoding fix in merged [#20](https://github.com/yuekaizhang/Fun-ASR-vllm/pull/20). |
| [vad-burn](https://github.com/di-osc/vad-burn) | FSMN VAD inference in pure Rust with Python bindings, including offline, streaming, and CPU-only modes. | [Project README](https://github.com/di-osc/vad-burn#readme) and the FunASR showcase in [#3106](https://github.com/modelscope/FunASR/issues/3106). |

## Before adopting an integration

- Follow the upstream project's installation and release notes; do not assume its dependency versions match FunASR `main`.
- Validate the exact model, language, audio format, and hardware path you plan to deploy.
- Report application or adapter bugs to the integration project. Report reproducible core FunASR model or runtime bugs to [FunASR issues](https://github.com/modelscope/FunASR/issues).

## Add your project

If you maintain a FunASR integration, open a [showcase issue](https://github.com/modelscope/FunASR/issues/new?template=showcase.md) with:

- repository link and maintenance status
- supported FunASR model or runtime path
- install and minimal usage instructions
- a merged change, release, benchmark, or other reproducible validation
- a note about whether the project is official, community-maintained, or experimental
