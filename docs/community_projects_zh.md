[English](./community_projects.md) | [简体中文](./community_projects_zh.md)

# FunASR 社区集成

本页收录已经在上游项目中落地、仍在维护的 FunASR、Fun-ASR-Nano、SenseVoice 或其他官方 FunASR 模型集成。下面每一项都已对照项目默认分支核验；如有对应的已合并 PR，文中也提供了链接。

这些集成由社区项目维护，其发布节奏、硬件支持和 API 稳定性由上游项目决定，不属于 FunASR 维护者的兼容性承诺。

## 语音 Agent 与应用

| 项目 | 已集成能力 | 从这里开始 |
|---|---|---|
| [Pipecat](https://github.com/pipecat-ai/pipecat) | 面向语音和多模态 Agent pipeline 的本地 `FunASRSTTService`，底层使用 SenseVoice，并提供转写与语音 Agent 示例。 | [FunASR 服务文档](https://docs.pipecat.ai/api-reference/server/services/stt/funasr)、[转写示例](https://github.com/pipecat-ai/pipecat/blob/main/examples/transcription/transcription-funasr.py)、[语音 Agent 示例](https://github.com/pipecat-ai/pipecat/blob/main/examples/voice/voice-funasr.py) 和已合并 [#4844](https://github.com/pipecat-ai/pipecat/pull/4844)。 |
| [xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server) | 默认本地 ASR provider 使用 FunASR + SenseVoiceSmall；识别语种可设为 `auto`、`zh`、`en`、`ja`、`ko` 或 `yue`，同时也提供独立 FunASR runtime server provider。 | [Provider 实现](https://github.com/xinnan-tech/xiaozhi-esp32-server/blob/main/main/xiaozhi-server/core/providers/asr/fun_local.py)、[配置](https://github.com/xinnan-tech/xiaozhi-esp32-server/blob/main/main/xiaozhi-server/config.yaml) 和已合并 [#3255](https://github.com/xinnan-tech/xiaozhi-esp32-server/pull/3255)。 |
| [AutoSubs](https://github.com/tmoroney/auto-subs) | 面向 DaVinci Resolve、Premiere 和 After Effects 字幕生成的端侧 int8 ONNX SenseVoice 引擎。 | [SenseVoice 模型说明](https://github.com/tmoroney/auto-subs#sensevoice)、[引擎源码](https://github.com/tmoroney/auto-subs/blob/main/AutoSubs-App/src-tauri/crates/transcription-engine/src/engines/sense_voice.rs) 和已合并 [#629](https://github.com/tmoroney/auto-subs/pull/629)。 |
| [TMSpeech](https://github.com/jxlpzqc/TMSpeech) | Windows 会议字幕与翻译工具，可通过外部命令识别器运行 Fun-ASR-Nano INT8。Silero VAD 先完成语音段切分再调用 Nano 解码，同时保留原有低成本 C# 流式识别器。 | [Fun-ASR-Nano 配置](https://github.com/jxlpzqc/TMSpeech/blob/master/README.md#%E4%BD%BF%E7%94%A8-fun-asr-nano)、[识别器源码](https://github.com/jxlpzqc/TMSpeech/blob/master/external_recognizer/simulate-streaming-funasr-nano.py) 和已合并 [#110](https://github.com/jxlpzqc/TMSpeech/pull/110)。 |
| [OmniVoice Studio](https://github.com/debpalash/OmniVoice-Studio) | 通过 OpenAI 兼容远程 ASR backend，把听写和配音工作流连接到自托管 FunASR 或 SenseVoice 服务。 | [OpenAI 兼容 ASR 指南](https://github.com/debpalash/OmniVoice-Studio/blob/main/docs/engines/openai-compatible-asr.md) 和已合并 [#1003](https://github.com/debpalash/OmniVoice-Studio/pull/1003)。 |
| [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | 使用 Fun-ASR-Nano、SenseVoice 和经典 FunASR 模型完成数据集预处理与 WebUI 转写。 | [`funasr_asr.py`](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/tools/asr/funasr_asr.py)、runtime fallback [#2801](https://github.com/RVC-Boss/GPT-SoVITS/pull/2801) 和 backend 文档 [#2803](https://github.com/RVC-Boss/GPT-SoVITS/pull/2803)。 |
| [AudioNotes](https://github.com/harry0703/AudioNotes) | 将音视频内容提取为结构化 Markdown 笔记，并把 Fun-ASR-MLT-Nano 路由到 Fun-ASR Nano 推理配置，保留 cache、batch size 和列表式 hotwords。 | [项目 README](https://github.com/harry0703/AudioNotes#readme)、[FunASR 服务](https://github.com/harry0703/AudioNotes/blob/main/app/services/asr_funasr.py) 和已合并 [#65](https://github.com/harry0703/AudioNotes/pull/65)。 |
| [wyoming-faster-whisper](https://github.com/OHF-Voice/wyoming-faster-whisper) | Wyoming 协议 speech-to-text server，提供可选 FunASR backend，让本地语音助手和 Home Assistant 相关部署可以通过 Wyoming 使用 SenseVoice/FunASR。 | [FunASR handler](https://github.com/OHF-Voice/wyoming-faster-whisper/blob/master/wyoming_faster_whisper/funasr_handler.py)、[模型注册](https://github.com/OHF-Voice/wyoming-faster-whisper/blob/master/wyoming_faster_whisper/models.py) 和已合并 [#95](https://github.com/OHF-Voice/wyoming-faster-whisper/pull/95)。 |
| [Dify official plugins](https://github.com/langgenius/dify-official-plugins) | 面向自托管 FunASR endpoint 的 Dify 官方 speech-to-text 模型 provider，内置 SenseVoice、Fun-ASR-Nano、Paraformer 和 Paraformer English 配置。 | [FunASR 插件 README](https://github.com/langgenius/dify-official-plugins/blob/main/models/funasr/README.md)、[provider 实现](https://github.com/langgenius/dify-official-plugins/blob/main/models/funasr/provider/funasr.py)、[speech-to-text adapter](https://github.com/langgenius/dify-official-plugins/blob/main/models/funasr/models/speech2text/speech2text.py) 和已合并 [#3281](https://github.com/langgenius/dify-official-plugins/pull/3281)。 |
| [RAGFlow](https://github.com/infiniflow/ragflow) | RAG 与 Agent 平台，提供本地 FunASR / SenseVoice speech-to-text provider，适合自托管文档和媒体摄取工作流。 | [Provider 注册](https://github.com/infiniflow/ragflow/blob/main/rag/llm/sequence2txt_model.py)、[支持模型文档](https://github.com/infiniflow/ragflow/blob/main/docs/guides/models/supported_models.mdx) 和已合并 [#16473](https://github.com/infiniflow/ragflow/pull/16473)。 |
| [LiveTalking](https://github.com/lipku/LiveTalking) | 实时交互数字人服务，包含本地 FunASR/SenseVoice ASR server 路径；已合并修复会串行化共享模型访问，避免并发请求在懒加载模型或 `generate()` 时竞争。 | [ASR server](https://github.com/lipku/LiveTalking/blob/main/server/asr_server.py)、[项目 README](https://github.com/lipku/LiveTalking#readme) 和已合并 [#611](https://github.com/lipku/LiveTalking/pull/611)。 |

## 模型服务与 Runtime

| 项目 | 已集成能力 | 从这里开始 |
|---|---|---|
| [Xinference](https://github.com/xorbitsai/inference) | 通过 Xinference 统一推理 API 提供 SenseVoiceSmall、Fun-ASR-Nano-2512 和 Fun-ASR-MLT-Nano-2512 内置音频模型规格。 | [音频模型规格](https://github.com/xorbitsai/inference/blob/main/xinference/model/audio/model_spec.json) 和已合并 [#5140](https://github.com/xorbitsai/inference/pull/5140) 中的 FunASR 1.3 兼容性更新。 |
| [Fun-ASR-vLLM](https://github.com/yuekaizhang/Fun-ASR-vllm) | 面向 Fun-ASR-Nano 和 Fun-ASR-MLT-Nano 的社区 vLLM 推理实现，包含批量评测和 NVIDIA Triton 部署。 | [安装与 Benchmark](https://github.com/yuekaizhang/Fun-ASR-vllm#readme) 和已合并 [#20](https://github.com/yuekaizhang/Fun-ASR-vllm/pull/20) 中的确定性 ASR 解码修复。 |
| [vad-burn](https://github.com/di-osc/vad-burn) | 纯 Rust FSMN VAD 推理与 Python binding，支持离线、流式和纯 CPU 模式。 | [项目 README](https://github.com/di-osc/vad-burn#readme) 和 FunASR showcase [#3106](https://github.com/modelscope/FunASR/issues/3106)。 |

## 采用社区集成前

- 按上游项目的安装和发布说明操作，不要假设其依赖版本与 FunASR `main` 完全一致。
- 用计划部署的模型、语种、音频格式和硬件路径做一次真实验证。
- 应用或 adapter 问题优先反馈给集成项目；可复现的 FunASR 核心模型或 runtime 问题反馈到 [FunASR issues](https://github.com/modelscope/FunASR/issues)。

## 收录你的项目

如果你维护了 FunASR 集成，请提交 [showcase issue](https://github.com/modelscope/FunASR/issues/new?template=showcase.md)，并提供：

- 仓库链接和维护状态
- 支持的 FunASR 模型或 runtime 路径
- 安装方式和最小使用示例
- 已合并改动、正式发布、benchmark 或其他可复现验证
- 官方、社区维护或实验性项目的明确说明
