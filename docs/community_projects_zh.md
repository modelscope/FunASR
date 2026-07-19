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
| [clowder-ai](https://github.com/zts212653/clowder-ai) | 本地 AI 服务启动器；`whisper-stt` 服务位现在包含 Qwen3-ASR 作为 backend model variant，让本地 ASR 安装入口保持可见，同时不再新增一个独立语音服务。 | [服务 manifest](https://github.com/zts212653/clowder-ai/blob/main/packages/cli/src/services/service-manifest.ts)、[推荐矩阵](https://github.com/zts212653/clowder-ai/blob/main/packages/cli/src/services/recommendation-matrix.yaml) 和已合并 [#1083](https://github.com/zts212653/clowder-ai/pull/1083)。 |
| [蛐蛐 / QuQu](https://github.com/yan5xu/ququ) | 中文桌面语音转文字工作流，也是 Wispr Flow 的开源替代方案；内置本地 FunASR Paraformer pipeline、`funasr_server.py`、VAD、标点恢复，并可接入 LLM 文本优化。 | [项目 README](https://github.com/yan5xu/ququ#readme)、[FunASR server](https://github.com/yan5xu/ququ/blob/main/funasr_server.py) 和 [package metadata](https://github.com/yan5xu/ququ/blob/main/package.json)。 |
| [OpenLess](https://github.com/Open-Less/openless) | macOS 与 Windows 开源语音输入应用；统一 Bailian ASR provider 暴露 Fun-ASR-Flash 录音文件转写，并与实时 ASR 选项一起服务于光标听写和 AI prompt 工作流。 | [项目 README](https://github.com/Open-Less/openless#readme)、[DashScope multimodal ASR provider](https://github.com/Open-Less/openless/blob/beta/openless-all/app/src-tauri/src/asr/dashscope_multimodal.rs)、[provider settings copy](https://github.com/Open-Less/openless/blob/beta/openless-all/app/src/i18n/zh-CN.ts) 和已合并 [#793](https://github.com/Open-Less/openless/pull/793)。 |
| [OmniVoice Studio](https://github.com/debpalash/OmniVoice-Studio) | 本地语音克隆、配音、听写和有声书应用；既可通过 OpenAI 兼容远程 ASR backend 连接自托管 FunASR/SenseVoice 服务，也提供原生 FunASR/SenseVoice + CAM++ 路径，让整段录音里的 speaker identity 保持一致。 | [OpenAI 兼容 ASR 指南](https://github.com/debpalash/OmniVoice-Studio/blob/main/docs/engines/openai-compatible-asr.md)、[FunASR backend](https://github.com/debpalash/OmniVoice-Studio/blob/main/backend/services/asr_backend.py)、已合并 [#1003](https://github.com/debpalash/OmniVoice-Studio/pull/1003) 和已合并 [#1167](https://github.com/debpalash/OmniVoice-Studio/pull/1167)。 |
| [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | 使用 Fun-ASR-Nano、SenseVoice 和经典 FunASR 模型完成数据集预处理与 WebUI 转写。 | [`funasr_asr.py`](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/tools/asr/funasr_asr.py)、runtime fallback [#2801](https://github.com/RVC-Boss/GPT-SoVITS/pull/2801) 和 backend 文档 [#2803](https://github.com/RVC-Boss/GPT-SoVITS/pull/2803)。 |
| [AudioNotes](https://github.com/harry0703/AudioNotes) | 将音视频内容提取为结构化 Markdown 笔记，并把 Fun-ASR-MLT-Nano 路由到 Fun-ASR Nano 推理配置，保留 cache、batch size 和列表式 hotwords。 | [项目 README](https://github.com/harry0703/AudioNotes#readme)、[FunASR 服务](https://github.com/harry0703/AudioNotes/blob/main/app/services/asr_funasr.py) 和已合并 [#65](https://github.com/harry0703/AudioNotes/pull/65)。 |
| [NarratoAI](https://github.com/linyqh/NarratoAI) | AI 视频解说与剪辑应用，可通过 OpenAI 兼容转写 endpoint，把字幕生成连接到自托管 FunASR 服务。 | [FunASR 字幕服务](https://github.com/linyqh/NarratoAI/blob/main/app/services/fun_asr_subtitle.py)、[配置示例](https://github.com/linyqh/NarratoAI/blob/main/config.example.toml) 和已合并 [#259](https://github.com/linyqh/NarratoAI/pull/259)。 |
| [wyoming-faster-whisper](https://github.com/OHF-Voice/wyoming-faster-whisper) | Wyoming 协议 speech-to-text server，提供可选 FunASR backend，让本地语音助手和 Home Assistant 相关部署可以通过 Wyoming 使用 SenseVoice/FunASR。 | [FunASR handler](https://github.com/OHF-Voice/wyoming-faster-whisper/blob/master/wyoming_faster_whisper/funasr_handler.py)、[模型注册](https://github.com/OHF-Voice/wyoming-faster-whisper/blob/master/wyoming_faster_whisper/models.py) 和已合并 [#95](https://github.com/OHF-Voice/wyoming-faster-whisper/pull/95)。 |
| [Dify official plugins](https://github.com/langgenius/dify-official-plugins) | 面向自托管 FunASR endpoint 的 Dify 官方 speech-to-text 模型 provider，内置 SenseVoice、Fun-ASR-Nano、Paraformer 和 Paraformer English 配置。 | [FunASR 插件 README](https://github.com/langgenius/dify-official-plugins/blob/main/models/funasr/README.md)、[provider 实现](https://github.com/langgenius/dify-official-plugins/blob/main/models/funasr/provider/funasr.py)、[speech-to-text adapter](https://github.com/langgenius/dify-official-plugins/blob/main/models/funasr/models/speech2text/speech2text.py) 和已合并 [#3281](https://github.com/langgenius/dify-official-plugins/pull/3281)。 |
| [RAGFlow](https://github.com/infiniflow/ragflow) | RAG 与 Agent 平台，提供本地 FunASR / SenseVoice speech-to-text provider，适合自托管文档和媒体摄取工作流；同时可通过 Tongyi-Qianwen provider 使用托管 Fun-ASR-Flash speech-to-text。 | [Provider 注册](https://github.com/infiniflow/ragflow/blob/main/rag/llm/sequence2txt_model.py)、[支持模型文档](https://github.com/infiniflow/ragflow/blob/main/docs/guides/models/supported_models.mdx)、已合并 [#16473](https://github.com/infiniflow/ragflow/pull/16473) 和已合并 [#16844](https://github.com/infiniflow/ragflow/pull/16844)。 |
| [LiveTalking](https://github.com/lipku/LiveTalking) | 实时交互数字人服务，包含本地 FunASR/SenseVoice ASR server 路径；已合并修复会串行化共享模型访问，避免并发请求在懒加载模型或 `generate()` 时竞争。 | [ASR server](https://github.com/lipku/LiveTalking/blob/main/server/asr_server.py)、[项目 README](https://github.com/lipku/LiveTalking#readme) 和已合并 [#611](https://github.com/lipku/LiveTalking/pull/611)。 |
| [Sokuji](https://github.com/kizuna-ai-lab/sokuji) | 实时语音翻译应用，本地推理 sidecar 支持通过 FunASR 使用 GPU-native SenseVoice，并在内置模型目录中提供离线 CPU/GPU Fun-ASR-Nano 与 Fun-ASR-MLT-Nano ASR。 | [Sidecar 模型目录](https://github.com/kizuna-ai-lab/sokuji/blob/main/sidecar/sokuji_sidecar/catalog.py)、[native model catalog](https://github.com/kizuna-ai-lab/sokuji/blob/main/src/lib/local-inference/native/nativeCatalog.ts)、已合并 [#268](https://github.com/kizuna-ai-lab/sokuji/pull/268)、已合并 [#270](https://github.com/kizuna-ai-lab/sokuji/pull/270) 和已合并 [#329](https://github.com/kizuna-ai-lab/sokuji/pull/329)。 |
| [VocaLinux](https://github.com/jatinkrmalik/vocalinux) | Linux 离线语音输入应用，可通过 remote ASR engine 连接自托管 FunASR 或 SenseVoice OpenAI 兼容 endpoint。 | [Remote ASR 指南](https://github.com/jatinkrmalik/vocalinux/blob/main/docs/HTTP_REMOTE.md)、[recognition manager](https://github.com/jatinkrmalik/vocalinux/blob/main/src/vocalinux/speech_recognition/recognition_manager.py) 和已合并 [#468](https://github.com/jatinkrmalik/vocalinux/pull/468)。 |
| [TranscriptionSuite](https://github.com/homelab-00/TranscriptionSuite) | 本地私有 speech-to-text 桌面/server 应用，提供 SenseVoice/FunASR backend、OpenAI 兼容音频路由和 CAM++ 说话人分离支持。 | [SenseVoice backend](https://github.com/homelab-00/TranscriptionSuite/blob/main/server/backend/core/stt/backends/sensevoice_backend.py)、[OpenAI audio route](https://github.com/homelab-00/TranscriptionSuite/blob/main/server/backend/api/routes/openai_audio.py)、已合并 [#198](https://github.com/homelab-00/TranscriptionSuite/pull/198) 和已合并 [#201](https://github.com/homelab-00/TranscriptionSuite/pull/201)。 |

## 模型服务与 Runtime

| 项目 | 已集成能力 | 从这里开始 |
|---|---|---|
| [Xinference](https://github.com/xorbitsai/inference) | 通过 Xinference 统一推理 API 提供 SenseVoiceSmall、Fun-ASR-Nano-2512 和 Fun-ASR-MLT-Nano-2512 内置音频模型规格。 | [音频模型规格](https://github.com/xorbitsai/inference/blob/main/xinference/model/audio/model_spec.json) 和已合并 [#5140](https://github.com/xorbitsai/inference/pull/5140) 中的 FunASR 1.3 兼容性更新。 |
| [Optimum Intel](https://github.com/huggingface/optimum-intel) | 通过 Hugging Face Optimum Intel 支持 Fun-ASR 模型的 OpenVINO 导出与推理，包括 FunASR 专用 modeling 代码以及 ASR、export、quantization 覆盖。 | [FunASR OpenVINO modeling](https://github.com/huggingface/optimum-intel/blob/main/optimum/intel/openvino/modeling_funasr.py)、[支持模型文档](https://github.com/huggingface/optimum-intel/blob/main/docs/source/openvino/models.mdx) 和已合并 [#1801](https://github.com/huggingface/optimum-intel/pull/1801)。 |
| [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks) | OpenVINO 官方 FunASR-Nano 教程资料，包含 helper 代码和 NPU device 修复，方便用户在 Intel 加速路径上运行 notebook。 | [FunASR-Nano notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/funasr-nano)、[OpenVINO FunASR helper](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/funasr-nano/ov_funasr_helper.py)、已合并 [#3497](https://github.com/openvinotoolkit/openvino_notebooks/pull/3497) 和 NPU 修复 [#3517](https://github.com/openvinotoolkit/openvino_notebooks/pull/3517)。 |
| [SGLang Omni](https://github.com/sgl-project/sglang-omni) | 面向 omni models 的 multi-stage pipeline runtime，已内置 Fun-ASR 模型支持、OpenAI 兼容服务、cookbook 文档和 benchmark 任务。 | [Fun-ASR cookbook](https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/fun_asr.md)、[Fun-ASR 模型 runtime](https://github.com/sgl-project/sglang-omni/tree/main/sglang_omni/models/fun_asr) 和已合并 [#1078](https://github.com/sgl-project/sglang-omni/pull/1078)。 |
| [Fun-ASR-vLLM](https://github.com/yuekaizhang/Fun-ASR-vllm) | 面向 Fun-ASR-Nano 和 Fun-ASR-MLT-Nano 的社区 vLLM 推理实现，包含批量评测和 NVIDIA Triton 部署。 | [安装与 Benchmark](https://github.com/yuekaizhang/Fun-ASR-vllm#readme) 和已合并 [#20](https://github.com/yuekaizhang/Fun-ASR-vllm/pull/20) 中的确定性 ASR 解码修复。 |
| [vad-burn](https://github.com/di-osc/vad-burn) | 纯 Rust FSMN VAD 推理与 Python binding，支持离线、流式和纯 CPU 模式。 | [项目 README](https://github.com/di-osc/vad-burn#readme) 和 FunASR showcase [#3106](https://github.com/modelscope/FunASR/issues/3106)。 |

## 发现入口

| 项目 | 已集成能力 | 从这里开始 |
|---|---|---|
| [awesome-python](https://github.com/vinta/awesome-python) | SenseVoice 已收录到大型 Python 资源目录的 Speech Recognition 部分，方便 Python 开发者发现本地多语种 ASR。 | [项目 README](https://github.com/vinta/awesome-python#readme) 和已合并 [#3246](https://github.com/vinta/awesome-python/pull/3246)。 |
| [speech-trident](https://github.com/ga642381/speech-trident) | SenseVoice 已收录到 speech/audio language models 目录，面向语音 LLM、representation learning 和 codec model 读者。 | [项目 README](https://github.com/ga642381/speech-trident#readme) 和已合并 [#31](https://github.com/ga642381/speech-trident/pull/31)。 |
| [voiceai](https://github.com/mahimairaja/voiceai) | 面向 Voice AI agent builder 的资源地图，在 Speech-to-text (STT / ASR) 部分同时收录 FunASR 和 SenseVoice，帮助开发者发现自托管本地 ASR 与多语种语音理解选项。 | [英文 README](https://github.com/mahimairaja/voiceai#readme)、[中文 README](https://github.com/mahimairaja/voiceai/blob/main/README_zh.md) 和已合并 [#16](https://github.com/mahimairaja/voiceai/pull/16)。 |
| [Large-Audio-Models](https://github.com/liusongxiang/Large-Audio-Models) | 面向音频领域 foundation model 的资源目录，已收录 FunAudioLLM 语音理解与生成论文及 SenseVoice 代码入口，方便语音、歌声和音乐模型读者发现项目。 | [项目 README](https://github.com/liusongxiang/Large-Audio-Models#readme) 和已合并 [#26](https://github.com/liusongxiang/Large-Audio-Models/pull/26)。 |
| [Neural-Codec-and-Speech-Language-Models](https://github.com/LqNoob/Neural-Codec-and-Speech-Language-Models) | 面向 neural codec、TTS 和 speech language model 的资源目录，已同时收录 Fun-ASR-Nano 与 SenseVoice，方便对比 ASR 和语音理解模型的读者发现项目。 | [项目 README](https://github.com/LqNoob/Neural-Codec-and-Speech-Language-Models#readme) 和已合并 [#4](https://github.com/LqNoob/Neural-Codec-and-Speech-Language-Models/pull/4)。 |

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
