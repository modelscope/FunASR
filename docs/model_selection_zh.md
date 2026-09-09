# FunASR 模型选择指南

当你第一次选择模型、评估是否从 Whisper 或云端 ASR 迁移，或者准备通过 OpenAI 兼容 API 暴露模型别名时，可以先看这份指南。

## 默认快速路径

如果有 GPU，需要评估中文、英文、日语或中文方言和地域口音，可以先试旗舰 **Fun-ASR-Nano**（SenseVoice 编码器 + Qwen3 解码器）。确定生产模型前，请用自己的音频做对比：

```python
from funasr import AutoModel

model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512", device="cuda")
result = model.generate(input="meeting.wav")
print(result[0]["text"])
```

需要非自回归多语种 ASR、情感/事件标签，或先在 CPU 上评估时，可以从 **SenseVoice-Small** 开始。下面的会议转写示例另外组合了 VAD 和说话人处理阶段；说话人分离不是 SenseVoice 单次识别直接提供的能力：

```python
from funasr import AutoModel

model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    spk_model="cam++",
    device="cuda",  # 便携 smoke test 可改为 "cpu"
)
result = model.generate(input="meeting.wav")
```

SenseVoice 生成转写和富文本标签；`fsmn-vad` 定位语音，`cam++` 提取说话人向量，
再由处理流水线聚类得到录音内的匿名编号。这些编号不识别已注册人物，也不是跨录音稳定的身份。

当你的场景是纯中文、需要字级时间戳或热词时，切换到 Paraformer。

## 决策表

| 需求 | 优先尝试 | 原因 | 下一步文档 |
|---|---|---|---|
| 快速多语种私有转写 | SenseVoice-Small | 兼顾 ASR、情感标签、音频事件标签和 CPU 可用性。 | [README 快速开始](../README_zh.md#快速开始) |
| 中文生产 ASR | Paraformer-Large | 成熟中文 ASR 路径，可组合 VAD 和标点。 | [教程](./tutorial/README_zh.md) |
| OpenAI API 示例中的英文路由 | `paraformer-en` alias | 适合在 OpenAI 风格客户端里验证较轻量英文路径。 | [OpenAI API 示例](../examples/openai_api/README_zh.md) |
| LLM-based ASR 或中文/英文/日语 + 方言实验 | Fun-ASR-Nano | 先评估 Python 路径；split-engine 与原生 vLLM 的加载契约不同。 | [选择 vLLM 路径](#vllm-checkpoint-paths) |
| 离线长音频 ASR 与匿名说话人标签 | MOSS-Transcribe-Diarize | 一次离线请求返回转写、时间戳和录音内匿名说话人标签；不识别已知人物，也不需要外部 VAD 或说话人模型。 | [MOSS 部署指南](./moss_transcribe_diarize_zh.md) |
| 实时字幕或客服流式音频 | Runtime WebSocket 服务 | 面向长连接流式会话和中间结果。 | [Runtime 服务文档](../runtime/readme_cn.md) |
| 录音归档批处理 | SenseVoice-Small 或 Paraformer-Large | 稳定离线转写路径；调用方负责 manifest、重试和日志。 | [批处理示例](../examples/batch_asr_improved.py) |
| 从 Whisper/云端 ASR 迁移 | 先用 SenseVoice-Small，再 benchmark 其他模型 | 先建立强基线，再做模型专项调优。 | [迁移指南](./migration_from_whisper_zh.md) |

## OpenAI 兼容 API 别名

`examples/openai_api` 服务提供短别名，应用团队不需要了解具体模型仓库 ID：

- **`sensevoice`** 使用 `iic/SenseVoiceSmall`，用于 CPU/GPU 多语种 HTTP 转写；返回文本已移除富文本标签。
- **`paraformer`** 使用 `paraformer-zh`，组合 VAD 和标点，适合评估中文转写。
- **`paraformer-en`** 使用 `paraformer-en`，组合 VAD，提供 OpenAI 风格客户端的英文转写路径。
- **`fun-asr-nano`** 使用 `FunAudioLLM/Fun-ASR-Nano-2512`，评估中文、英文、日语与中文方言/口音覆盖；测试 vLLM 加速时须选择兼容的运行路径。
- **`moss-transcribe-diarize`** 使用第三方 `OpenMOSS-Team/MOSS-Transcribe-Diarize`，用于离线转写和录音内匿名说话人标签。先按 [MOSS 指南](./moss_transcribe_diarize_zh.md) 准备独立依赖并审查远程代码；需要结构化分段时请求 `verbose_json`。它不需要外部 VAD/说话人模型，也不识别已知人物。

这些别名属于加载 `AutoModel` 的[示例服务](../examples/openai_api/server.py)，
不会配置原生 vLLM，也不会自动选择 `AutoModelVLLM`。
包内 `funasr-server` 有独立的加载与后端选择逻辑；不要跨服务直接套用别名或性能结果，
先核对对应的 [HTTP 指南](../examples/openai_api/README_zh.md)。

示例 HTTP 服务会清理顶层 `text` 和 `verbose_json` 中各分段的 `text`；
切换到该格式不会恢复情感/事件标签。需要原始标签时，请使用 Python SDK，
在展示后处理前保存返回的 `text`，参见[原始标签示例](./speaker_emotion_zh.md)。

如果部署目标是昇腾 NPU，请把 `fun-asr-nano` 和 SenseVoice / Paraformer 分开看。Fun-ASR-Nano 的 PyTorch `AutoModel` 路径在修复 NPU autocast 后已有 310P3 社区兼容性 smoke 结果，但该测试明显慢于 CPU；`AutoModelVLLM` 仍依赖 vLLM-Ascend 算子支持，并已遇到 Qwen3 rotary / `TransData` 失败。生产部署优先使用 CUDA/vLLM、标准 PyTorch CPU/GPU 或 GGUF runtime，除非你正在主动验证 Ascend 后端。

接入客户端前先检查在线服务：

```bash
curl http://localhost:8000/v1/models
python examples/openai_api/smoke_test.py --base-url http://localhost:8000 --model sensevoice
```

SDK、JavaScript、工作流、Postman、OpenAPI、Docker 和 Kubernetes 路径可从 [OpenAI API 示例](../examples/openai_api/README_zh.md) 开始。

## 按工作负载选择运行路径

| 工作负载 | 运行路径 | 说明 |
|---|---|---|
| Notebook 或一次性评估 | Python `AutoModel` | 验证安装、模型下载和输出结构的最短路径。 |
| 内部 HTTP 服务 | OpenAI 兼容 API | 复用 OpenAI 风格客户端、Dify、n8n、LangChain、AutoGen 和 HTTP 节点。 |
| 可复现本地容器 demo | Docker Compose API | CPU-first smoke test；使用 CUDA 前先适配镜像。 |
| 集群内私有服务 | Kubernetes API 模板 | 私有 `ClusterIP`、持久化模型缓存、`/health` probes 和 port-forward smoke test。 |
| 实时音频 | Runtime WebSocket 服务 | 用真实音频验证 chunk size、VAD、断句、重连和客户端背压。 |
| LLM-based ASR 吞吐 | 在下方选择 split-engine 或原生 vLLM | 同时匹配 checkpoint、加载接口和已测环境；这不是 Paraformer 后端。 |

选择部署方式时可以参考 [部署选型表](./deployment_matrix_zh.md)。

<a id="vllm-checkpoint-paths"></a>

## 选择 vLLM 权重与接口

| 路径 | 权重与接口 | 下一步 |
| --- | --- | --- |
| FunASR split-engine | 基础 `FunAudioLLM/Fun-ASR-Nano-2512` 资产，由 `AutoModelVLLM` 加载；FunASR 处理音频部分，vLLM 处理解码器。 | [Split-engine 准备与边界](./vllm_guide_zh.md) |
| 官方原生 vLLM | 转换后的 `FunAudioLLM/Fun-ASR-Nano-2512-vllm` 快照，通过 vLLM 原生模型实现和 `/v1/audio/transcriptions` 提供服务；不是 `AutoModelVLLM` 加载。 | [官方功能验证](./vllm_official_native_validation_zh.md) |
| 历史社区原生 vLLM | 社区 `allendou/Fun-ASR-Nano-2512-vllm`，测试日期 2026-08-13；耗时只属于当时的权重与环境。 | [历史社区记录](./vllm_native_funasr_validation.md) |

官方记录固定了模型 revision 和既有环境，不是全新安装配方、持续负载性能评测，
也不是 `/v1/realtime` 流式验证。不要把社区历史耗时用于官方模型。
MOSS 请遵循其独立部署指南：以上 Nano 权重和验证不能证明 MOSS 的运行时兼容性。
先一起确定模型、权重、接口与环境，再评估自己的工作负载。

## 上线前先 benchmark

不要只用一个干净 demo 文件选型。先准备一个小而有代表性的集合：

- 20-50 条音频，覆盖短音频、长会议、静音、噪声、多人重叠、领域词汇和目标语言。
- 记录模型名、模型版本、FunASR 版本、设备、CPU/GPU 型号、CUDA/PyTorch 版本、运行路径、batch size，以及是否排除 warmup/模型下载时间。
- 使用你已有的 WER/CER 流程或人工审阅，不要只看转写文本是否“读起来还行”。
- 同时记录延迟、吞吐、内存、失败样例和上传大小限制。
- 保留至少一个公开样例用于 smoke test，也保留至少一个真实私有样本用于部署验证。

迁移场景可以使用 [迁移评测示例](../examples/migration/) 和 [迁移指南](./migration_from_whisper_zh.md)。

## 实用建议

- demo、私有 API、Agent 语音输入和多语种场景优先试 SenseVoice-Small。
- 中文生产流量优先试 Paraformer，尤其是希望走成熟非自回归 ASR 路径时。
- 明确需要 LLM-based 模型路径或 vLLM 加速实验时，再试 Fun-ASR-Nano；如需单独的 31 语种覆盖，请改用 Fun-ASR-MLT-Nano。
- 离线长录音需要同一次请求给出录音内匿名说话人标签时，使用 MOSS-Transcribe-Diarize；它不是实时 WebSocket 或已知人物身份识别路径。
- 需要中间结果和长连接时，优先使用 streaming runtime，而不是普通 HTTP 转写接口。
- 生产 runbook 中固定模型 alias，保证 benchmark 和问题复现可追踪。
- 遇到阻塞时，用 [Deployment Help issue](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md) 提供模型、设备、命令、日志、音频时长和运行路径。
