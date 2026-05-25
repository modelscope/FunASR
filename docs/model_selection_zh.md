# FunASR 模型选择指南

当你第一次选择模型、评估是否从 Whisper 或云端 ASR 迁移，或者准备通过 OpenAI 兼容 API 暴露模型别名时，可以先看这份指南。

## 默认快速路径

如果还不确定，先从 **SenseVoice-Small** 开始：

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

它适合 demo、私有 API、多语种转写、带说话人信息的会议转写和 Agent 语音输入。只有当你的场景明确需要中文生产识别、低延迟流式结果或 LLM-based ASR 实验时，再切换到其他路径。

## 决策表

| 需求 | 优先尝试 | 原因 | 下一步文档 |
|---|---|---|---|
| 快速多语种私有转写 | SenseVoice-Small | 兼顾 ASR、情感标签、音频事件标签和 CPU 可用性。 | [README 快速开始](../README_zh.md#快速开始) |
| 中文生产 ASR | Paraformer-Large | 成熟中文 ASR 路径，可组合 VAD 和标点。 | [教程](./tutorial/README_zh.md) |
| OpenAI API 示例中的英文路由 | `paraformer-en` alias | 适合在 OpenAI 风格客户端里验证较轻量英文路径。 | [OpenAI API 示例](../examples/openai_api/README_zh.md) |
| LLM-based ASR 或 31 语种实验 | Fun-ASR-Nano | LLM-based 模型路径；解码吞吐敏感时配合 vLLM。 | [vLLM 指南](./vllm_guide.md) |
| 实时字幕或客服流式音频 | Runtime WebSocket 服务 | 面向长连接流式会话和中间结果。 | [Runtime 服务文档](../runtime/readme_cn.md) |
| 录音归档批处理 | SenseVoice-Small 或 Paraformer-Large | 稳定离线转写路径；调用方负责 manifest、重试和日志。 | [批处理示例](../examples/batch_asr_improved.py) |
| 从 Whisper/云端 ASR 迁移 | 先用 SenseVoice-Small，再 benchmark 其他模型 | 先建立强基线，再做模型专项调优。 | [迁移指南](./migration_from_whisper_zh.md) |

## OpenAI 兼容 API 别名

`examples/openai_api` 服务提供短别名，应用团队不需要了解具体模型仓库 ID：

| Alias | 底层路径 | 适合场景 |
|---|---|---|
| `sensevoice` | `iic/SenseVoiceSmall` | 默认私有语音 API，多语种 ASR、事件标签和 CPU/GPU 行为较均衡。 |
| `paraformer` | `paraformer-zh` + VAD + 标点 | 中文生产流量优先尝试。 |
| `paraformer-en` | `paraformer-en` + VAD | OpenAI 风格客户端里的英文轻量路由。 |
| `fun-asr-nano` | `FunAudioLLM/Fun-ASR-Nano-2512` | 评估 LLM-based ASR、31 语种覆盖或 vLLM 加速。 |

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
| LLM-based ASR 吞吐 | Fun-ASR-Nano 的 vLLM 路径 | vLLM 加速自回归解码；不适用于非自回归 Paraformer。 |

选择部署方式时可以参考 [部署选型表](./deployment_matrix_zh.md)。

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
- 明确需要 LLM-based 模型路径或 vLLM 加速实验时，再试 Fun-ASR-Nano。
- 需要中间结果和长连接时，优先使用 streaming runtime，而不是普通 HTTP 转写接口。
- 生产 runbook 中固定模型 alias，保证 benchmark 和问题复现可追踪。
- 遇到阻塞时，用 [Deployment Help issue](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md) 提供模型、设备、命令、日志、音频时长和运行路径。
