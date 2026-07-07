# FunASR 部署选型表

这个页面帮助你为产品、demo、benchmark 或内部工作流选择最短部署路径。先选择能满足目标的最小方案，只有在吞吐、延迟或集成方式有明确要求时，再切换到更重的运行时。

## 快速决策表

| 路径 | 适合场景 | 从这里开始 | 运维提示 |
|---|---|---|---|
| Colab Notebook | 浏览器 smoke test、首次评估、可分享 demo | [Colab 快速体验](../examples/colab/README_zh.md) | 不需要本地环境；首次运行会下载模型，GPU runtime 更快。 |
| Python API | Notebook、离线任务、首次模型评测 | [README 快速开始](../README_zh.md#快速开始) | 最简单；调用方自己负责批处理、重试和文件管理。 |
| OpenAI 兼容 API | 私有语音 API、Agent、Dify/LangChain/AutoGen 风格客户端 | [OpenAI API 示例](../examples/openai_api/README_zh.md) | 已支持 OpenAI audio API 的应用最容易接入。 |
| Xinference | 已经使用 Xinference 统一管理模型服务的团队 | [Xinference 仓库](https://github.com/xorbitsai/inference) | 使用包含 [xorbitsai/inference#5140](https://github.com/xorbitsai/inference/pull/5140) 的版本或 commit，确保 Fun-ASR-Nano 使用打包发布的 `funasr~=1.3.0`，而不是旧的 git commit pin。 |
| Docker Compose API | 可复现本地 smoke test 或小型内部服务 | [OpenAI API Docker 文档](../examples/openai_api/README_zh.md) | 默认 CPU；容器里使用 CUDA 前需要先适配 CUDA-capable 镜像。 |
| Kubernetes API | 集群内私有语音 API | [Kubernetes 模板](../examples/openai_api/kubernetes/README_zh.md) | 默认私有 `ClusterIP`；对外开放前补齐鉴权、TLS、网络策略和 GPU 调度。 |
| Runtime WebSocket 服务 | 实时字幕、会议、客服流式音频 | [Runtime 服务文档](../runtime/readme_cn.md) | 需要中间结果、断句或长连接音频流时选择。 |
| vLLM 加速 | Fun-ASR-Nano 等 LLM-based ASR 高吞吐 | [vLLM 指南](./vllm_guide.md) | 适合 LLM 解码吞吐；不适用于非自回归 Paraformer。 |
| MCP 服务 | Claude/Cursor/桌面 Agent 语音工具 | [MCP 示例](../examples/mcp_server/) | 适合把 ASR 结果暴露成一个本地工具。 |
| 字幕生成 | 从长音频或视频生成 SRT/VTT | [字幕示例](../examples/subtitle/) | 需要可读性时使用 verbose segments 和说话人标签。 |
| 批处理脚本 | 录音归档、会议纪要、数据集处理 | [批处理示例](../examples/batch_asr_improved.py) | 生产使用时建议增加队列、manifest 和重试日志。 |
| Triton Runtime | 专门的高性能推理服务 | [Triton 文档](../runtime/triton_gpu/README.md) | 配置更重；适合已经在运维 Triton/GPU serving 的团队。 |

## 常见选择

### 我想五分钟内试跑 FunASR

如果只想在浏览器里 smoke test，可以先用 [Colab 快速体验](../examples/colab/README_zh.md)；本地工作再使用 README 里的 Python API。它是验证安装、模型下载、设备选择和基础输出格式的最短路径。如果还不确定先用哪个模型，请看 [模型选择指南](./model_selection_zh.md)。

### 我想替代云端转写服务

使用 OpenAI 兼容 API。它提供 `/v1/audio/transcriptions`、`/v1/models`、`/health` 和 Swagger docs。先用 `sensevoice` 跑通 `examples/openai_api/smoke_test.sh` 或 `examples/openai_api/smoke_test.py`，再根据 [客户端配方](../examples/openai_api/CLIENTS.md) 和 [JavaScript/TypeScript 配方](../examples/openai_api/JAVASCRIPT_zh.md) 接入 SDK 或 HTTP 客户端。浏览器上传或麦克风 demo 可使用 [Gradio 浏览器 Demo](../examples/openai_api/GRADIO_zh.md)。Dify、n8n、HTTP 节点或 webhook worker 可参考 [工作流配方](../examples/openai_api/WORKFLOWS_zh.md)。API 网关、开发者门户或按 schema 导入时可使用 [OpenAPI 规范](../examples/openai_api/OPENAPI_zh.md)。跨团队共享服务前，请先阅读 [安全与网关指南](../examples/openai_api/SECURITY_zh.md)。

### 我已经在使用 Xinference

如果你的系统已经用 Xinference 管理模型注册、virtualenv 隔离和服务生命周期，可以选择 Xinference 路径。请确认使用的 Xinference 版本或 commit 包含 [xorbitsai/inference#5140](https://github.com/xorbitsai/inference/pull/5140)；该更新把 Fun-ASR-Nano model spec 从旧的 FunASR git SHA 改为打包发布的 `funasr~=1.3.0` 依赖。首次评估 FunASR，或需要面向 Agent 的 OpenAI 兼容转写接口时，仍建议先从上面的 FunASR 原生 OpenAI API 示例开始。

### 我想要可复现的容器 demo

使用 `examples/openai_api/docker-compose.yml` 跑 CPU smoke test：

```bash
cd examples/openai_api
cp .env.example .env
docker compose up --build
```

在没有 CUDA-capable PyTorch/FunASR 镜像前保持 CPU 模式。准备好 CUDA 镜像后，再设置 `FUNASR_DEVICE=cuda` 并用同一个 smoke test 验证。没有 bash/curl 时可运行 `python examples/openai_api/smoke_test.py --base-url http://localhost:8000`。

### 我想部署集群内服务

使用 [Kubernetes 模板](../examples/openai_api/kubernetes/README_zh.md) 部署私有 `ClusterIP` OpenAI 兼容 API，包含持久化模型缓存、`/health` probes 和 port-forward smoke test 路径。在没有 CUDA-capable 镜像和集群 GPU 调度前，请保持默认 CPU 模式。

### 我需要流式识别或实时字幕

使用 Runtime WebSocket 服务。上线前请用真实音频验证 chunk size、VAD、断句、标点、说话人分离、重连行为和客户端背压。

### 我需要更高的 LLM-based ASR 吞吐

Fun-ASR-Nano 走 vLLM 路径。请用自己的音频分布做 benchmark，并关注 GPU 显存、tensor parallel size、首 token 延迟和 warmup 时间。

### 我想在昇腾 NPU 上跑 Fun-ASR-Nano

Fun-ASR-Nano 的 LLM-based 路径目前主要按 CUDA/vLLM、标准 PyTorch CPU/GPU，以及 CPU/边缘 GGUF runtime 记录和验证；Ascend NPU（`torch_npu`）还不是这个模型的官方验证运行时。不要因为 SenseVoice 或 Paraformer 能在 NPU 上跑，就默认 Fun-ASR-Nano 也能直接跑通，因为 Nano 还会经过 Qwen 解码器、`inputs_embeds` 和 autocast 路径。若要适配，请先从 `torch.bfloat16` 开始，记录 `torch` / `torch_npu` / CANN 版本，并在最小 PR 或 deployment issue 里附上最小命令和完整错误栈。

## 上线检查清单

- 选择模型 alias，并写入部署说明。
- 记录 FunASR 版本、模型版本、设备、CUDA/PyTorch 版本、Docker 镜像 tag 和启动命令。
- 跑一个公开短音频 smoke sample，再跑至少一个真实私有样本。
- 每次请求记录音频时长、模型、设备、延迟、响应格式和错误类型。
- API 暴露到可信网络外之前，增加上传大小限制、鉴权、TLS 和限流；可用 [安全与网关指南](../examples/openai_api/SECURITY_zh.md) 规划边界。
- 流式场景需要测试静音、噪声、多人重叠、长连接、重连和慢客户端。
- 发布 benchmark 结论时，说明输入时长、硬件、batch size、模型、运行路径，以及是否排除模型下载和 warmup 时间。

## 什么时候开 issue

Runtime、Docker、vLLM、Triton、Android、浏览器或 Agent 集成问题，请使用 [Deployment Help](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md)。请附上部署路径、完整命令/config、日志、模型、设备和音频特征。
