# FunASR 部署选型表

这个页面帮助你为产品、demo、benchmark 或内部工作流选择最短部署路径。先选择能满足目标的最小方案，只有在吞吐、延迟或集成方式有明确要求时，再切换到更重的运行时。

## 快速决策表

| 路径 | 适合场景 | 从这里开始 | 运维提示 |
|---|---|---|---|
| Python API | Notebook、离线任务、首次模型评测 | [README 快速开始](../README_zh.md#快速开始) | 最简单；调用方自己负责批处理、重试和文件管理。 |
| OpenAI 兼容 API | 私有语音 API、Agent、Dify/LangChain/AutoGen 风格客户端 | [OpenAI API 示例](../examples/openai_api/README_zh.md) | 已支持 OpenAI audio API 的应用最容易接入。 |
| Docker Compose API | 可复现本地 smoke test 或小型内部服务 | [OpenAI API Docker 文档](../examples/openai_api/README_zh.md) | 默认 CPU；容器里使用 CUDA 前需要先适配 CUDA-capable 镜像。 |
| Runtime WebSocket 服务 | 实时字幕、会议、客服流式音频 | [Runtime 服务文档](../runtime/readme_cn.md) | 需要中间结果、断句或长连接音频流时选择。 |
| vLLM 加速 | Fun-ASR-Nano 等 LLM-based ASR 高吞吐 | [vLLM 指南](./vllm_guide.md) | 适合 LLM 解码吞吐；不适用于非自回归 Paraformer。 |
| MCP 服务 | Claude/Cursor/桌面 Agent 语音工具 | [MCP 示例](../examples/mcp_server/) | 适合把 ASR 结果暴露成一个本地工具。 |
| 字幕生成 | 从长音频或视频生成 SRT/VTT | [字幕示例](../examples/subtitle/) | 需要可读性时使用 verbose segments 和说话人标签。 |
| 批处理脚本 | 录音归档、会议纪要、数据集处理 | [批处理示例](../examples/batch_asr_improved.py) | 生产使用时建议增加队列、manifest 和重试日志。 |
| Triton Runtime | 专门的高性能推理服务 | [Triton 文档](../runtime/triton_gpu/README.md) | 配置更重；适合已经在运维 Triton/GPU serving 的团队。 |

## 常见选择

### 我想五分钟内试跑 FunASR

使用 README 里的 Python API。它是验证安装、模型下载、设备选择和基础输出格式的最短路径。

### 我想替代云端转写服务

使用 OpenAI 兼容 API。它提供 `/v1/audio/transcriptions`、`/v1/models`、`/health` 和 Swagger docs。先用 `sensevoice` 跑通 `examples/openai_api/smoke_test.sh` 或 `examples/openai_api/smoke_test.py`，再根据 [客户端配方](../examples/openai_api/CLIENTS.md) 接入 SDK 或 HTTP 客户端。Dify、n8n、HTTP 节点或 webhook worker 可参考 [工作流配方](../examples/openai_api/WORKFLOWS_zh.md)。API 网关、开发者门户或按 schema 导入时可使用 [OpenAPI 规范](../examples/openai_api/OPENAPI_zh.md)。

### 我想要可复现的容器 demo

使用 `examples/openai_api/docker-compose.yml` 跑 CPU smoke test：

```bash
cd examples/openai_api
cp .env.example .env
docker compose up --build
```

在没有 CUDA-capable PyTorch/FunASR 镜像前保持 CPU 模式。准备好 CUDA 镜像后，再设置 `FUNASR_DEVICE=cuda` 并用同一个 smoke test 验证。没有 bash/curl 时可运行 `python examples/openai_api/smoke_test.py --base-url http://localhost:8000`。

### 我需要流式识别或实时字幕

使用 Runtime WebSocket 服务。上线前请用真实音频验证 chunk size、VAD、断句、标点、说话人分离、重连行为和客户端背压。

### 我需要更高的 LLM-based ASR 吞吐

Fun-ASR-Nano 走 vLLM 路径。请用自己的音频分布做 benchmark，并关注 GPU 显存、tensor parallel size、首 token 延迟和 warmup 时间。

## 上线检查清单

- 选择模型 alias，并写入部署说明。
- 记录 FunASR 版本、模型版本、设备、CUDA/PyTorch 版本、Docker 镜像 tag 和启动命令。
- 跑一个公开短音频 smoke sample，再跑至少一个真实私有样本。
- 每次请求记录音频时长、模型、设备、延迟、响应格式和错误类型。
- API 暴露到可信网络外之前，增加上传大小限制、鉴权、TLS 和限流。
- 流式场景需要测试静音、噪声、多人重叠、长连接、重连和慢客户端。
- 发布 benchmark 结论时，说明输入时长、硬件、batch size、模型、运行路径，以及是否排除模型下载和 warmup 时间。

## 什么时候开 issue

Runtime、Docker、vLLM、Triton、Android、浏览器或 Agent 集成问题，请使用 [Deployment Help](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md)。请附上部署路径、完整命令/config、日志、模型、设备和音频特征。
