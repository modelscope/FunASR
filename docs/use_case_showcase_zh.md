# FunASR 场景速览

FunASR 不只是一个离线转写命令。这个页面把常见的评测、部署、Agent 集成和生产场景整理到一起，方便新用户快速找到入口。

## 选择合适路径

| 目标 | 从这里开始 | 为什么重要 |
|---|---|---|
| 本地转写一个文件 | [README 快速开始](../README_zh.md#快速开始) 和 [模型选择指南](./model_selection_zh.md) | 几分钟内验证安装、模型选择、模型下载和首次推理。 |
| 对比准确率和速度 | [性能评测报告](https://modelscope.github.io/FunASR/zh/benchmark.html) | 选型前先查看 184 条长音频评测结果。 |
| 从 Whisper/云端 ASR 迁移 | [迁移指南](./migration_from_whisper_zh.md) | 将现有流水线映射到 FunASR，用代表性音频评测并规划安全上线。 |
| 搭建私有语音 API | [OpenAI 兼容 API 示例](../examples/openai_api/README_zh.md)、[Gradio 浏览器 Demo](../examples/openai_api/GRADIO_zh.md)、[客户端配方](../examples/openai_api/CLIENTS.md)、[JavaScript/TypeScript 配方](../examples/openai_api/JAVASCRIPT_zh.md) 和 [工作流配方](../examples/openai_api/WORKFLOWS_zh.md) | 复用 LangChain、Dify、n8n、AutoGen 等 OpenAI 风格客户端，音频不出内网。 |
| 给 Agent 增加语音输入 | [MCP 服务](../examples/mcp_server/) 和 [语音输入示例](../examples/voice_input/) | 将本地 ASR 接入 Claude、Cursor 和桌面 Agent 工作流。 |
| 选择部署路径 | [部署选型表](./deployment_matrix_zh.md) | 对比 Python API、OpenAI API、Docker Compose、Kubernetes、WebSocket、vLLM、MCP、批处理、字幕和 Triton。 |
| 部署流式 ASR | [Runtime 服务文档](../runtime/readme_cn.md) | 面向实时字幕、客服、会议等低延迟场景。 |
| 加速 LLM-based ASR | [vLLM 指南](./vllm_guide.md) | 为 Fun-ASR-Nano 使用 tensor parallel 解码和流式服务能力。 |
| 生成字幕 | [字幕示例](../examples/subtitle/) | 将长音频或视频转成字幕文件。 |
| 批量处理录音 | [批处理示例](../examples/batch_asr_improved.py) | 为录音归档、会议纪要、数据集处理搭建可重复流水线。 |

## 面向生产的配方

### 私有转写 API

当应用已经兼容 OpenAI 风格接口，或音频不能离开私有环境时，优先使用这个路径。

```bash
pip install funasr fastapi uvicorn python-multipart
funasr-server --model sensevoice --device cuda
```

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

建议下一步：

- 运行 [OpenAI 兼容 API smoke test](../examples/openai_api/smoke_test.sh) 或跨平台 [Python smoke test](../examples/openai_api/smoke_test.py)。
- 浏览器上传或麦克风 demo 可从 [Gradio 浏览器 Demo](../examples/openai_api/GRADIO_zh.md) 开始。
- Node.js 或 Next.js 服务可从 [JavaScript/TypeScript 配方](../examples/openai_api/JAVASCRIPT_zh.md) 开始。
- 集群内服务可从 [Kubernetes 部署模板](../examples/openai_api/kubernetes/README_zh.md) 开始。
- 在服务边界增加鉴权、限流和网络访问控制。
- 记录模型、设备、驱动、音频时长和处理耗时，便于复现问题和 benchmark。

### Agent 语音输入

当你想把语音输入接到编码助手、内部助手或工作流工具时，使用这个路径。

- Claude/Cursor 类工具优先看 [MCP 服务示例](../examples/mcp_server/)。
- 桌面语音输入实验可以从 [voice input 示例](../examples/voice_input/) 开始。
- 保持延迟可见：每次请求记录音频时长、处理耗时和模型名称。

### 流式与客服场景

当你更关注低延迟和中间结果，而不是单次完整转写时，使用这个路径。

- 从 [Runtime 服务文档](../runtime/readme_cn.md) 开始。
- 需要给人阅读时，把 ASR 与 VAD、标点恢复、说话人分离一起使用。
- 用真实音频验证：背景噪声、长静音、多人重叠、不同麦克风质量。

### 从 Whisper 迁移前先评测

当你在评估是否用 FunASR 替代 Whisper 或云端 ASR 时，使用这个路径。

- 按 [迁移指南](./migration_from_whisper_zh.md) 映射功能并评测代表性音频。
- 阅读 [公开性能评测](https://modelscope.github.io/FunASR/zh/benchmark.html)。
- 用自己的样本集再测一次；同时包含短音频和长音频。
- 同时记录成本和吞吐：GPU 速度、CPU 可用性、模型下载体积、部署复杂度。

## 模型选择建议

如需更完整地比较 SenseVoice、Paraformer、Fun-ASR-Nano、streaming runtime 和 OpenAI API alias，请看 [模型选择指南](./model_selection_zh.md)。

| 需求 | 推荐先试 | 说明 |
|---|---|---|
| 快速多语种转写 | SenseVoice-Small | 本地 demo 和私有 API 的稳妥默认选择。 |
| 中文生产 ASR | Paraformer-Large | 中文语音识别的成熟选择。 |
| LLM-based ASR 实验 | Fun-ASR-Nano | 吞吐敏感时配合 [vLLM 指南](./vllm_guide.md)。 |
| 带说话人信息的转写 | SenseVoice 或 Paraformer + `spk_model="cam++"` | 适合会议、访谈、客服录音。 |
| 实时音频 | Runtime WebSocket 服务 | 用真实流量验证分块、VAD 和断句。 |

## 分享你的结果

如果 FunASR 在你的项目里效果不错，欢迎通过 [showcase issue](https://github.com/modelscope/FunASR/issues/new?template=showcase.md)、[Migration Benchmark Report](https://github.com/modelscope/FunASR/issues/new?template=migration_benchmark.md) 或 GitHub Discussion 分享：

- 使用场景和部署方式。
- 模型、设备和处理速度。
- 音频领域、语言和大致时长。
- 可以公开的 demo、截图、benchmark 摘要或集成链接。

具体的使用反馈能帮助新用户更快选型，也能帮助维护者决定下一批文档和示例优先级。
