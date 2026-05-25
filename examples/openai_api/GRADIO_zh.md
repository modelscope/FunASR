# FunASR OpenAI 兼容 API Gradio 浏览器 Demo

当 FunASR OpenAI 兼容 API 已经在本地、Docker 或私有 Kubernetes 服务中运行，而你想用浏览器上传或录制音频时，可以使用这个 Gradio demo。

Gradio app 本身不加载 FunASR 模型。它调用和 smoke test、SDK 配方、Postman 集合、OpenAPI 规范相同的 `/health`、`/v1/models` 和 `/v1/audio/transcriptions` 接口。

## 1. 启动 API 服务

在 `examples/openai_api` 目录中执行：

```bash
pip install funasr fastapi uvicorn python-multipart
python server.py --model sensevoice --device cuda --port 8000
```

便携 CPU 验证可以使用 `--device cpu`。如果使用 Docker Compose 或 Kubernetes，请保持服务私有，并通过文档中的端口映射或 `kubectl port-forward` 暴露到本机验证。

## 2. 安装并启动浏览器 UI

在另一个终端执行：

```bash
pip install gradio
python gradio_app.py --base-url http://localhost:8000
```

打开命令行输出的本地 URL，上传或录制音频，选择模型 alias，然后点击 **Transcribe**。

## 3. 先验证后端服务

UI 中有 **Check service** 按钮。你也可以在终端运行相同检查：

```bash
python smoke_test.py --base-url http://localhost:8000
curl http://localhost:8000/v1/models
```

如果 API 服务在远端，请显式设置可访问的地址：

```bash
python gradio_app.py --base-url http://funasr-api.speech.svc.cluster.local:8000
```

OpenAI SDK 的 base URL 需要包含 `/v1`；这个 Gradio demo 使用的是不带 `/v1` 的直接服务 base URL。

## 模型别名

| Alias | 适合场景 |
|---|---|
| `sensevoice` | 快速多语种私有转写和 Agent 语音输入。 |
| `paraformer` | 中文生产转写。 |
| `paraformer-en` | 英文兼容性检查。 |
| `fun-asr-nano` | LLM-based ASR 和 vLLM 实验。 |

更完整的对比见 [模型选择指南](../../docs/model_selection_zh.md)。

## 生产注意事项

- Gradio app 适合作为 demo 或内部操作界面，不建议直接作为公网生产前端。
- 任何音频上传 UI 对可信网络外开放前，都需要先增加鉴权、TLS、上传大小限制和限流；见 [安全与网关指南](SECURITY_zh.md)。
- 浏览器上传应尽量靠近你的后端服务，不要把私有音频发送到未鉴权的公网端点。
- 排查问题时记录模型 alias、音频时长、延迟、响应格式和错误文本。
