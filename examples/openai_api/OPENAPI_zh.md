# FunASR OpenAI 兼容 API OpenAPI 规范

[English](OPENAPI.md)

当你希望在接入应用、API 网关、开发者门户、工作流引擎或 SDK 生成器之前，先查看、mock、导入或发布 FunASR 语音 API 时，可以使用 [`openapi.json`](openapi.json)。

运行中的 FastAPI 服务也会在 `/docs` 暴露 Swagger UI，并在 `/openapi.json` 暴露实时 schema。仓库里的 `openapi.json` 是这个示例服务的便携参考规范，便于在服务启动前完成评估和导入。

## 导入方式

| 工具 | 使用方式 |
|---|---|
| Swagger Editor 或 Redoc | 导入 `openapi.json`，查看 `/health`、`/v1/models` 和 `/v1/audio/transcriptions`。 |
| Postman | 如果偏好 schema 驱动集合，可导入 `openapi.json`；如果想直接 smoke test，可使用现成的 [Postman 集合](POSTMAN_zh.md)。 |
| Dify、n8n 或内部工作流工具 | 结合规范中的 multipart 请求结构和 [工作流配方](WORKFLOWS_zh.md) 配置 HTTP 节点。 |
| API 网关或内部开发者门户 | 发布该规范，并把 server URL 改成你的 FunASR API 可访问地址。 |
| 客户端生成 | 生成内部小客户端，并确保 multipart `file` 字段映射为二进制上传。 |

## Server URL

规范内置了两个示例地址：

- `http://localhost:8000`
- `http://funasr-api:8000`

请替换为你的应用、容器或工作流运行环境能访问到的地址。

## 端点

| Endpoint | Method | 用途 |
|---|---|---|
| `/health` | `GET` | 健康检查、设备、已加载模型和可用别名。 |
| `/v1/models` | `GET` | OpenAI 风格模型列表，包含 `ready` 状态。 |
| `/v1/audio/transcriptions` | `POST` | multipart 音频转写；使用 `response_format=verbose_json` 返回 segments。 |

## Multipart 转写字段

| Field | Type | Required | 说明 |
|---|---|---|---|
| `file` | binary | yes | 音频文件，例如 wav、mp3、flac、m4a、ogg 或 webm。 |
| `model` | string | no | 默认 `sensevoice`；可用别名由 `/v1/models` 返回。 |
| `language` | string | no | 可选语言提示。 |
| `response_format` | string | no | 使用 `json` 或 `verbose_json`。 |

## 对照运行中的服务验证

```bash
cd examples/openai_api
python server.py --model sensevoice --device cuda --port 8000
curl -fsS http://localhost:8000/openapi.json > /tmp/funasr-openapi-live.json
```

实时 FastAPI schema 可能包含框架级校验细节；仓库中的静态规范保留更小、更稳定的公开集成面。
