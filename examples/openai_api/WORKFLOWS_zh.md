# FunASR OpenAI 兼容 API 低代码工作流配方

[English](WORKFLOWS.md)

当你希望让 Dify、n8n、HTTP 节点、webhook worker 或其他低代码工作流引擎调用私有 FunASR 语音 API 时，可以从这份配方开始。建议先在本目录跑通本地 smoke test，再把 `localhost` 换成工作流运行环境能够访问到的服务名或内网地址。

## 服务预检

```bash
cd examples/openai_api
python server.py --model sensevoice --device cuda --port 8000
```

在工作流所在主机或容器中执行：

```bash
export FUNASR_BASE_URL=http://<funasr-host>:8000
curl -fsS "$FUNASR_BASE_URL/health"
curl -fsS "$FUNASR_BASE_URL/v1/models"
```

如果工作流引擎运行在 Docker 中，`localhost` 通常指的是工作流容器自身。请改用 Docker Compose service name、Kubernetes service name 或内网主机名。

## Postman smoke test

在配置低代码工具前，可以先导入 [Postman collection](POSTMAN_zh.md)，从图形界面跑通 health、模型列表和转写请求；需要按 schema 导入时可使用 [OpenAPI spec](OPENAPI_zh.md)。设置 `FUNASR_BASE_URL`，在 multipart `file` 字段选择本地音频文件，第一次测试建议保持 `MODEL_ALIAS=sensevoice`。

## Multipart HTTP 请求

所有工作流引擎最终都需要发出下面这种请求：

| 字段 | 值 |
|---|---|
| Method | `POST` |
| URL | `http://<funasr-host>:8000/v1/audio/transcriptions` |
| Body type | `multipart/form-data` |
| File field | `file` |
| Text field | `model=sensevoice` |
| Text field | `response_format=verbose_json` |
| Timeout | 根据最长音频时长设置，例如长录音可先设为 300 秒。 |

等价 curl 命令：

```bash
curl "$FUNASR_BASE_URL/v1/audio/transcriptions" \
  -F file=@meeting.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

常用响应字段映射：

| 路径 | 用途 |
|---|---|
| `text` | 纯文本转写结果，可传给聊天机器人、工单、知识库或后续摘要节点。 |
| `segments` | 请求 `verbose_json` 时返回的时间戳、说话人等分段信息。 |
| `duration` | API 返回的音频处理时长，适合写入日志和监控。 |
| `model` | 本次请求使用的模型别名。 |

## Dify 自定义工具或 HTTP 节点

当 Dify 应用接收上传音频文件，或收到内部音频存储 URL 时，可以使用下面两种模式。

### 直接上传文件

在 HTTP request 节点或自定义工具中配置：

- Method: `POST`
- URL: `http://<funasr-host>:8000/v1/audio/transcriptions`
- Body: `multipart/form-data`
- File part: `file`，绑定到上传音频变量
- Text parts: `model=sensevoice`、`response_format=verbose_json`
- Output variable: 把 `text` 映射为转写文本；需要时间戳或说话人信息时保留 `segments`

### 音频 URL 转写

有些工作流工具只能传文件 URL，而不能直接传 multipart 二进制。此时建议加一个内网 worker：

1. Dify 将音频 URL 和元数据发送给 worker。
2. worker 从可信存储下载音频。
3. worker 使用 multipart 请求调用 FunASR。
4. worker 将 `text`、`segments` 和运行日志返回给 Dify。

```python
import requests

FUNASR_URL = "http://funasr-api:8000/v1/audio/transcriptions"

def transcribe_from_url(audio_url: str) -> dict:
    audio_response = requests.get(audio_url, timeout=120)
    audio_response.raise_for_status()
    files = {"file": ("audio.wav", audio_response.content, "audio/wav")}
    data = {"model": "sensevoice", "response_format": "verbose_json"}
    response = requests.post(FUNASR_URL, files=files, data=data, timeout=300)
    response.raise_for_status()
    return response.json()
```

请把这个 worker 放在可信内网中，并在下载用户提供的链接前校验允许访问的域名或存储桶。

## n8n HTTP Request 节点

一个常见 n8n 流程是：触发器 -> 二进制音频数据 -> HTTP Request -> 转写结果消费节点。

推荐 HTTP Request 配置：

| n8n 设置 | 值 |
|---|---|
| Method | `POST` |
| URL | `http://<funasr-host>:8000/v1/audio/transcriptions` |
| Send Body | enabled |
| Body Content Type | `Form-Data` / multipart |
| Binary file field | `file` |
| Additional form fields | `model=sensevoice`、`response_format=verbose_json` |
| Response Format | JSON |
| Timeout | 长录音场景需要调大。 |

请求之后，使用 `{{$json.text}}` 作为转写文本。如果启用了 `verbose_json`，可以把 `{{$json.segments}}` 传给字幕、说话人分析或质检节点。

## Webhook worker 模式

当工作流引擎不能稳定发送 multipart 文件，或音频需要预处理时，可以把转写封装成一个内部 webhook worker。

```python
from pathlib import Path
import tempfile
import requests

FUNASR_URL = "http://localhost:8000/v1/audio/transcriptions"

def transcribe_bytes(filename: str, payload: bytes, content_type: str = "audio/wav") -> dict:
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".wav") as tmp:
        tmp.write(payload)
        tmp.flush()
        with open(tmp.name, "rb") as audio:
            response = requests.post(
                FUNASR_URL,
                files={"file": (filename, audio, content_type)},
                data={"model": "sensevoice", "response_format": "verbose_json"},
                timeout=300,
            )
    response.raise_for_status()
    return response.json()
```

这个 worker 也适合集中做音频转码、文件大小限制、请求 ID、鉴权、重试和审计日志。

## 生产环境护栏

- 在跨团队共享 FunASR 服务前，先加好鉴权、TLS、上传大小限制和限流；代理和网关模式见 [安全与网关指南](SECURITY_zh.md)。
- 使用 `/health` 做工作流 readiness check，使用 `/v1/models` 校验模型别名。
- 记录 request id、音频时长、模型别名、响应格式、设备、延迟和错误类型。
- 按最长音频时长设置工作流超时；超长录音建议先切分，再交给低代码工具处理。
- 私有音频放在可信存储中，避免把签名 URL、凭据或转写文本写入公开日志。
- 上生产前，至少用一个公开 smoke 样例和一个真实业务样例完整跑通同一条工作流。

## 故障排查

| 现象 | 处理方式 |
|---|---|
| 工作流能访问 `/health`，但转写失败 | 确认请求是 `multipart/form-data`，且二进制字段名是 `file`。 |
| Dify 或 n8n 访问 `localhost` 失败 | 换成工作流运行时可访问的主机名、Compose service name 或 Kubernetes service name。 |
| 响应中没有 `segments` | 设置 `response_format=verbose_json`。 |
| 请求超时 | 调大 HTTP timeout，或先切分长录音。 |
| 第一次请求很慢 | 使用 `--model sensevoice` 预加载模型，并用 `/health` 做 readiness check。 |
| 模型别名未知 | 调用 `/v1/models`，使用返回列表中的别名。 |
