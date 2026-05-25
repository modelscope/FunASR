([English](README.md)|简体中文)

# FunASR OpenAI 兼容 API 服务

FunASR OpenAI 兼容 API 提供 `/v1/audio/transcriptions`，可作为私有语音转写服务接入 OpenAI 风格 SDK、Agent 框架、Dify、n8n、HTTP 节点和内部业务系统。

## 快速开始

```bash
pip install funasr fastapi uvicorn python-multipart
python server.py --model sensevoice --device cuda --port 8000
```

服务通常会在模型加载后启动。健康检查：`GET /health`。

需要直接复制的接入示例？可以继续查看 [客户端配方](CLIENTS.md)、[JavaScript/TypeScript 配方](JAVASCRIPT_zh.md)、[工作流配方](WORKFLOWS_zh.md)、[Postman 集合](POSTMAN_zh.md)、[OpenAPI 规范](OPENAPI_zh.md) 和 [Kubernetes 部署模板](kubernetes/README_zh.md)。

### 端到端 smoke test

在另一个终端运行：

```bash
bash smoke_test.sh
# 不依赖 curl/bash 的跨平台方式：
python smoke_test.py
```

等价手动命令：

```bash
curl -L https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav
curl http://localhost:8000/health
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

## 使用 OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

result = client.audio.transcriptions.create(
    model="sensevoice",  # 也可以使用 "paraformer"、"paraformer-en"、"fun-asr-nano"
    file=open("meeting.wav", "rb"),
)
print(result.text)

verbose = client.audio.transcriptions.create(
    model="sensevoice",
    file=open("meeting.wav", "rb"),
    response_format="verbose_json",
)
print(verbose.segments)
```

## 使用 curl

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=sensevoice

curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

## 可用模型

| Model | GPU 速度 | CPU 速度 | 语言 | 特性 |
|---|---|---|---|---|
| `sensevoice` | 170x realtime | 17x realtime | zh/en/ja/ko/yue | 情感与事件标签 |
| `paraformer` | 120x realtime | 15x realtime | zh/en | 标点恢复 |
| `paraformer-en` | 120x realtime | 15x realtime | en | 英文识别 |
| `fun-asr-nano` | 17x realtime | 3.6x realtime | 31 languages | LLM-based，时间戳 |

## API 端点

| Endpoint | Method | 说明 |
|---|---|---|
| `/v1/audio/transcriptions` | POST | OpenAI 兼容音频转写 |
| `/v1/models` | GET | 列出模型别名 |
| `/health` | GET | 健康检查、已加载模型和可用模型 |
| `/docs` | GET | FastAPI Swagger 文档 |

不想写代码验证接口时，可以导入 [Postman 集合](POSTMAN_zh.md)。如果要接入 API 网关、开发者门户或生成内部客户端，可以使用 [OpenAPI 规范](OPENAPI_zh.md)。

## Agent 与低代码工作流

适用场景包括 **LangChain**、**LlamaIndex**、**AutoGen**、**CrewAI**、**Semantic Kernel**、**Dify**、**n8n** 和任何支持 OpenAI audio API 或 multipart HTTP 的系统。

- SDK、JavaScript/TypeScript 和 Agent tool 写法见 [客户端配方](CLIENTS.md) 与 [JavaScript/TypeScript 配方](JAVASCRIPT_zh.md)。
- Dify、n8n、HTTP 节点和 webhook worker 见 [工作流配方](WORKFLOWS_zh.md)。
- 图形界面 smoke test 见 [Postman 集合](POSTMAN_zh.md)。
- schema 驱动导入见 [OpenAPI 规范](OPENAPI_zh.md)。

## Docker 部署

默认镜像以 CPU 模式启动，适合作为可复现 smoke test。

```bash
cd examples/openai_api
cp .env.example .env

docker compose up --build
```

等价 `docker run`：

```bash
docker build -t funasr-api .

docker run --rm -p 8000:8000 \
  -e FUNASR_DEVICE=cpu \
  -e FUNASR_MODEL=sensevoice \
  funasr-api
```

GPU 环境需要 NVIDIA Container Toolkit 和 CUDA-capable PyTorch/FunASR 镜像。适配 CUDA 依赖后，可使用：

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e FUNASR_DEVICE=cuda \
  -e FUNASR_MODEL=sensevoice \
  funasr-api
```

验证容器：

```bash
BASE_URL=http://localhost:8000 bash smoke_test.sh
python smoke_test.py --base-url http://localhost:8000
```

## Kubernetes 部署

如果需要在集群内部提供带持久化模型缓存、健康检查和私有 `ClusterIP` 的语音 API，可以从 [Kubernetes 部署模板](kubernetes/README_zh.md) 开始。先构建并推送示例镜像，应用 manifests，再通过 `kubectl port-forward` 和 `python smoke_test.py --base-url http://localhost:8000` 验证。

在没有 CUDA-capable 镜像和 GPU 调度配置前，请保持默认 CPU 模式。

## 配置

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8000` | 监听端口 |
| `--device` | `cuda` | `cuda`、`cpu` 或 `mps` |
| `--model` | `sensevoice` | 启动时预加载模型 |

Docker 环境变量：

| Env | 默认值 | 说明 |
|---|---|---|
| `FUNASR_PORT` | `8000` | 传给 `server.py` 的容器端口 |
| `FUNASR_DEVICE` | `cpu` | 容器设备模式；只有在镜像已适配 CUDA 时才设为 `cuda` |
| `FUNASR_MODEL` | `sensevoice` | 容器启动时加载的模型别名 |

## 故障排查

| 现象 | 处理方式 |
|---|---|
| CUDA 不可用 | 先用 `--device cpu` 跑通 smoke test。 |
| 8000 端口被占用 | 改用 `--port 9000`，并运行 `BASE_URL=http://localhost:9000 bash smoke_test.sh` 或 `python smoke_test.py --base-url http://localhost:9000`。 |
| 模型下载很慢 | 换稳定网络，或提前从 ModelScope/Hugging Face 下载模型。 |
| Dify/n8n 容器里访问 `localhost` 失败 | 使用工作流运行时可访问的主机名、Compose service name 或 Kubernetes service name。 |
| 响应中没有 `segments` | 设置 `response_format=verbose_json`。 |
