(English|[简体中文](README_zh.md)|[日本語](README_ja.md)|[한국어](README_ko.md))

# FunASR OpenAI-Compatible API Server

Drop-in replacement for OpenAI's `/v1/audio/transcriptions` endpoint. Works with **any agent framework** that supports OpenAI audio API.

## Quick Start

```bash
pip install funasr fastapi uvicorn python-multipart
python server.py --model sensevoice --device cuda --port 8000
```

Server starts in ~20s (model loading). Health check: `GET /health`

Need copy-paste integration snippets for Python SDK, JavaScript/TypeScript, HTTP clients, agent tools, a browser demo, Postman, OpenAPI imports, Kubernetes deployment, or Dify/n8n-style workflows? See [Client recipes](CLIENTS.md), [JavaScript/TypeScript recipes](JAVASCRIPT.md), [Gradio browser demo](GRADIO.md), [workflow recipes](WORKFLOWS.md), the [Chinese workflow recipes](WORKFLOWS_zh.md), the [Postman collection](POSTMAN.md), the [OpenAPI spec](OPENAPI.md), the [security and gateway guide](SECURITY.md), and the [Kubernetes deployment template](kubernetes/README.md).

### End-to-end smoke test

In another terminal, download a public sample and verify both health and transcription:

```bash
bash smoke_test.sh
# Cross-platform alternative without curl/bash:
python smoke_test.py
```

Equivalent manual commands:

```bash
curl -L https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav
curl http://localhost:8000/health
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

## Browser demo with Gradio

If you want a local browser UI for upload or microphone testing, run the API server first and then launch the optional Gradio frontend:

```bash
pip install gradio
python gradio_app.py --base-url http://localhost:8000
```

The browser demo calls the same OpenAI-compatible API endpoints as the smoke tests. See [Gradio browser demo](GRADIO.md) for Docker, Kubernetes, and production notes.

## Usage with OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Basic transcription
result = client.audio.transcriptions.create(
    model="sensevoice",  # or "paraformer", "paraformer-en", "fun-asr-nano"
    file=open("meeting.wav", "rb"),
)
print(result.text)

# With timestamps/segments
result = client.audio.transcriptions.create(
    model="sensevoice",
    file=open("meeting.wav", "rb"),
    response_format="verbose_json",
)
# Returns: text, segments (with start/end/speaker), duration
```

## Usage with curl

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=sensevoice

# With verbose output
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

## Available Models

| Model | Speed (GPU) | Speed (CPU) | Languages | Features |
|-------|-------------|-------------|-----------|----------|
| `sensevoice` | 170x realtime | 17x realtime | zh/en/ja/ko/yue | Emotion detection |
| `paraformer` | 120x realtime | 15x realtime | zh/en | Punctuation |
| `paraformer-en` | 120x realtime | 15x realtime | en | English only |
| `fun-asr-nano` | 17x realtime | 3.6x realtime | zh/en/ja + Chinese dialects/accents | LLM-based, timestamps |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/transcriptions` | POST | Transcribe audio (OpenAI-compatible) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check + loaded models |
| `/docs` | GET | Interactive API documentation (Swagger) |

Prefer no-code API checks? Use the [Gradio browser demo](GRADIO.md) for local upload or microphone testing, or import the [Postman collection](POSTMAN.md) and run health, model-list, and transcription requests from Postman. For API gateways, developer portals, or client generation, use the [OpenAPI spec](OPENAPI.md).

## Agent Framework Integration

Works with: **LangChain**, **LlamaIndex**, **AutoGen**, **CrewAI**, **Semantic Kernel**, **Dify**, **n8n**, or any framework using OpenAI audio API. See [Client recipes](CLIENTS.md) and [JavaScript/TypeScript recipes](JAVASCRIPT.md) for SDK and agent-tool patterns, plus [workflow recipes](WORKFLOWS.md) for low-code HTTP nodes and webhook workers ([中文](WORKFLOWS_zh.md)).

### LangChain Example
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

def transcribe_for_agent(audio_path: str) -> str:
    """Tool function for LangChain agent."""
    result = client.audio.transcriptions.create(
        model="sensevoice", file=open(audio_path, "rb")
    )
    return result.text
```

## Docker Deployment

Build the example image from this directory. The default image starts in CPU mode so it can be used as a portable smoke test.

```bash
cd examples/openai_api
cp .env.example .env

docker compose up --build
```

Equivalent one-off `docker run` command:

```bash
docker build -t funasr-api .

docker run --rm -p 8000:8000 \
  -e FUNASR_DEVICE=cpu \
  -e FUNASR_MODEL=sensevoice \
  funasr-api
```

For GPU hosts, use NVIDIA Container Toolkit and a CUDA-capable PyTorch/FunASR image. After adapting the image dependencies for CUDA, run the same server with `FUNASR_DEVICE=cuda`:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e FUNASR_DEVICE=cuda \
  -e FUNASR_MODEL=sensevoice \
  funasr-api
```

Verify the container from another terminal:

```bash
BASE_URL=http://localhost:8000 bash smoke_test.sh
python smoke_test.py --base-url http://localhost:8000
```

## Kubernetes Deployment

Before sharing the service across a team or exposing it through a gateway, review the [security and gateway guide](SECURITY.md) for TLS, authentication, upload limits, rate limits, and logging.

For an internal cluster service with persistent model cache, health probes, and a private `ClusterIP`, start from the [Kubernetes deployment template](kubernetes/README.md). Build and push the example image, apply the manifests, then verify through `kubectl port-forward` with `python smoke_test.py --base-url http://localhost:8000`.

Keep the default CPU mode until you have built a CUDA-capable image and configured GPU scheduling for your cluster.

## Configuration

| Arg | Default | Description |
|-----|---------|-------------|
| `--host` | 0.0.0.0 | Bind address |
| `--port` | 8000 | Port |
| `--device` | cuda | Device (cuda/cpu/mps) |
| `--model` | sensevoice | Pre-load model at startup |

Docker environment variables:

| Env | Default | Description |
|-----|---------|-------------|
| `FUNASR_PORT` | 8000 | Container port passed to `server.py` |
| `FUNASR_DEVICE` | cpu | Container device mode; set to `cuda` only when the image has CUDA-capable dependencies |
| `FUNASR_MODEL` | sensevoice | Model alias loaded at container startup |

## Troubleshooting

- If CUDA is unavailable, use `--device cpu` for a slower but simple smoke test.
- If port 8000 is occupied, start with `--port 9000` and run `BASE_URL=http://localhost:9000 bash smoke_test.sh` or `python smoke_test.py --base-url http://localhost:9000`.
- If model download is slow, retry with a stable network or pre-download the model from ModelScope/Hugging Face.
