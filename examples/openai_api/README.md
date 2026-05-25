(English|[简体中文](README_zh.md))

# FunASR OpenAI-Compatible API Server

Drop-in replacement for OpenAI's `/v1/audio/transcriptions` endpoint. Works with **any agent framework** that supports OpenAI audio API.

## Quick Start

```bash
pip install funasr fastapi uvicorn python-multipart
python server.py --model sensevoice --device cuda --port 8000
```

Server starts in ~20s (model loading). Health check: `GET /health`

Need copy-paste integration snippets for Python SDK, JavaScript/TypeScript, HTTP clients, agent tools, Postman, OpenAPI imports, or Dify/n8n-style workflows? See [Client recipes](CLIENTS.md), [JavaScript/TypeScript recipes](JAVASCRIPT.md), [workflow recipes](WORKFLOWS.md), the [Chinese workflow recipes](WORKFLOWS_zh.md), the [Postman collection](POSTMAN.md), and the [OpenAPI spec](OPENAPI.md).

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
| `fun-asr-nano` | 17x realtime | 3.6x realtime | 31 languages | LLM-based, timestamps |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/transcriptions` | POST | Transcribe audio (OpenAI-compatible) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check + loaded models |
| `/docs` | GET | Interactive API documentation (Swagger) |

Prefer no-code API checks? Import the [Postman collection](POSTMAN.md) and run health, model-list, and transcription requests from Postman. For API gateways, developer portals, or client generation, use the [OpenAPI spec](OPENAPI.md).

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
