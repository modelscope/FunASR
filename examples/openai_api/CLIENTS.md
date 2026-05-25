# Client Recipes for the FunASR OpenAI-Compatible API

Use this page when `funasr-server` is already running and you want to connect an existing application, agent tool, or workflow engine to local speech recognition. For Dify, n8n, HTTP nodes, and webhook workers, see the [workflow recipes](WORKFLOWS.md) or [Chinese workflow recipes](WORKFLOWS_zh.md). For no-code API smoke tests, import the [Postman collection](POSTMAN.md).

## Preflight

```bash
export BASE_URL=http://localhost:8000
curl -fsS "$BASE_URL/health"
curl -fsS "$BASE_URL/v1/models"
```

If the server is on another machine, replace `localhost` with the reachable host name or service address. Keep `/v1` in SDK base URLs, and omit `/v1` for direct endpoint checks like `/health`.

## Model aliases

| Alias | Good first use | Notes |
|---|---|---|
| `sensevoice` | Private multilingual API | Fast default with language, emotion, and event tags. |
| `paraformer` | Mandarin production transcription | Includes VAD and punctuation. |
| `paraformer-en` | English transcription | Smaller English-only route. |
| `fun-asr-nano` | LLM-based ASR experiments | Pair with vLLM for higher throughput deployments. |

## Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

with open("meeting.wav", "rb") as audio:
    result = client.audio.transcriptions.create(
        model="sensevoice",
        file=audio,
        response_format="verbose_json",
    )

print(result.text)
for segment in getattr(result, "segments", []):
    print(segment)
```

Most OpenAI SDKs require an API key value even when the local FunASR server does not check it. Use any placeholder for local development, then add real authentication at your gateway if the service is shared.

## Plain Python requests

```python
import requests

with open("meeting.wav", "rb") as audio:
    response = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        files={"file": ("meeting.wav", audio, "audio/wav")},
        data={"model": "sensevoice", "response_format": "verbose_json"},
        timeout=300,
    )
response.raise_for_status()
print(response.json()["text"])
```

This is the most portable pattern for internal services, queues, notebooks, and low-code tools that can issue multipart HTTP requests.

## Agent tool pattern

Expose transcription as a regular tool function. The agent does not need to know FunASR internals; it only needs a file path or uploaded audio object.

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

def transcribe_audio(audio_path: str) -> str:
    """Transcribe a local audio file with FunASR and return plain text."""
    path = Path(audio_path)
    with path.open("rb") as audio:
        result = client.audio.transcriptions.create(
            model="sensevoice",
            file=audio,
        )
    return result.text
```

For LangChain, LlamaIndex, AutoGen, CrewAI, Semantic Kernel, and similar frameworks, register the function above using that framework's normal tool or function-calling mechanism.

## Dify, workflow engines, and HTTP nodes

Use a multipart HTTP node or custom tool:

| Setting | Value |
|---|---|
| Method | `POST` |
| URL | `http://<funasr-host>:8000/v1/audio/transcriptions` |
| Body type | `multipart/form-data` |
| File field | `file` |
| Text fields | `model=sensevoice`, `response_format=verbose_json` |
| Result path | `text` for transcript, `segments` for timestamps/speakers |

When the workflow system cannot send files directly, upload audio to an internal object store first, then run a small worker that downloads the object and calls FunASR with the `requests` recipe above. See [workflow recipes](WORKFLOWS.md) for Dify, n8n, and webhook-worker patterns, or the [Chinese workflow recipes](WORKFLOWS_zh.md).

## Response formats

`response_format=json` returns a compact response:

```json
{"text": "recognized speech"}
```

`response_format=verbose_json` adds operational fields useful for agents and subtitles:

```json
{
  "text": "recognized speech",
  "segments": [
    {"start": 0.0, "end": 3.2, "text": "recognized speech", "speaker": 0}
  ],
  "language": "auto",
  "duration": 0.42,
  "model": "sensevoice"
}
```

## Production checklist

- Put TLS, authentication, rate limits, and upload-size limits in front of the service before exposing it outside a trusted network.
- Preload the default model at startup and use `/health` for readiness checks.
- Set client timeouts based on maximum audio duration; long recordings need longer HTTP timeouts.
- Log audio duration, model alias, device, latency, response format, and error type for every request.
- Pin model aliases and deployment images in production notes so benchmark results remain reproducible.
- For GPU hosts, keep one worker per GPU until you have measured memory headroom and concurrency behavior.

## Troubleshooting quick checks

| Symptom | Check |
|---|---|
| SDK says authentication is missing | Pass any placeholder `api_key` for local development. |
| 400 unknown model | Call `/v1/models` and use one of the listed aliases. |
| Request times out | Increase client timeout or split very long recordings. |
| First request is slow | The model may be loading; preload with `--model sensevoice`. |
| CUDA is unavailable | Start with `--device cpu` to verify the API path, then fix GPU drivers/runtime. |
| Port conflict | Start with `--port 9000` and set `BASE_URL=http://localhost:9000`. |
