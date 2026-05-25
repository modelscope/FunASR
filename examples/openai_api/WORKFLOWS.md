# Low-code Workflow Recipes for the FunASR OpenAI-Compatible API

[中文](WORKFLOWS_zh.md)

Use this guide when you want Dify, n8n, webhook workers, or another workflow engine to call a private FunASR speech API. Start with the local smoke test in this directory, then replace `localhost` with the reachable service name inside your network.

## Server preflight

```bash
cd examples/openai_api
python server.py --model sensevoice --device cuda --port 8000
```

From the workflow host or container:

```bash
export FUNASR_BASE_URL=http://<funasr-host>:8000
curl -fsS "$FUNASR_BASE_URL/health"
curl -fsS "$FUNASR_BASE_URL/v1/models"
```

If the workflow engine runs in Docker, `localhost` usually means the workflow container itself. Use a Docker Compose service name, Kubernetes service name, or LAN host name instead.

## Postman smoke test

Before configuring a low-code tool, you can import the [Postman collection](POSTMAN.md) and run health, model-list, and transcription requests from a GUI. For schema-driven imports, use the [OpenAPI spec](OPENAPI.md). Set `FUNASR_BASE_URL`, choose a local audio file for the multipart `file` field, and keep `MODEL_ALIAS=sensevoice` for the first test.

## Multipart HTTP request

Every workflow engine eventually needs to send this request shape:

| Field | Value |
|---|---|
| Method | `POST` |
| URL | `http://<funasr-host>:8000/v1/audio/transcriptions` |
| Body type | `multipart/form-data` |
| File field | `file` |
| Text field | `model=sensevoice` |
| Text field | `response_format=verbose_json` |
| Timeout | Set according to maximum audio duration, for example 300 seconds for long files. |

Equivalent curl command:

```bash
curl "$FUNASR_BASE_URL/v1/audio/transcriptions" \
  -F file=@meeting.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

Typical JSON fields to map downstream:

| Path | Use |
|---|---|
| `text` | Plain transcript for a chatbot, ticket, or knowledge-base step. |
| `segments` | Timestamps and speaker labels when `verbose_json` is requested. |
| `duration` | Audio processing time reported by the API, useful for logs. |
| `model` | Model alias used for the request. |

## Dify custom tool or HTTP node

Use this pattern when a Dify application receives an uploaded audio file or a URL to internal audio storage.

### Direct file upload path

Configure an HTTP request node or custom tool with:

- Method: `POST`
- URL: `http://<funasr-host>:8000/v1/audio/transcriptions`
- Body: `multipart/form-data`
- File part: `file`, bound to the uploaded audio variable
- Text parts: `model=sensevoice`, `response_format=verbose_json`
- Output variable: map `text` as the transcript, and keep `segments` when timestamps or speaker labels matter

### Audio URL path

Some workflow tools pass a file URL rather than raw multipart bytes. In that case, add a small internal worker:

1. Dify sends the audio URL and metadata to the worker.
2. The worker downloads the file from trusted storage.
3. The worker posts multipart data to FunASR.
4. The worker returns `text`, `segments`, and operational logs to Dify.

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

Keep this worker inside your trusted network and validate allowed URL domains before downloading user-provided links.

## n8n HTTP Request node

A common n8n flow is: trigger -> binary audio data -> HTTP Request -> transcript consumer.

Recommended HTTP Request settings:

| n8n setting | Value |
|---|---|
| Method | `POST` |
| URL | `http://<funasr-host>:8000/v1/audio/transcriptions` |
| Send Body | enabled |
| Body Content Type | `Form-Data` / multipart |
| Binary file field | `file` |
| Additional form fields | `model=sensevoice`, `response_format=verbose_json` |
| Response Format | JSON |
| Timeout | Increase for long recordings. |

After the request, use `{{$json.text}}` as the transcript. If `verbose_json` is enabled, route `{{$json.segments}}` to subtitle, speaker, or QA steps.

## Webhook worker pattern

Use this when the workflow engine cannot send multipart files reliably or when audio needs pre-processing.

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

This worker is also the right place to add audio conversion, file-size checks, request IDs, authentication, and retries.

## Production guardrails

- Put authentication, TLS, upload-size limits, and rate limits in front of FunASR before sharing it across teams; use the [security and gateway guide](SECURITY.md) for proxy and gateway patterns.
- Use `/health` for workflow readiness checks and `/v1/models` to validate model aliases.
- Log request id, audio duration, model alias, response format, device, latency, and error type.
- Set workflow timeouts according to maximum audio duration; split very long recordings before sending them through low-code tools.
- Keep private audio in trusted storage and avoid putting signed URLs, credentials, or transcripts into public logs.
- Run the same workflow with at least one public smoke sample and one realistic private sample before production use.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Workflow can call `/health` but transcription fails | Confirm the request is `multipart/form-data` and the binary field is named `file`. |
| `localhost` connection fails from Dify or n8n | Use the host, Compose service, or Kubernetes service reachable from the workflow runtime. |
| Response has no `segments` | Set `response_format=verbose_json`. |
| Requests time out | Increase HTTP timeout or split long recordings. |
| First request is slow | Preload the model with `--model sensevoice` and use `/health` as a readiness check. |
| Unknown model alias | Call `/v1/models` and use one of the returned aliases. |
