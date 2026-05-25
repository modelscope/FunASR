# Gradio Browser Demo for the FunASR OpenAI-Compatible API

Use this demo when you want a browser UI for uploading or recording audio while the FunASR OpenAI-compatible API server runs locally, in Docker, or behind a private Kubernetes service.

The Gradio app does not load FunASR models itself. It calls the same `/health`, `/v1/models`, and `/v1/audio/transcriptions` endpoints used by the smoke tests, SDK recipes, Postman collection, and OpenAPI spec.

## 1. Start the API server

From `examples/openai_api`:

```bash
pip install funasr fastapi uvicorn python-multipart
python server.py --model sensevoice --device cuda --port 8000
```

For a portable CPU check, use `--device cpu`. For Docker Compose or Kubernetes, keep the service private and expose it locally with the documented port mapping or `kubectl port-forward`.

## 2. Install and launch the browser UI

In another terminal:

```bash
pip install gradio
python gradio_app.py --base-url http://localhost:8000
```

Open the printed local URL, upload or record an audio file, choose a model alias, and click **Transcribe**.

## 3. Verify the backend first

The UI has a **Check service** button. You can run the same check in a terminal:

```bash
python smoke_test.py --base-url http://localhost:8000
curl http://localhost:8000/v1/models
```

If the API server is remote, set the reachable URL explicitly:

```bash
python gradio_app.py --base-url http://funasr-api.speech.svc.cluster.local:8000
```

For OpenAI SDK clients, remember that SDK base URLs include `/v1`; this Gradio demo expects the direct service base URL without `/v1`.

## Model aliases

| Alias | Good first use |
|---|---|
| `sensevoice` | Fast multilingual private transcription and agent voice input. |
| `paraformer` | Mandarin-oriented production transcription. |
| `paraformer-en` | English-only compatibility checks. |
| `fun-asr-nano` | LLM-based ASR and vLLM experiments. |

See the [model selection guide](../../docs/model_selection.md) for a deeper comparison.

## Production notes

- Treat the Gradio app as a demo or internal operator UI, not a public production frontend.
- Add authentication, TLS, upload-size limits, and rate limits before exposing any audio upload UI outside a trusted network.
- Keep browser uploads close to your backend; do not send private audio to an unauthenticated public endpoint.
- Log model alias, audio duration, latency, response format, and error text when debugging.
