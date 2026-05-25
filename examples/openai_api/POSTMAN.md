# Postman Collection for the FunASR OpenAI-Compatible API

(English|[简体中文](POSTMAN_zh.md))

Use the Postman collection when you want to verify a private FunASR speech API before wiring it into Dify, n8n, an agent framework, or an internal service. If you prefer schema-driven imports, use the [OpenAPI spec](OPENAPI.md).

## Import

1. Start the server:

   ```bash
   cd examples/openai_api
   python server.py --model sensevoice --device cuda --port 8000
   ```

2. Import [`funasr-openai-api.postman_collection.json`](funasr-openai-api.postman_collection.json) into Postman.
3. Set the collection variable `FUNASR_BASE_URL` to the reachable server URL, for example `http://localhost:8000` or `http://funasr-api:8000`.
4. Keep `MODEL_ALIAS=sensevoice` for the first smoke test, or switch it to one of the aliases returned by `/v1/models`.

## Requests

| Request | Purpose |
|---|---|
| `Health check` | Confirms the server is reachable and returns JSON. |
| `List model aliases` | Shows available OpenAI-compatible model aliases. |
| `Transcribe audio - verbose JSON` | Uploads an audio file and returns `text`, `segments`, and timing metadata. |
| `Transcribe audio - text only` | Minimal transcription request for OpenAI-compatible clients. |

For the transcription requests, open the `Body` tab and choose a local audio file for the `file` form-data field before sending.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ECONNREFUSED` | Confirm the server is running and that `FUNASR_BASE_URL` is reachable from Postman. |
| Docker service works but Postman cannot connect | Use the host port exposed by Docker Compose, for example `http://localhost:8000`. |
| `422` or missing file errors | Make sure the `file` form-data row is enabled and points to a local audio file. |
| Unknown model alias | Run `List model aliases` and copy one of the returned aliases into `MODEL_ALIAS`. |
| No `segments` in the response | Set `RESPONSE_FORMAT=verbose_json`. |
