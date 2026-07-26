# FunASR Server Trusted Browser CORS Design

Date: 2026-07-27

## Context

Browser clients such as NextChat send OpenAI-compatible multipart transcription requests directly to a local `funasr-server`. The current server returns a valid transcription to command-line clients, but it does not return CORS headers. A browser therefore hides the successful response, and requests with an `Authorization` header fail their preflight with HTTP 405.

## Goals

- Let operators explicitly authorize one or more browser origins.
- Keep the current no-CORS behavior when no option is supplied.
- Support both simple multipart requests and preflighted requests with bearer tokens.
- Keep the server usable through the Python `create_app` API and the `funasr-server` CLI.
- Document a reproducible local-browser configuration in English and Chinese.

## Non-Goals

- Do not enable permissive CORS by default.
- Do not add authentication, cookies, an HTTP proxy, or an origin regular expression.
- Do not change transcription behavior or model loading.
- Do not make the NextChat server proxy requests to a user's local machine.

## Interface

The CLI gains a repeatable option:

```bash
funasr-server \
  --device cpu \
  --model sensevoice \
  --cors-origin http://localhost:3000 \
  --cors-origin http://127.0.0.1:3000
```

`create_app` gains a backward-compatible optional parameter:

```python
def create_app(
    device: str = "cuda",
    preload_model: str = "auto",
    model_path: str | None = None,
    hub: str = "ms",
    cors_origins: list[str] | None = None,
) -> FastAPI:
```

Empty values are ignored, surrounding whitespace is removed, and duplicate origins retain first-seen order. Passing `*` is allowed only as an explicit operator choice.

## Middleware Policy

When the normalized origin list is non-empty, the app adds Starlette's `CORSMiddleware` with:

- `allow_origins`: the normalized exact origins
- `allow_methods`: `GET`, `POST`, and `OPTIONS`
- `allow_headers`: `Authorization` and `Content-Type`
- `allow_credentials`: `False`

When the list is empty or omitted, no middleware is installed. This preserves the existing security boundary and response behavior.

## Data Flow

1. The operator supplies trusted browser origins on the CLI or to `create_app`.
2. The server normalizes and de-duplicates the values.
3. CORS middleware answers matching preflight requests before route dispatch.
4. The existing transcription route processes the multipart audio unchanged.
5. Middleware adds the matching `Access-Control-Allow-Origin` response header.

## Error Handling

An unlisted origin receives no CORS authorization header. The server still behaves normally for non-browser clients. CLI parsing remains responsible for option shape; origin reachability is not checked at startup because a valid browser origin may be offline when the service starts.

## Verification

- Unit tests prove default-disabled behavior and exact middleware configuration.
- CLI tests prove repeated `--cors-origin` values reach `create_app` unchanged.
- Existing server tests prove model and transcription behavior is unchanged.
- A real CPU server must return a successful matching-origin preflight and a real SenseVoice transcription with CORS headers.
- A request from an unlisted origin must not receive `Access-Control-Allow-Origin`.
- English and Chinese docs must include the explicit trusted-origin command and avoid recommending wildcard access.

## Rollback

The design, implementation, and documentation are separate signed commits. The feature branch is preserved remotely before merge, and the default-disabled behavior allows operators to remove the option without changing any other server configuration.
