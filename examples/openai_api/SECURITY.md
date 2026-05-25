# Security and Gateway Guide for the FunASR OpenAI-Compatible API

Use this guide before sharing the example OpenAI-compatible API with a team, workflow engine, browser UI, or service outside your laptop. The example server is intentionally small: it focuses on `/v1/audio/transcriptions` compatibility and does not enforce authentication by itself.

## Recommended topology

```text
OpenAI SDK / Dify / n8n / browser UI
        |
        v
TLS + auth + upload limits + logs
(reverse proxy, API gateway, ingress, or service mesh)
        |
        v
FunASR OpenAI-compatible API
(private host, VM, container, or Kubernetes ClusterIP)
```

Keep FunASR on a private network whenever possible. Put public TLS, identity, request limits, and audit logging at the boundary that your team already operates.

## Minimum controls before sharing

| Control | Why it matters | Where to enforce it |
|---|---|---|
| TLS | Audio often contains private data. | Reverse proxy, API gateway, or ingress. |
| Authentication | The local example accepts any SDK `api_key` placeholder. | Gateway bearer token, basic auth, OAuth/OIDC, or internal SSO. |
| Upload-size limits | Prevent accidental multi-GB uploads and memory pressure. | Gateway request-body limit and app-level checks. |
| Timeouts | Long recordings need longer HTTP timeouts, but stuck clients should not hang forever. | Client, proxy, and server process manager. |
| Rate limits | Protect GPU/CPU capacity from bursts. | Gateway, ingress controller, or queue worker. |
| Private `/health` | Health output is useful operational data, not a public product endpoint. | Network allowlist or private monitoring path. |
| Logs and retention | Request metadata is useful; raw audio may be sensitive. | Central logging policy and storage lifecycle. |

## NGINX reverse proxy sketch

This is a starting point, not a complete production policy. Add your own certificates, identity provider, and secret management.

```nginx
server {
    listen 443 ssl http2;
    server_name funasr.example.com;

    client_max_body_size 200m;
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;

    location / {
        # Add auth_request, basic auth, mTLS, or an API gateway policy here.
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Authorization $http_authorization;
    }

    location = /health {
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://127.0.0.1:8000/health;
    }
}
```

## Caddy reverse proxy sketch

Generate the password hash with `caddy hash-password` and store the real credential outside the repository.

```caddyfile
funasr.example.com {
    request_body {
        max_size 200MB
    }

    basicauth /* {
        team_user <hashed-password>
    }

    reverse_proxy 127.0.0.1:8000 {
        transport http {
            read_timeout 600s
            write_timeout 600s
        }
    }
}
```

For production teams, prefer your standard SSO/OIDC gateway over shared passwords.

## Kubernetes notes

The Kubernetes template keeps the service private with `ClusterIP`. Before adding an ingress or load balancer:

- Add an ingress controller or API gateway that enforces TLS, authentication, upload-size limits, and rate limits.
- Keep model cache volumes private to the namespace or node pool that owns the service.
- Use `NetworkPolicy` to restrict which namespaces can call the service.
- Use `kubectl port-forward` plus `smoke_test.py` for first validation before exposing a route.
- If you add GPUs, pin scheduling rules and record the image tag, CUDA runtime, and model alias in deployment notes.

## Client configuration

OpenAI SDKs usually require an API key string even when FunASR does not check it locally. After you add a gateway, use the gateway-issued token as the SDK key:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://funasr.example.com/v1",
    api_key=os.environ["FUNASR_API_KEY"],
)
```

For internal HTTP workers, read tokens from environment variables or your secret manager. Do not commit tokens to workflow definitions, notebooks, screenshots, or Postman exports.

## Data handling checklist

- Decide whether raw audio can be stored, for how long, and who can access it.
- Log request IDs, duration, model alias, status, latency, and error class; avoid logging raw transcript text unless your policy allows it.
- If transcripts may contain personal data, document retention and deletion rules before onboarding users.
- Keep public samples separate from private customer or employee audio when writing benchmark reports.
- Redact headers, tokens, file names, and speaker names before opening GitHub issues.

## Rollout checklist

1. Start locally and run `bash smoke_test.sh` or `python smoke_test.py`.
2. Add the gateway and verify `/health`, `/v1/models`, and `/v1/audio/transcriptions` through the gateway URL.
3. Test a small file, a large allowed file, and a file above the upload limit.
4. Confirm unauthorized requests fail before reaching FunASR.
5. Confirm timeout behavior for long audio and slow clients.
6. Record the model alias, device, image tag, FunASR version, and gateway policy.

Related guides: [OpenAI API README](README.md), [client recipes](CLIENTS.md), [workflow recipes](WORKFLOWS.md), [Gradio browser demo](GRADIO.md), [Kubernetes template](kubernetes/README.md), and the repository [security policy](../../SECURITY.md).
