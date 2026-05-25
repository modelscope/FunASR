# Kubernetes Deployment for the FunASR OpenAI-Compatible API

This folder provides a CPU-first Kubernetes template for the `examples/openai_api` server. Use it when you want an internal OpenAI-compatible speech endpoint reachable by agents, web backends, workflow engines, or batch workers inside a cluster.

The manifest is intentionally conservative:

- `ClusterIP` service by default, not a public `LoadBalancer`.
- `FUNASR_DEVICE=cpu` by default so the image matches the portable Dockerfile.
- A persistent cache volume mounted at `/root/.cache` so model downloads survive pod restarts.
- `/health` startup, readiness, and liveness probes.
- A memory-backed `/dev/shm` volume for PyTorch and audio preprocessing.

## 1. Build and push the image

From `examples/openai_api`:

```bash
docker build -t registry.example.com/speech/funasr-api:cpu-latest .
docker push registry.example.com/speech/funasr-api:cpu-latest
```

Edit `kustomization.yaml` if you use a different registry, repository, or tag.

## 2. Deploy

```bash
kubectl create namespace speech --dry-run=client -o yaml | kubectl apply -f -
kubectl -n speech apply -k .
kubectl -n speech rollout status deploy/funasr-api --timeout=15m
```

Model download and first load can take several minutes. The `startupProbe` allows up to 10 minutes before Kubernetes restarts the container.

## 3. Smoke test

Keep the service private and verify it through `port-forward` first:

```bash
kubectl -n speech port-forward svc/funasr-api 8000:8000
```

From another terminal in `examples/openai_api`:

```bash
python smoke_test.py --base-url http://localhost:8000
```

For in-cluster clients, use `http://funasr-api.speech.svc.cluster.local:8000` as the direct HTTP base URL and `http://funasr-api.speech.svc.cluster.local:8000/v1` as the OpenAI SDK base URL.

## 4. Tune for your cluster

| Setting | Default | When to change it |
|---|---|---|
| `FUNASR_MODEL` | `sensevoice` | Use another alias after checking `/v1/models`. |
| `FUNASR_DEVICE` | `cpu` | Set to `cuda` only after building a CUDA-capable image and configuring GPU scheduling. |
| PVC size | `20Gi` | Increase when caching multiple models or large model revisions. |
| Memory request | `8Gi` | Tune after observing startup and real audio workloads. |
| Startup probe | 10 minutes | Increase if your registry, model hub, or storage backend is slow. |

## GPU notes

The example Dockerfile is CPU-first. For GPU clusters you need to adapt the image to CUDA-capable PyTorch/FunASR dependencies, then add your cluster's GPU scheduling fields, for example:

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
nodeSelector:
  nvidia.com/gpu.present: "true"
```

Exact GPU labels, runtime classes, and device plugin configuration vary by Kubernetes distribution. Keep the service private until authentication, TLS, upload-size limits, and rate limits are in place.

## Operational checks

- Use `/health` for readiness and `/v1/models` to confirm model aliases.
- Log model alias, device, audio duration, response format, latency, and error text.
- Start with one replica because the cache PVC is `ReadWriteOnce`; scale horizontally with a registry image, per-pod cache, or a shared read-only model cache after measuring memory and startup time.
- Put authentication and network policy in front of the service before exposing it outside a trusted namespace.
- For Dify, n8n, or web backends inside the same cluster, point them at the Kubernetes service name instead of `localhost`.
