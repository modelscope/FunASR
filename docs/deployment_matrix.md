# FunASR Deployment Matrix

Use this page to choose the shortest deployment path for a product, demo, benchmark, or internal workflow. Start with the smallest surface that satisfies the job, then move to heavier runtimes only when throughput, latency, or integration requirements demand it.

## Quick decision table

| Path | Best for | Start here | Operational notes |
|---|---|---|---|
| Colab notebook | Browser smoke tests, first evaluation, shareable demos | [Colab quickstart](../examples/colab/) | No local setup; first run downloads model files, GPU runtime is faster. |
| Python API | Notebooks, offline jobs, first model evaluation | [README quick start](../README.md#quick-start) | Lowest ceremony; caller owns batching, retries, and files. |
| OpenAI-compatible API | Private speech API, agents, Dify/LangChain/AutoGen-style clients | [OpenAI API example](../examples/openai_api/) | Easiest integration for apps that already support OpenAI audio APIs. |
| Docker Compose API | Reproducible local smoke test or small internal service | [OpenAI API Docker docs](../examples/openai_api/#docker-deployment) | CPU by default; adapt the image before using CUDA in containers. |
| Kubernetes API | Internal speech API for cluster services | [Kubernetes template](../examples/openai_api/kubernetes/) | Starts as private `ClusterIP`; add auth, TLS, network policy, and GPU scheduling before broader exposure. |
| Runtime WebSocket service | Live captions, meetings, call-center streams | [Runtime service docs](../runtime/readme.md) | Use when partial results, endpointing, or long-lived audio streams matter. |
| vLLM acceleration | Higher-throughput LLM-based ASR with Fun-ASR-Nano | [vLLM guide](./vllm_guide.md) | Use for LLM decoder throughput; does not apply to non-autoregressive Paraformer. |
| MCP server | Claude/Cursor/desktop agent speech tools | [MCP example](../examples/mcp_server/) | Good when the ASR result should be exposed as a local tool. |
| Subtitle generator | SRT/VTT from long audio or video | [Subtitle example](../examples/subtitle/) | Use verbose segments and speaker labels when readability matters. |
| Batch ASR script | Archives, meetings, datasets, repeated offline runs | [Batch example](../examples/batch_asr_improved.py) | Add queueing, manifests, and retry logs for production use. |
| Triton runtime | Specialized high-performance serving | [Triton runtime docs](../runtime/triton_gpu/README.md) | Heavier setup; choose when your team already operates Triton/GPU serving. |

## Common choices

### I want to try FunASR in five minutes

Use the [Colab quickstart](../examples/colab/) when you want a browser-only smoke test, or use the Python API from the README for local work. It is the shortest route for validating installation, model download, device selection, and basic output shape. If you are unsure which model to start with, use the [model selection guide](./model_selection.md).

### I want a local replacement for cloud transcription

Use the OpenAI-compatible API. It exposes `/v1/audio/transcriptions`, `/v1/models`, `/health`, and Swagger docs. Start with `sensevoice`, run `examples/openai_api/smoke_test.sh` or `examples/openai_api/smoke_test.py`, then connect existing SDK or HTTP clients using [client recipes](../examples/openai_api/CLIENTS.md) and [JavaScript/TypeScript recipes](../examples/openai_api/JAVASCRIPT.md). For browser upload or microphone demos, use the [Gradio browser demo](../examples/openai_api/GRADIO.md). For Dify, n8n, HTTP nodes, or webhook workers, follow the [workflow recipes](../examples/openai_api/WORKFLOWS.md). For API gateways, developer portals, and schema-driven imports, use the [OpenAPI spec](../examples/openai_api/OPENAPI.md). Before sharing the service, review the [security and gateway guide](../examples/openai_api/SECURITY.md).

### I want a repeatable container demo

Use `examples/openai_api/docker-compose.yml` for a CPU-mode smoke test:

```bash
cd examples/openai_api
cp .env.example .env
docker compose up --build
```

Keep CPU mode until you have a CUDA-capable PyTorch/FunASR image. After that, set `FUNASR_DEVICE=cuda` and verify with the same smoke test. Use `python examples/openai_api/smoke_test.py --base-url http://localhost:8000` on systems without bash/curl.

### I want an internal Kubernetes service

Use the [Kubernetes template](../examples/openai_api/kubernetes/) for a private `ClusterIP` OpenAI-compatible API with persistent model cache, `/health` probes, and a port-forward smoke-test path. Keep the CPU default until you have a CUDA-capable image and cluster GPU scheduling in place.

### I need streaming or live captioning

Use the runtime WebSocket service. Validate chunk size, VAD, endpointing, punctuation, speaker diarization, reconnect behavior, and client backpressure with real audio before production rollout.

### I need more LLM-based ASR throughput

Use the vLLM path for Fun-ASR-Nano. Benchmark with your own audio distribution and watch GPU memory, tensor parallel size, first-token latency, and warmup time.

### I want to run Fun-ASR-Nano on Ascend NPU

Fun-ASR-Nano's LLM-based path is currently documented and validated for CUDA/vLLM, standard PyTorch CPU/GPU runs, and CPU/edge GGUF runtimes. Ascend NPU (`torch_npu`) is not an officially validated runtime for this model yet. Do not assume that a SenseVoice or Paraformer NPU deployment means Fun-ASR-Nano will also work, because Nano also exercises the Qwen decoder, `inputs_embeds`, and autocast path. If you are adapting it, start with `torch.bfloat16`, capture the `torch` / `torch_npu` / CANN versions, and open a focused PR or deployment issue with a minimal command and full stack trace.

## Readiness checklist

- Pick a model alias and pin it in deployment notes.
- Record FunASR version, model version, device, CUDA/PyTorch version, Docker image tag, and command line.
- Run a short public smoke sample and at least one realistic private sample.
- Log audio duration, model, device, latency, response format, and error type for every request.
- Add upload-size limits, authentication, TLS, and rate limits before exposing an API outside a trusted network; use the [security and gateway guide](../examples/openai_api/SECURITY.md) to plan the boundary.
- For streaming, test silence, noise, overlapping speakers, long sessions, reconnects, and slow clients.
- For benchmark claims, include input duration, hardware, batch size, model, runtime path, and whether model download/warmup time is excluded.

## When to open an issue

Use [Deployment Help](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md) for runtime, Docker, vLLM, Triton, Android, browser, or agent integration problems. Include your deployment path, exact command/config, logs, model, device, and audio characteristics.
