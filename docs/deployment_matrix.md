# FunASR Deployment Matrix

Use this page to choose the shortest deployment path for a product, demo, benchmark, or internal workflow. Start with the smallest surface that satisfies the job, then move to heavier runtimes only when throughput, latency, or integration requirements demand it.

## Quick decision table

| Path | Best for | Start here | Operational notes |
|---|---|---|---|
| Python API | Notebooks, offline jobs, first model evaluation | [README quick start](../README.md#quick-start) | Lowest ceremony; caller owns batching, retries, and files. |
| OpenAI-compatible API | Private speech API, agents, Dify/LangChain/AutoGen-style clients | [OpenAI API example](../examples/openai_api/) | Easiest integration for apps that already support OpenAI audio APIs. |
| Docker Compose API | Reproducible local smoke test or small internal service | [OpenAI API Docker docs](../examples/openai_api/#docker-deployment) | CPU by default; adapt the image before using CUDA in containers. |
| Runtime WebSocket service | Live captions, meetings, call-center streams | [Runtime service docs](../runtime/readme.md) | Use when partial results, endpointing, or long-lived audio streams matter. |
| vLLM acceleration | Higher-throughput LLM-based ASR with Fun-ASR-Nano | [vLLM guide](./vllm_guide.md) | Use for LLM decoder throughput; does not apply to non-autoregressive Paraformer. |
| MCP server | Claude/Cursor/desktop agent speech tools | [MCP example](../examples/mcp_server/) | Good when the ASR result should be exposed as a local tool. |
| Subtitle generator | SRT/VTT from long audio or video | [Subtitle example](../examples/subtitle/) | Use verbose segments and speaker labels when readability matters. |
| Batch ASR script | Archives, meetings, datasets, repeated offline runs | [Batch example](../examples/batch_asr_improved.py) | Add queueing, manifests, and retry logs for production use. |
| Triton runtime | Specialized high-performance serving | [Triton runtime docs](../runtime/triton_gpu/README.md) | Heavier setup; choose when your team already operates Triton/GPU serving. |

## Common choices

### I want to try FunASR in five minutes

Use the Python API from the README. It is the shortest route for validating installation, model download, device selection, and basic output shape.

### I want a local replacement for cloud transcription

Use the OpenAI-compatible API. It exposes `/v1/audio/transcriptions`, `/v1/models`, `/health`, and Swagger docs. Start with `sensevoice`, run `examples/openai_api/smoke_test.sh`, then connect existing SDK or HTTP clients using [client recipes](../examples/openai_api/CLIENTS.md).

### I want a repeatable container demo

Use `examples/openai_api/docker-compose.yml` for a CPU-mode smoke test:

```bash
cd examples/openai_api
cp .env.example .env
docker compose up --build
```

Keep CPU mode until you have a CUDA-capable PyTorch/FunASR image. After that, set `FUNASR_DEVICE=cuda` and verify with the same smoke test.

### I need streaming or live captioning

Use the runtime WebSocket service. Validate chunk size, VAD, endpointing, punctuation, speaker diarization, reconnect behavior, and client backpressure with real audio before production rollout.

### I need more LLM-based ASR throughput

Use the vLLM path for Fun-ASR-Nano. Benchmark with your own audio distribution and watch GPU memory, tensor parallel size, first-token latency, and warmup time.

## Readiness checklist

- Pick a model alias and pin it in deployment notes.
- Record FunASR version, model version, device, CUDA/PyTorch version, Docker image tag, and command line.
- Run a short public smoke sample and at least one realistic private sample.
- Log audio duration, model, device, latency, response format, and error type for every request.
- Add upload-size limits, authentication, TLS, and rate limits before exposing an API outside a trusted network.
- For streaming, test silence, noise, overlapping speakers, long sessions, reconnects, and slow clients.
- For benchmark claims, include input duration, hardware, batch size, model, runtime path, and whether model download/warmup time is excluded.

## When to open an issue

Use [Deployment Help](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md) for runtime, Docker, vLLM, Triton, Android, browser, or agent integration problems. Include your deployment path, exact command/config, logs, model, device, and audio characteristics.
