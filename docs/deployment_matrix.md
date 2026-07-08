# FunASR Deployment Matrix

Use this page to choose the shortest deployment path for a product, demo, benchmark, or internal workflow. Start with the smallest surface that satisfies the job, then move to heavier runtimes only when throughput, latency, or integration requirements demand it.

## Quick decision table

| Path | Best for | Start here | Operational notes |
|---|---|---|---|
| Colab notebook | Browser smoke tests, first evaluation, shareable demos | [Colab quickstart](../examples/colab/) | No local setup; first run downloads model files, GPU runtime is faster. |
| Python API | Notebooks, offline jobs, first model evaluation | [README quick start](../README.md#quick-start) | Lowest ceremony; caller owns batching, retries, and files. |
| OpenAI-compatible API | Private speech API, agents, Dify/LangChain/AutoGen-style clients | [OpenAI API example](../examples/openai_api/) | Easiest integration for apps that already support OpenAI audio APIs. |
| Xinference | Teams that already operate Xinference model serving | [Xinference repository](https://github.com/xorbitsai/inference) | Use a Xinference version containing [xorbitsai/inference#5140](https://github.com/xorbitsai/inference/pull/5140) so Fun-ASR-Nano uses packaged `funasr~=1.3.0` instead of the old pinned git commit. |
| Docker Compose API | Reproducible local smoke test or small internal service | [OpenAI API Docker docs](../examples/openai_api/#docker-deployment) | CPU by default; adapt the image before using CUDA in containers. |
| Kubernetes API | Internal speech API for cluster services | [Kubernetes template](../examples/openai_api/kubernetes/) | Starts as private `ClusterIP`; add auth, TLS, network policy, and GPU scheduling before broader exposure. |
| Runtime WebSocket service | Live captions, meetings, call-center streams | [Runtime service docs](../runtime/readme.md) | Use when partial results, endpointing, or long-lived audio streams matter. |
| ONNX/C++ runtime | High-concurrency CPU services or embedded realtime ASR | [ONNX runtime docs](../runtime/onnxruntime/readme.md) | Keep this path when latency/concurrency is already proven; add text post-processing for fixed business terms before moving to GPU LLMs. |
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

### I already run Xinference

Use Xinference when your stack already standardizes on its model registry,
virtualenv isolation, and serving lifecycle. Make sure your Xinference build
includes [xorbitsai/inference#5140](https://github.com/xorbitsai/inference/pull/5140);
that update moved the Fun-ASR-Nano model specs from an old FunASR git SHA to the
packaged `funasr~=1.3.0` dependency line. For first-time FunASR evaluation or
agent-oriented OpenAI-compatible transcription, start with the native FunASR
OpenAI API example above instead.

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

### I already have high-concurrency ONNX and need hotwords

Do not move an entire working CPU ONNX/C++ realtime stack to a GPU LLM only
because the current path lacks hotwords. First decide what "hotword" means in
your product:

- For fixed company names, product names, or common misrecognitions, keep the
  ONNX path and add deterministic text post-processing on final results. This
  preserves proven CPU concurrency and is easier to audit.
- For true decoder-time biasing, pronunciation ambiguity, or multilingual
  accuracy gaps, test a GPU path such as Fun-ASR-Nano or Qwen3-ASR, but size it
  against your target first-token latency, final latency, audio chunk length,
  VAD policy, and simultaneous speaking sessions.
- If both are needed, use the GPU model only where it wins on quality or
  language coverage, and keep ONNX for the high-volume traffic that already
  meets latency and accuracy requirements.

When opening a deployment issue, include your current ONNX concurrency, CPU
cores, target terms, acceptable end-to-end latency, GPU model, model command,
and whether hotwords are post-recognition corrections or decoder-time biasing.

### I need more LLM-based ASR throughput

Use the vLLM path for Fun-ASR-Nano. Benchmark with your own audio distribution and watch GPU memory, tensor parallel size, first-token latency, and warmup time.

### I want to run Fun-ASR-Nano on Ascend NPU

Fun-ASR-Nano's LLM-based path is documented and validated for CUDA/vLLM, standard PyTorch CPU/GPU runs, and CPU/edge GGUF runtimes. Ascend NPU (`torch_npu`) is still not an official production runtime for this model. Do not assume that a SenseVoice or Paraformer NPU deployment means Fun-ASR-Nano will also work, because Nano also exercises the Qwen decoder, `inputs_embeds`, and backend-specific autocast/operator paths.

A community smoke test in [#3034](https://github.com/modelscope/FunASR/issues/3034) confirmed that the PyTorch `AutoModel(..., device="npu:*")` path can now enter the NPU backend after the autocast device fix, and produced the expected transcript on a 310P3 with `torch_npu 2.5.1` / CANN 8.5.1. That run was much slower than CPU (`rtf_avg` about 124 on NPU vs about 1.9 on CPU), so treat it as compatibility evidence, not a performance recommendation. The same report showed `AutoModelVLLM` on `vllm_ascend 0.9.2rc1` failing inside Qwen3 rotary embedding / `TransData` operator support; debug that path as a vLLM-Ascend runtime/operator issue and capture logs with `ASCEND_LAUNCH_BLOCKING=1`.

If you are adapting this backend, keep the first PR narrow: start with `torch.bfloat16`, capture `torch` / `torch_npu` / CANN / driver / NPU model, separate PyTorch `AutoModel` from `AutoModelVLLM`, and attach the minimal command plus full stack trace.

## Readiness checklist

- Pick a model alias and pin it in deployment notes.
- Record FunASR version, model version, device, CUDA/PyTorch version, Docker image tag, and command line.
- Run a short public smoke sample and at least one realistic private sample.
- Log audio duration, model, device, latency, response format, and error type for every request.
- Add upload-size limits, authentication, TLS, and rate limits before exposing an API outside a trusted network; use the [security and gateway guide](../examples/openai_api/SECURITY.md) to plan the boundary.
- For hotword or correction requirements, record whether the change is
  deterministic post-processing or decoder-time biasing, then benchmark quality
  and latency before replacing an existing runtime.
- For streaming, test silence, noise, overlapping speakers, long sessions, reconnects, and slow clients.
- For benchmark claims, include input duration, hardware, batch size, model, runtime path, and whether model download/warmup time is excluded.

## When to open an issue

Use [Deployment Help](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md) for runtime, Docker, vLLM, Triton, Android, browser, or agent integration problems. Include your deployment path, exact command/config, logs, model, device, and audio characteristics.
