# FunASR Model Selection Guide

Use this guide when you are choosing a first model, comparing FunASR with Whisper or a cloud ASR provider, or deciding which model alias to expose through the OpenAI-compatible API.

## Fast default path

If you have a GPU, start with the flagship **Fun-ASR-Nano** — an LLM-based ASR model (SenseVoice encoder + a Qwen3 decoder) covering 31 languages, with the strongest accuracy on hard cases, context, and proper nouns:

```python
from funasr import AutoModel

model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512", device="cuda")
result = model.generate(input="meeting.wav")
print(result[0]["text"])
```

On CPU, or when you want multilingual + emotion/event tags and speaker-aware meeting transcripts in one fast non-autoregressive pass, use **SenseVoice-Small**:

```python
from funasr import AutoModel

model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    spk_model="cam++",
    device="cuda",  # use "cpu" for a portable smoke test
)
result = model.generate(input="meeting.wav")
```

Switch to Paraformer when your workload is Mandarin-only and you want character-level timestamps or hotwords.

## Decision table

| Need | Start with | Why | Next doc |
|---|---|---|---|
| Fast multilingual private transcription | SenseVoice-Small | Strong default with ASR, emotion tags, audio event tags, and CPU viability. | [README quick start](../README.md#quick-start) |
| Mandarin production ASR | Paraformer-Large | Mature Chinese ASR path with VAD and punctuation. | [Tutorial](./tutorial/README.md) |
| English-only route in the OpenAI API example | `paraformer-en` alias | Smaller English route for API compatibility checks. | [OpenAI API example](../examples/openai_api/) |
| LLM-based ASR or 31-language experiments | Fun-ASR-Nano | LLM-based model path; use vLLM when decoder throughput matters. | [vLLM guide](./vllm_guide.md) |
| Live captions or call-center streams | Runtime WebSocket service | Designed for long-lived streaming sessions and partial results. | [Runtime service docs](../runtime/readme.md) |
| Batch archive processing | SenseVoice-Small or Paraformer-Large | Stable offline transcription path; caller owns manifests, retries, and logs. | [Batch ASR example](../examples/batch_asr_improved.py) |
| Migration from Whisper/cloud ASR | SenseVoice-Small first, then benchmark alternatives | Gives a strong baseline before deeper model-specific tuning. | [Migration guide](./migration_from_whisper.md) |

## OpenAI-compatible API aliases

The `examples/openai_api` server exposes short aliases so application teams do not need to know model repository IDs:

| Alias | Underlying path | Use when |
|---|---|---|
| `sensevoice` | `iic/SenseVoiceSmall` | You want the default private speech API with multilingual ASR, event tags, and good CPU/GPU behavior. |
| `paraformer` | `paraformer-zh` with VAD and punctuation | You want a Mandarin-oriented production route. |
| `paraformer-en` | `paraformer-en` with VAD | You want a compact English route in OpenAI-style clients. |
| `fun-asr-nano` | `FunAudioLLM/Fun-ASR-Nano-2512` | You are evaluating LLM-based ASR, 31-language coverage, or vLLM acceleration. |

For Ascend NPU deployments, treat `fun-asr-nano` separately from SenseVoice / Paraformer. The Fun-ASR-Nano PyTorch `AutoModel` path has community compatibility evidence on 310P3 after the NPU autocast fix, but it was much slower than CPU in that smoke test; `AutoModelVLLM` still depends on vLLM-Ascend operator support and has hit Qwen3 rotary / `TransData` failures. Use CUDA/vLLM, standard PyTorch CPU/GPU, or GGUF runtime for production unless you are actively validating an Ascend backend.

Check the live service before wiring clients:

```bash
curl http://localhost:8000/v1/models
python examples/openai_api/smoke_test.py --base-url http://localhost:8000 --model sensevoice
```

For SDK, JavaScript, workflow, Postman, OpenAPI, Docker, and Kubernetes paths, start from the [OpenAI API example](../examples/openai_api/).

## Runtime choice by workload

| Workload | Runtime path | Notes |
|---|---|---|
| Notebook or one-off evaluation | Python `AutoModel` | Shortest path for install, model download, and output-shape checks. |
| Internal HTTP service | OpenAI-compatible API | Reuse OpenAI-style clients, Dify, n8n, LangChain, AutoGen, and HTTP nodes. |
| Repeatable local container demo | Docker Compose API | CPU-first smoke test; adapt the image before using CUDA. |
| Internal cluster service | Kubernetes API template | Private `ClusterIP`, persistent model cache, `/health` probes, and port-forward smoke test. |
| Live audio | Runtime WebSocket service | Validate chunk size, VAD, endpointing, reconnects, and client backpressure with real audio. |
| LLM-based ASR throughput | vLLM path for Fun-ASR-Nano | vLLM accelerates autoregressive decoding; it does not apply to non-autoregressive Paraformer. |

See the [deployment matrix](./deployment_matrix.md) when you are choosing between these paths.

## Benchmark before committing

Do not choose a model from a single clean demo file. Use a small representative set first:

- 20-50 audio files that cover short clips, long meetings, silence, noise, overlapping speakers, domain vocabulary, and target languages.
- Record model name, model revision, FunASR version, device, CPU/GPU type, CUDA/PyTorch version, runtime path, batch size, and whether warmup/model download time is excluded.
- Track quality with your normal WER/CER or human review process, not only transcript readability.
- Track latency, throughput, memory, failures, and upload size limits together.
- Keep at least one public sample for smoke tests and at least one private realistic sample for deployment validation.

For migration work, use the [migration benchmark example](../examples/migration/) and the [migration guide](./migration_from_whisper.md).

## Practical recommendations

- With a GPU, default to Fun-ASR-Nano — the flagship LLM-based model (31 languages), strongest on hard, contextual, and proper-noun-heavy audio.
- On CPU, or for multilingual + emotion workloads, use SenseVoice-Small (fast non-autoregressive, CPU-viable).
- Use Paraformer when your production traffic is primarily Mandarin and you want timestamps or hotwords.
- Use the streaming runtime when partial results and long-lived connections matter more than a single final transcript.
- Keep model aliases stable in production runbooks so benchmark results and bug reports are reproducible.
- Open a [Deployment Help issue](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md) with model, device, command, logs, audio duration, and runtime path when you get stuck.
