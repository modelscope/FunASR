# FunASR Use-case Showcase

FunASR is useful far beyond a single offline transcription command. This page collects the fastest paths for developers who want to evaluate, deploy, or integrate speech understanding in real products.

## Choose the right path

| Goal | Start here | Why it matters |
|---|---|---|
| Transcribe one file locally | [README quick start](../README.md#quick-start) | Verify install and model download in minutes. |
| Compare accuracy and speed | [Benchmark report](https://modelscope.github.io/FunASR/benchmark.html) | Reproduce the 184-file long-audio benchmark before choosing a model. |
| Migrate from Whisper/cloud ASR | [Migration guide](./migration_from_whisper.md) | Map existing pipelines to FunASR, benchmark representative audio, and plan a safe rollout. |
| Build a private speech API | [OpenAI-compatible API example](../examples/openai_api/), [client recipes](../examples/openai_api/CLIENTS.md), and [workflow recipes](../examples/openai_api/WORKFLOWS.md) | Reuse LangChain, Dify, n8n, AutoGen, and other OpenAI-style clients without sending audio to a cloud ASR provider. |
| Add speech input to agents | [MCP server](../examples/mcp_server/) and [voice input](../examples/voice_input/) | Connect local ASR to Claude, Cursor, and desktop agent workflows. |
| Choose a deployment path | [Deployment matrix](./deployment_matrix.md) | Compare Python API, OpenAI API, Docker Compose, WebSocket, vLLM, MCP, batch, subtitles, and Triton. |
| Serve streaming ASR | [Runtime service docs](../runtime/readme.md) | Run WebSocket or service-mode ASR for live captioning and call-center style workloads. |
| Accelerate LLM-based ASR | [vLLM guide](./vllm_guide.md) | Use tensor parallel decoding and streaming service support for Fun-ASR-Nano. |
| Generate subtitles | [Subtitle example](../examples/subtitle/) | Turn long audio or video into subtitle files for media workflows. |
| Process many recordings | [Batch ASR example](../examples/batch_asr_improved.py) | Build repeatable offline jobs for archives, meetings, and datasets. |

## Production-oriented recipes

### Private transcription API

Use this path when an application already speaks OpenAI-style APIs or when audio cannot leave your environment.

```bash
pip install funasr fastapi uvicorn python-multipart
funasr-server --model sensevoice --device cuda
```

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

Recommended next steps:

- Run the [OpenAI-compatible API smoke test](../examples/openai_api/smoke_test.sh).
- Add authentication and network controls at your service boundary.
- Record model name, device, driver, and audio duration in bug reports and benchmarks.

### Agent speech input

Use this path when you want to talk to coding agents, internal assistants, or workflow tools.

- Start with the [MCP server example](../examples/mcp_server/) for Claude/Cursor-style tools.
- Use the [voice input example](../examples/voice_input/) for desktop speech input experiments.
- Keep latency visible: log audio duration, processing time, and selected model for each request.

### Streaming and call-center workloads

Use this path when partial results and low perceived latency matter more than a single final transcript.

- Start from the [runtime service docs](../runtime/readme.md).
- Pair ASR with VAD, punctuation, and speaker diarization when the transcript needs to be readable by humans.
- Validate with realistic audio: background noise, long silence, overlapping speakers, and different microphone quality.

### Benchmark before migrating from Whisper

Use this path when deciding whether FunASR is a good replacement for Whisper or a cloud ASR provider.

- Follow the [migration guide](./migration_from_whisper.md) to map features and benchmark representative audio.
- Read the [public benchmark report](https://modelscope.github.io/FunASR/benchmark.html).
- Benchmark your own sample set before migration; include both short clips and long-form recordings.
- Track cost and throughput together: GPU speed, CPU viability, model download size, and deployment complexity.

## Model selection hints

| Need | Good first choice | Notes |
|---|---|---|
| Fast multilingual transcription | SenseVoice-Small | Strong default for local demos and private APIs. |
| Mandarin production ASR | Paraformer-Large | Mature choice for Chinese speech recognition. |
| LLM-based ASR experiments | Fun-ASR-Nano | Pair with the [vLLM guide](./vllm_guide.md) when throughput matters. |
| Speaker-aware transcripts | SenseVoice or Paraformer with `spk_model="cam++"` | Useful for meetings, interviews, and customer calls. |
| Live audio | Runtime WebSocket service | Validate chunking, VAD, and endpointing with real traffic. |

## Share your result

If FunASR works well in your project, consider opening a [showcase issue](https://github.com/modelscope/FunASR/issues/new?template=showcase.md), [Migration Benchmark Report](https://github.com/modelscope/FunASR/issues/new?template=migration_benchmark.md), or GitHub Discussion with:

- Use case and deployment mode.
- Model, device, and processing speed.
- Audio domain, language, and rough duration.
- A public demo, screenshot, benchmark summary, or integration link when available.

Concrete usage reports help new users choose the right path and help maintainers prioritize the next round of docs and examples.
