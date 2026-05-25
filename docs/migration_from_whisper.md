# Migrate from Whisper or Cloud ASR to FunASR

Use this guide when you already have a Whisper, OpenAI/Cloud ASR, or custom speech pipeline and want to decide whether FunASR is worth switching to. The goal is not to prove a benchmark with one sample file; it is to compare quality, speed, cost, and deployment fit on audio that looks like your real workload.

## When FunASR is a good fit

FunASR is usually worth evaluating when you need one or more of these properties:

- Private or self-hosted transcription where audio should stay inside your environment.
- High-throughput long-form transcription for meetings, archives, media, or call recordings.
- Speaker-aware transcripts with VAD, punctuation, timestamps, and diarization in one pipeline.
- An OpenAI-compatible audio endpoint for agents, Dify, LangChain, AutoGen, or internal apps.
- Streaming ASR or live captions with WebSocket/runtime service support.
- CPU-viable smoke tests before moving to GPU deployment.

Stay on your current pipeline if you need a managed service with no operations work, a vendor SLA, or a language/domain that your own benchmark shows FunASR does not handle well enough yet.

## Fast evaluation plan

1. Pick 20-50 representative audio files. Include short clips, long recordings, noisy samples, different speakers, and the languages or dialects you care about.
2. Run your current Whisper or cloud ASR pipeline exactly as you use it in production. Save transcripts, latency, cost, and failure cases.
3. Run FunASR locally with the README quick start, or use the [migration benchmark example](../examples/migration/) to measure a representative audio folder. Then choose a deployment path from the [deployment matrix](./deployment_matrix.md).
4. Compare output with human review or your normal WER/CER process. Do not compare only one clean demo file.
5. Run the OpenAI-compatible API smoke test if your application already uses OpenAI-style clients.
6. Record warmup time, model download time, device, GPU/CPU type, batch size, and audio duration separately from steady-state throughput.

## Feature mapping

| Existing workflow | FunASR path | What to validate |
|---|---|---|
| Whisper file transcription | [README quick start](../README.md#quick-start) with SenseVoice, Paraformer, or Fun-ASR-Nano | Transcript quality, timestamps, speed, model download, CPU/GPU behavior. |
| Whisper plus pyannote | `spk_model="cam++"` with VAD and punctuation | Speaker labels, speaker changes, overlapping speech, long silences. |
| OpenAI audio API or cloud batch ASR | [OpenAI-compatible API example](../examples/openai_api/) | `/v1/audio/transcriptions`, response format, client compatibility, upload limits. |
| Dify/LangChain/AutoGen agent audio | [Client recipes](../examples/openai_api/CLIENTS.md) or [MCP server](../examples/mcp_server/) | Tool latency, file handling, auth boundary, error reporting. |
| Live captions or call-center streaming | [Runtime service docs](../runtime/readme.md) | Chunking, endpointing, reconnects, backpressure, partial/final result behavior. |
| Subtitle generation | [Subtitle example](../examples/subtitle/) | Segment readability, line length, speaker labels, SRT/VTT compatibility. |
| Offline archive processing | [Batch ASR example](../examples/batch_asr_improved.py) | Manifest handling, retries, progress logs, throughput, failed-file recovery. |

## Minimal local comparison

Install FunASR and run the same file you used for your baseline:

```bash
pip install funasr
```

```python
from funasr import AutoModel

model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    spk_model="cam++",
    device="cuda",  # use "cpu" for a portable smoke test
)
result = model.generate(input="sample.wav")
print(result)
```

For an API-style comparison:

```bash
pip install funasr fastapi uvicorn python-multipart
funasr-server --model sensevoice --device cuda

curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

If you want a repeatable folder-level benchmark, run [`examples/migration/benchmark_funasr.py`](../examples/migration/benchmark_funasr.py) to produce `results.jsonl` and `summary.md` for your own audio set. For a container smoke test, start from `examples/openai_api/docker-compose.yml` and verify it with `BASE_URL=http://localhost:8000 bash examples/openai_api/smoke_test.sh`.

## Quality and speed checklist

Track these fields for both the old pipeline and FunASR:

- Audio duration, language, domain, sample rate, channel count, and speaker count.
- Model name, model version, FunASR version, Python/PyTorch/CUDA versions, and Docker image tag if used.
- Hardware, device mode, batch size, streaming chunk size, and whether warmup/model download is excluded.
- WER/CER or human review notes for names, numbers, punctuation, diarization, timestamps, and domain terms.
- Latency, throughput, GPU/CPU memory, cost per hour of audio, and failed-file rate.
- Operational requirements: authentication, upload limits, TLS, logs, monitoring, retries, and retention rules.

## Rollout checklist

- Keep the old pipeline available until FunASR passes your representative benchmark.
- Start with an internal endpoint or batch job before exposing a public API.
- Add request IDs and log audio duration, model, device, latency, and error type for every request.
- Pin the model alias and deployment command in your runbook.
- Test noisy audio, silence, overlapping speakers, long files, non-UTF-8 filenames, and network interruptions.
- Open a [Deployment Help issue](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md) with your command, logs, model, device, and sample characteristics if you hit a blocker.

## Share the result

If FunASR replaces or complements your existing ASR stack, consider opening a [showcase issue](https://github.com/modelscope/FunASR/issues/new?template=showcase.md). Migration reports with hardware, speed, quality notes, and deployment details help new users choose the right path faster.
