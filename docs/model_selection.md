# FunASR Model Selection Guide

Use this guide when you are choosing a first model, comparing FunASR with Whisper or a cloud ASR provider, or deciding which model alias to expose through the OpenAI-compatible API.

## Fast default path

For GPU evaluation of Chinese, English, Japanese, or Chinese dialects and regional accents, start with the flagship **Fun-ASR-Nano** (SenseVoice encoder + Qwen3 decoder). Compare it on your own audio before choosing a production model:

```python
from funasr import AutoModel

model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512", device="cuda")
result = model.generate(input="meeting.wav")
print(result[0]["text"])
```

For non-autoregressive multilingual ASR with emotion/event tags, or a CPU evaluation path, start with **SenseVoice-Small**. The example below adds separate VAD and speaker-processing stages for meeting transcripts; speaker diarization is not a SenseVoice output from the same recognition pass:

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

SenseVoice produces transcription and rich tags. `fsmn-vad` locates speech;
`cam++` produces speaker embeddings, which the pipeline clusters into anonymous
labels within a recording. These labels do not identify registered people and
are not stable identities across recordings.

Switch to Paraformer when your workload is Mandarin-only and you want character-level timestamps or hotwords.

## Decision table

| Need | Start with | Why | Next doc |
|---|---|---|---|
| Fast multilingual private transcription | SenseVoice-Small | Strong default with ASR, emotion tags, audio event tags, and CPU viability. | [README quick start](../README.md#quick-start) |
| Mandarin production ASR | Paraformer-Large | Mature Chinese ASR path with VAD and punctuation. | [Tutorial](./tutorial/README.md) |
| English-only route in the OpenAI API example | `paraformer-en` alias | Smaller English route for API compatibility checks. | [OpenAI API example](../examples/openai_api/) |
| LLM-based ASR or Chinese/English/Japanese + dialect experiments | Fun-ASR-Nano | Evaluate the Python path first; split-engine and native vLLM use different loading contracts. | [Choose a vLLM path](#vllm-checkpoint-paths) |
| Offline long-form ASR with anonymous diarization | MOSS-Transcribe-Diarize | One offline request returns transcription, timestamps, and per-recording anonymous speaker labels; it does not identify known people and needs no external VAD or speaker model. | [MOSS deployment guide](./moss_transcribe_diarize.md) |
| Live captions or call-center streams | Runtime WebSocket service | Designed for long-lived streaming sessions and partial results. | [Runtime service docs](../runtime/readme.md) |
| Batch archive processing | SenseVoice-Small or Paraformer-Large | Stable offline transcription path; caller owns manifests, retries, and logs. | [Batch ASR example](../examples/batch_asr_improved.py) |
| Migration from Whisper/cloud ASR | SenseVoice-Small first, then benchmark alternatives | Gives a strong baseline before deeper model-specific tuning. | [Migration guide](./migration_from_whisper.md) |

## OpenAI-compatible API aliases

The `examples/openai_api` server exposes short aliases so application teams do not need to know model repository IDs:

- **`sensevoice`** uses `iic/SenseVoiceSmall` for multilingual HTTP transcription on CPU/GPU. Returned text has rich tags removed.
- **`paraformer`** uses `paraformer-zh` with VAD and punctuation for a Mandarin-oriented route.
- **`paraformer-en`** uses `paraformer-en` with VAD for English transcription in OpenAI-style clients.
- **`fun-asr-nano`** uses `FunAudioLLM/Fun-ASR-Nano-2512` for evaluating Chinese, English, Japanese, and Chinese dialect/accent coverage. Select a compatible runtime when evaluating vLLM acceleration.
- **`moss-transcribe-diarize`** uses the third-party `OpenMOSS-Team/MOSS-Transcribe-Diarize` model for offline transcription and anonymous per-recording speaker labels. Prepare its separate dependencies and reviewed remote code using the [MOSS guide](./moss_transcribe_diarize.md); request `verbose_json` for structured segments. It does not require an external VAD/speaker model and does not identify known people.

These aliases describe [the example server](../examples/openai_api/server.py),
which loads `AutoModel`. They do not configure native vLLM or automatically select
`AutoModelVLLM`. The packaged `funasr-server` has a separate loader and backend
selection; do not copy an alias or a performance result between services without
checking the corresponding [HTTP guide](../examples/openai_api/README.md).

The example HTTP service cleans both top-level `text` and segment `text` in
`verbose_json`; that format does not restore emotion/event tags. If you need
the original tags, use the Python SDK and preserve the returned `text` before
display postprocessing. See the [raw-tag recipe](./speaker_emotion.md).

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
| LLM-based ASR throughput | Choose split-engine or native vLLM below | Match the checkpoint, loading API and tested environment; this is not a Paraformer backend. |

See the [deployment matrix](./deployment_matrix.md) when you are choosing between these paths.

<a id="vllm-checkpoint-paths"></a>

## Choose the vLLM Checkpoint and Interface

| Path | Checkpoint and interface | Read next |
| --- | --- | --- |
| FunASR split-engine | Base `FunAudioLLM/Fun-ASR-Nano-2512` assets through `AutoModelVLLM`; FunASR handles the audio side and vLLM the decoder. | [Split-engine preparation and limits](./vllm_guide.md) |
| Official native vLLM | Converted `FunAudioLLM/Fun-ASR-Nano-2512-vllm` snapshot through vLLM's native model implementation and `/v1/audio/transcriptions`. Not an `AutoModelVLLM` load. | [Official functional validation](./vllm_official_native_validation.md) |
| Historical community native vLLM | Community `allendou/Fun-ASR-Nano-2512-vllm`, tested on 2026-08-13. Its timings belong to that checkpoint and environment. | [Historical community record](./vllm_native_funasr_validation.md) |

The official record pins a model revision and an existing environment; it is not
a clean-install recipe, a sustained-load benchmark, or proof of `/v1/realtime`
streaming. Do not reuse the historical community timings for the official model.
For MOSS, follow its own deployment guide: the Nano checkpoints and validation
above do not establish MOSS runtime compatibility. Choose model, checkpoint,
interface and environment together before testing your own workload.

## Benchmark before committing

Do not choose a model from a single clean demo file. Use a small representative set first:

- 20-50 audio files that cover short clips, long meetings, silence, noise, overlapping speakers, domain vocabulary, and target languages.
- Record model name, model revision, FunASR version, device, CPU/GPU type, CUDA/PyTorch version, runtime path, batch size, and whether warmup/model download time is excluded.
- Track quality with your normal WER/CER or human review process, not only transcript readability.
- Track latency, throughput, memory, failures, and upload size limits together.
- Keep at least one public sample for smoke tests and at least one private realistic sample for deployment validation.

For migration work, use the [migration benchmark example](../examples/migration/) and the [migration guide](./migration_from_whisper.md).

## Practical recommendations

- With a GPU, evaluate Fun-ASR-Nano for Chinese, English, Japanese, and Chinese dialects/accents. Include difficult audio, context and proper nouns in your own comparison. For the separate 31-language checkpoint, use Fun-ASR-MLT-Nano.
- On CPU, or for multilingual + emotion workloads, use SenseVoice-Small (fast non-autoregressive, CPU-viable).
- Use Paraformer when your production traffic is primarily Mandarin and you want timestamps or hotwords.
- For offline long recordings that need anonymous per-recording speaker labels in the same request, use MOSS-Transcribe-Diarize; it is not a realtime WebSocket or known-person identification path.
- Use the streaming runtime when partial results and long-lived connections matter more than a single final transcript.
- Keep model aliases stable in production runbooks so benchmark results and bug reports are reproducible.
- Open a [Deployment Help issue](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md) with model, device, command, logs, audio duration, and runtime path when you get stuck.
