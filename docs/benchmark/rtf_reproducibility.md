# Benchmark RTF and Reproducibility Notes

Use this page when comparing FunASR with Whisper, a cloud ASR provider, a Rust
runtime, or another self-hosted engine. Speed numbers are only useful when the
timing scope, data, model, runtime, and hardware are reported together.

## RTF and RTFx

FunASR benchmark tables usually report throughput as `RTFx`, or "times
realtime":

```text
RTF  = processing_time_seconds / input_audio_seconds
RTFx = input_audio_seconds / processing_time_seconds
     = 1 / RTF
```

For example, an `RTFx` value of `340` means 340 seconds of input audio are
processed in about 1 second, under that benchmark's data, runtime, batching, and
hardware setup. On the public vLLM table, the 184-file set has 11,541 seconds of
audio, so `340x` corresponds to roughly 34 seconds of measured processing time
for the whole set if the same scope is used:

```text
11541 / 340 = 33.94 seconds
```

Do not compare an offline batch `RTFx` result with streaming first-token latency
or end-to-end product latency. They measure different things.
For realtime WebSocket service sizing, use the
[Realtime WebSocket Benchmark](./realtime_ws_benchmark.md) instead.

## Current Public vLLM Benchmark Scope

The vLLM guide currently reports the following public scope for the Fun-ASR-Nano
and GLM-ASR-Nano table:

| Field | Value |
|-------|-------|
| Audio set | 184 long-form files |
| Total audio duration | 11,541 seconds, about 192 minutes |
| Models | Fun-ASR-Nano and GLM-ASR-Nano |
| Reported metric | CER and `RTFx` throughput |
| Fun-ASR-Nano vLLM batch result | `RTFx 340`, CER `8.20%` |
| Fun-ASR-Nano PyTorch baseline | `RTFx 21`, CER `8.06%` |
| Fun-ASR-Nano offline service without speaker diarization | `RTFx 102`, CER `8.14%` |
| Fun-ASR-Nano offline service with speaker diarization | `RTFx 46`, CER `8.19%` |

The table describes offline throughput on the stated long-form set. It should
not be read as a guarantee for every GPU, batch shape, language mix, streaming
chunk size, or service deployment.

The main website benchmark page is a separate public table for the broader ASR
comparison. It reports 184 long-form Chinese audio files, 11,539 seconds total,
and an NVIDIA H100 80GB HBM3 GPU. Keep the two tables separate when citing
numbers: the website table documents the general ASR benchmark, while the vLLM
guide table documents the Fun-ASR-Nano / GLM-ASR-Nano vLLM throughput rows.

## Required Fields for Reproducible Benchmark Claims

When publishing a FunASR benchmark, include these fields with the number:

| Category | What to record |
|----------|----------------|
| Data | File count, total audio duration, language/domain, sample rate, mono/stereo handling, and whether test files are public |
| Model | Model ID, checkpoint source, model revision or commit, language setting, hotwords, and text normalization |
| Runtime | Python SDK, ONNX, C++, vLLM, llama.cpp/GGUF, API server, or another path |
| Hardware | CPU model and thread count, GPU/NPU model, GPU count, memory, driver, CUDA/CANN/runtime versions |
| Software | `funasr`, PyTorch, torchaudio, vLLM, ONNX Runtime, CUDA, Python, and operating system versions |
| Pipeline | VAD, punctuation, speaker diarization, ITN, timestamps, and post-processing on/off |
| Batching | Batch size, `batch_size_s`, concurrent requests, tensor parallel size, chunk size, VAD segment policy |
| Timing scope | Whether timing includes model download, cold start, warmup, file I/O, audio decoding/resampling, VAD, post-processing, and result serialization |
| Quality | CER/WER method, reference normalization, ignored tokens, and failed-file handling |

For official README or website numbers, include the fields above or link to a
report that includes them.

## Suggested Timing Protocol

1. Put all input audio in a manifest or directory and compute total duration
   before running ASR.
2. Warm the model once if the published number is intended to represent steady
   state throughput. If you include cold start, say so explicitly.
3. Time exactly one scope: model-only, pipeline-only, or end-to-end service.
4. Run the same scope at least three times and report median plus min/max.
5. Keep transcript output, failed-file list, and timing JSON/CSV with the run.

For migration or product evaluation, start from
[`examples/migration/benchmark_funasr.py`](../../examples/migration/benchmark_funasr.py).
It writes per-file timing and a Markdown summary for your own audio set. The
same reporting fields above also apply when you use vLLM, ONNX, C++, GGUF, or a
custom runtime instead of the migration example.

## Comparing with a Rust or Other Custom Runtime

For a fair engine-to-engine comparison:

- use the same audio files and total duration;
- resample and downmix with the same policy;
- keep VAD, punctuation, speaker diarization, and timestamps either all on or
  all off;
- compare both speed and quality, because a faster decode path can change CER;
- report `RTFx` and raw processing seconds, not only a relative speedup.

If you can share your result publicly, open a Migration Benchmark Report issue
with the fields above. That makes the comparison useful to other users and gives
maintainers enough context to reproduce or improve the path.
