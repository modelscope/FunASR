# Realtime WebSocket Benchmark

Use this benchmark when you need to measure the client-observable behavior of
`examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py` under
real streaming traffic. Offline `RTFx` and realtime service latency are
different metrics: this page focuses on first update latency, final latency
after `STOP`, response lag, and multi-client behavior.

The benchmark client accepts only 16 kHz mono PCM16 WAV input. Keeping the input
format strict removes resampling and file decoding from the measurement.

## Start the Service

For long continuous speech or multiple browser clients, start with a bounded
partial window and a moderate partial refresh interval:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py \
    --port 10095 --language 中文 \
    --partial-window-sec 8 --decode-interval 0.8
```

Speaker diarization is disabled by default. Add `--enable-spk` only when the
`spk` field is required, and report that setting with the benchmark result.

## Run a Single Realtime Replay

```bash
python examples/industrial_data_pretraining/fun_asr_nano/realtime_ws_benchmark.py \
    audio_16k_mono_pcm16.wav \
    --server ws://localhost:10095 \
    --clients 1 \
    --output-jsonl realtime_ws_1c.jsonl
```

With pacing enabled, the client sends audio at realtime speed using 100 ms
frames. This is the closest mode to a microphone or browser stream.

## Run Concurrent Replays

```bash
python examples/industrial_data_pretraining/fun_asr_nano/realtime_ws_benchmark.py \
    audio_16k_mono_pcm16.wav \
    --server ws://localhost:10095 \
    --clients 8 \
    --loops 3 \
    --chunk-ms 100 \
    --language 中文 \
    --output-jsonl realtime_ws_8c.jsonl
```

Use a representative audio file. A long, pauseless monologue creates a very
different load shape from turn-taking meetings, because nearly every client is
speaking and triggering partial decodes at the same time.

For an unpaced stress test, add `--no-pace`. Treat that result as a throughput
stress signal, not as user-facing realtime latency.

## Metrics

| Metric | Meaning |
|--------|---------|
| `aggregate_audio_per_wall` | Total input audio seconds across all clients divided by benchmark wall time |
| `first_update_ms_p50/p95` | Time from first audio frame to first result message with `sentences`, `partial`, or `is_final` |
| `final_after_stop_ms_p50/p95` | Time from sending `STOP` to receiving the final result |
| `client_response_lag_ms_p95_max` | Largest per-client p95 of `(client receive time - audio start) - server duration_ms`; useful mainly in paced mode |
| `partial_messages` | Count of non-final result messages with a non-empty `partial` |
| `final_messages` | Count of final result messages |
| `errors` | Connection, timeout, protocol, or client-side validation errors |

The script can observe only client-side timing and fields returned by the
server. If you are debugging service internals, collect server logs separately
for queue wait, VAD time, ASR decode time, speaker diarization time, GPU memory,
and GPU utilization.

## Report Template

When publishing a realtime WebSocket benchmark or issue report, include:

| Category | What to record |
|----------|----------------|
| Data | Audio duration, sample rate, language/domain, silence ratio or speaking pattern, and whether the same file was looped |
| Load | `--clients`, `--loops`, `--chunk-ms`, paced or `--no-pace`, and total benchmark wall time |
| Service | `serve_realtime_ws.py` command, `--partial-window-sec`, `--decode-interval`, `--enable-spk`, language, and hotwords |
| Hardware | GPU/NPU model, GPU count, memory, driver, CUDA/CANN/runtime versions, CPU model, and available RAM |
| Software | `funasr`, PyTorch, torchaudio, vLLM, Python, OS, and container image if any |
| Output | Summary line, JSONL artifact, server logs, and any failed client IDs |

Do not reuse an offline `RTFx` number as a concurrency claim. For realtime
service sizing, benchmark with the actual traffic shape, especially sentence
length, pause distribution, simultaneous speakers, and whether speaker
diarization is enabled.
