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
    --partial-window-sec 8 --decode-interval 0.8 \
    --vad-device cpu --vad-ncpu 1 \
    --decode-batch-wait-ms 10 --decode-max-batch-size 16 \
    --log-decode-profile
```

Speaker diarization is disabled by default. Add `--enable-spk` only when the
`spk` field is required, and report that setting with the benchmark result.
Compatible cross-session decodes arriving within `--decode-batch-wait-ms` are
submitted as one engine batch. Set the wait to `0` for a no-wait comparison,
and keep all batching flags identical when comparing releases.

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
    --client-ping-interval 20 \
    --client-ping-timeout 0 \
    --language 中文 \
    --output-jsonl realtime_ws_8c.jsonl
```

Values `<=0` disable the corresponding client ping setting. Record both client
settings when comparing disconnects. The `websockets` library's `max_queue`
setting bounds receive buffering for incoming messages; it doesn't change
ping/pong timeout semantics.

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
| `client_response_lag_ms_p95_max` | Largest per-client p95 of non-final `(client receive time - audio start) - server duration_ms`; useful mainly in paced mode for preview/partial lag |
| `partial_messages` | Count of non-final result messages with a non-empty `partial` |
| `final_messages` | Count of final result messages |
| `errors` | Connection, timeout, protocol, or client-side validation errors |

The script can observe only client-side timing and fields returned by the
server. For a performance investigation, add `--log-decode-profile` to record
one structured line per engine call with the request and sample counts, audio
duration range, queue-wait p50/max, and total engine latency. The underlying
Fun-ASR-Nano vLLM path also logs audio-encoder and vLLM-generation time. Collect
those server logs together with GPU memory/utilization and the client JSONL.

When comparing releases, align `partial_messages` as well as audio, clients,
and service flags. A server that blocks its WebSocket event loop can appear to
finish sooner simply because it processes fewer provisional decodes; that is
not an engine-throughput improvement and gives users fewer live updates.

## Concurrency Regression Reference

The following result compares the `v1.4.3` service with the concurrent decode
batching defaults introduced after it. Each service used one H100 80 GB GPU,
vLLM 0.19.1, PyTorch 2.10.0 with CUDA 12.8, server-side FSMN VAD, and speaker
diarization disabled. The candidate used CPU VAD with one thread per session,
the 10 ms decode batch wait, and a maximum decode batch size of 16.

The workload was a 47-second looped Chinese recording sent in paced 100 ms
frames. All clients replayed the same file once and started together.

| Clients | Version | Wall time | Aggregate audio/wall | First update p50/p95 | Final after STOP p50/p95 | Response lag p95 max | Errors |
|---------|---------|-----------|----------------------|----------------------|--------------------------|-----------------------|--------|
| 12 | `v1.4.3` | 66.897 s | 8.431x | 462.0 / 462.1 ms | 19,550.3 / 19,832.9 ms | 17,564.7 ms | 0 |
| 12 | batched candidate | 48.765 s | 11.566x | 484.9 / 488.0 ms | 414.4 / 414.9 ms | 1,047.0 ms | 0 |
| 16 | `v1.4.3` | 87.898 s | 8.555x | 483.4 / 483.5 ms | 40,463.0 / 40,823.3 ms | 36,786.7 ms | 0 |
| 16 | batched candidate | 57.085 s | 13.173x | 515.9 / 525.7 ms | 9,757.6 / 10,051.6 ms | 10,231.4 ms | 0 |

This is a regression reference, not a universal capacity claim. Repeat the
test with production audio and service options before choosing a concurrency
limit. In particular, long speech segments create synchronized, expensive
final decodes that are not representative of every meeting or voice-agent
workload.

## Report Template

When publishing a realtime WebSocket benchmark or issue report, include:

| Category | What to record |
|----------|----------------|
| Data | Audio duration, sample rate, language/domain, silence ratio or speaking pattern, and whether the same file was looped |
| Load | `--clients`, `--loops`, `--chunk-ms`, paced or `--no-pace`, client ping interval/timeout, and total benchmark wall time |
| Service | `serve_realtime_ws.py` command, WebSocket ping interval/timeout, `--partial-window-sec`, `--decode-interval`, `--vad-device`, `--vad-ncpu`, `--decode-batch-wait-ms`, `--decode-max-batch-size`, `--log-decode-profile`, `--enable-spk`, language, and hotwords |
| Hardware | GPU/NPU model, GPU count, memory, driver, CUDA/CANN/runtime versions, CPU model, and available RAM |
| Software | `funasr`, PyTorch, torchaudio, vLLM, Python, OS, and container image if any |
| Output | Summary line, JSONL artifact, server logs, and any failed client IDs |

Do not reuse an offline `RTFx` number as a concurrency claim. For realtime
service sizing, benchmark with the actual traffic shape, especially sentence
length, pause distribution, simultaneous speakers, and whether speaker
diarization is enabled.
