# Fun-ASR-Nano Streaming vLLM Inference

Chunk-by-chunk streaming ASR using vLLM for high-throughput LLM decoding.

## Design

```
Audio stream (720ms chunks)
    │
    ▼ (cumulative re-encoding)
┌────────────────────┐
│ Encoder + Adaptor  │  (PyTorch, single GPU)
└────────┬───────────┘
         ▼
┌────────────────────┐
│  vLLM Batch Gen    │  (all chunks batched in single call)
└────────┬───────────┘
         ▼
Per-chunk results with fixed/unfixed regions:
  chunk1: ""
  chunk2: "开"
  chunk3: "开放时间早"
  chunk4: "开放时间早上九点至下午五点。"  ← stabilized
```

### Key Design Decisions

1. **Cumulative audio re-encoding**: Each chunk re-processes ALL audio from start. This ensures the encoder has full context.

2. **Batch-all-chunks**: All chunk prompts are batched into a single vLLM `generate()` call. This avoids KV-cache corruption issues with sequential EmbedsPrompt calls and maximizes throughput.

3. **8-character rollback**: The last 8 characters of each chunk's output are "unfixed" (may change in subsequent chunks). Text beyond the rollback boundary is "fixed".

4. **Repetition penalty**: `repetition_penalty=1.3` prevents degenerate repetitions for short audio chunks.

## Quick Start

```python
from funasr.models.fun_asr_nano.inference_vllm_streaming import FunASRNanoStreamingVLLM

engine = FunASRNanoStreamingVLLM.from_pretrained(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    tensor_parallel_size=1,
    chunk_ms=720,
    rollback_chars=8,
)

# Stream results
for result in engine.streaming_generate("audio.wav", language="中文"):
    if result["is_final"]:
        print(f"Final: {result['text']}")
    else:
        print(f"[{result['audio_duration_ms']:.0f}ms] {result['text']}")
        print(f"  Fixed: {result['fixed_text']}")
```

## API Reference

### `FunASRNanoStreamingVLLM.from_pretrained()`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"FunAudioLLM/Fun-ASR-Nano-2512"` | Model name or local path |
| `hub` | `"ms"` | `"ms"` or `"hf"` |
| `device` | `"cuda:0"` | Device for audio encoder/adaptor |
| `dtype` | `"bf16"` | Compute precision |
| `tensor_parallel_size` | `1` | GPUs for vLLM |
| `gpu_memory_utilization` | `0.8` | GPU memory fraction for vLLM KV cache |
| `max_model_len` | `2048` | Maximum sequence length |
| `chunk_ms` | `720` | Chunk duration in ms |
| `rollback_chars` | `8` | Unfixed region size |

### `engine.streaming_generate()`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `audio_input` | - | File path, numpy array, or tensor (16kHz) |
| `chunk_ms` | `720` | Override chunk size |
| `rollback_chars` | `8` | Override rollback |
| `hotwords` | `None` | List of hotwords |
| `language` | `None` | Language hint (e.g. "中文") |
| `itn` | `True` | Inverse text normalization |
| `max_new_tokens` | `200` | Max tokens per chunk |
| `temperature` | `0.0` | Sampling temperature |

> Note: `repetition_penalty=1.3` is applied internally to prevent degenerate repetitions on short chunks.

**Yields** per chunk:
```python
{
    "text": "full recognized text so far",
    "fixed_text": "confirmed text (won't change)",
    "is_final": False,  # True for last chunk
    "chunk_idx": 4,
    "audio_duration_ms": 2880.0,
}
```

## Output Characteristics

| Audio Duration | Output Quality |
|---------------|---------------|
| < 1.5s | Empty or garbage (insufficient context) |
| 1.5 - 3.0s | Partial/noisy (model warming up) |
| > 3.0s | Correct transcription |
| Full audio | Best accuracy |

The model needs ~3 seconds of cumulative audio before producing accurate results. This is inherent to the model architecture, not a vLLM limitation.

## Example Output

```
[c01]   720ms | ""
[c02]  1440ms | "－"
[c03]  2160ms | "time/day."
[c04]  2880ms | "time早上九"
[c05]  3600ms | "期限：早上九点至"
         fixed: "期限：早上"
[c06]  4320ms | "期限：开放时间早上9点~下午10:35/4."
         fixed: "期限：开放时间早上9点~下午10:"
[c07]  5040ms | "期限：开放时间早上九点至下午五点。"
         fixed: "期限：开放时间早上九点至下午"
[FINAL]  5616ms | "期限，开放时间早上九点至下午五点。"
```

## Compared with Other Modes

| Mode | Class | Latency | Accuracy | Use Case |
|------|-------|---------|----------|----------|
| Offline | `AutoModelVLLM` / `FunASRNanoVLLM` | Full audio | Best | Batch transcription |
| Streaming SDK | `FunASRNanoStreamingVLLM` | Per-chunk | Good (after ~3s) | Real-time display (SDK) |
| WebSocket Server | `serve_realtime_ws.py` | Per-VAD-segment | Best | Production service with VAD+SPK |

For offline batch inference:

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

model = AutoModelVLLM(model="FunAudioLLM/Fun-ASR-Nano-2512")
results = model.generate(["audio.wav"], language="中文")
```

For real-time WebSocket service (VAD + ASR + Speaker Diarization):

```bash
python serve_realtime_ws.py --port 10095 --language 中文
```

> Note: `serve_realtime_ws.py` uses `AutoModelVLLM.generate()` per-segment (not `streaming_generate`), because VAD already provides natural sentence boundaries. This gives better accuracy than chunk-level streaming.

## Related

- **[Offline vLLM Inference](vllm_inference.md)** — Batch inference with maximum accuracy
- **[Real-time WebSocket Server](realtime_demo.md)** — Production-ready streaming service with VAD + Speaker Diarization + Hotwords (uses vLLM as backend via `AutoModelVLLM`)
