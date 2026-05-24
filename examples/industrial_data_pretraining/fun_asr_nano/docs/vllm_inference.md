# Fun-ASR-Nano vLLM Inference Guide

Fun-ASR-Nano supports [vLLM](https://github.com/vllm-project/vllm) as a high-throughput inference backend. This enables significantly faster batch processing and multi-GPU tensor parallelism for production deployments.

## Overview

The vLLM integration splits inference into two parts:
- **Audio processing** (PyTorch): Frontend → Audio Encoder → Audio Adaptor
- **LLM decoding** (vLLM): High-throughput autoregressive text generation with KV-cache optimization

This architecture allows vLLM to efficiently batch multiple requests and leverage tensor parallelism across GPUs for the compute-intensive LLM decoding step.

## Requirements

```bash
# Core dependencies
pip install funasr>=1.3.0
pip install vllm>=0.12.0
pip install safetensors
pip install tiktoken

# For ModelScope model download
pip install modelscope
```

## Quick Start

### AutoModelVLLM (Recommended)

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

# Generic interface - auto-detects model type, works for all LLM-based ASR models
model = AutoModelVLLM(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    hub="ms",
    tensor_parallel_size=1,
)

results = model.generate(["audio.wav"], language="中文")
print(results[0]["text"])
```

### FunASRNanoVLLM (Direct API)

```python
from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM

# Initialize with single GPU
engine = FunASRNanoVLLM.from_pretrained(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    hub="ms",  # or "hf" for HuggingFace
    tensor_parallel_size=1,
)

# Single file inference
results = engine.generate(["path/to/audio.wav"])
print(results[0]["text"])

# Batch inference with hotwords
results = engine.generate(
    inputs=["audio1.wav", "audio2.wav", "audio3.wav"],
    hotwords=["开放时间", "人工智能"],
    language="中文",
)
for r in results:
    print(f"[{r['key']}] {r['text']}")
```

### Multi-GPU Tensor Parallel

```python
# Use 2 GPUs for vLLM (doubles decoding throughput)
engine = FunASRNanoVLLM.from_pretrained(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.8,
)
```

### Command-Line Demo

```bash
# Single file
python demo_vllm.py --input audio.wav --language 中文

# Batch processing from wav.scp
python demo_vllm.py --input wav.scp --tensor-parallel-size 4 --batch-size 32

# With hotwords
python demo_vllm.py --input audio.wav --hotwords 开放时间 周一 --language 中文

# Save results to file
python demo_vllm.py --input wav.scp --output results.jsonl
```

## API Reference

### `FunASRNanoVLLM.from_pretrained()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"FunAudioLLM/Fun-ASR-Nano-2512"` | Model name or local path |
| `hub` | str | `"ms"` | Model hub: `"ms"` (ModelScope) or `"hf"` (HuggingFace) |
| `device` | str | `"cuda:0"` | Device for audio encoder/adaptor |
| `dtype` | str | `"bf16"` | Compute dtype: `"bf16"`, `"fp16"`, `"fp32"` |
| `tensor_parallel_size` | int | `1` | Number of GPUs for vLLM tensor parallel |
| `gpu_memory_utilization` | float | `0.8` | Fraction of GPU memory for vLLM KV cache |
| `max_model_len` | int | `2048` | Maximum sequence length |

### `engine.generate()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | str/List[str] | - | Audio file path(s) or numpy arrays |
| `hotwords` | List[str] | `None` | Keywords to boost recognition |
| `language` | str | `None` | Language hint (e.g. "中文", "英文", "日文") |
| `itn` | bool | `True` | Apply inverse text normalization |
| `max_new_tokens` | int | `512` | Max generated tokens per sample |
| `temperature` | float | `0.0` | Sampling temperature (0 = greedy) |
| `top_p` | float | `1.0` | Nucleus sampling |
| `repetition_penalty` | float | `1.0` | Repetition penalty |

**Returns:** `List[dict]` with keys:
- `"key"`: Sample identifier (filename without extension)
- `"text"`: Recognized text
- `"timestamps"`: Character-level timestamps (if CTC decoder available)

## Performance

### Throughput Comparison

| Backend | Batch Size | 1 GPU | 2 GPUs | 4 GPUs |
|---------|-----------|-------|--------|--------|
| HuggingFace (baseline) | 1 | 1x | - | - |
| vLLM | 1 | ~1.5x | ~2x | ~2.5x |
| vLLM | 16 | ~4x | ~7x | ~12x |
| vLLM | 32 | ~5x | ~9x | ~15x |

*Approximate speedup for batch processing. Actual numbers depend on audio length and hardware.*

### Memory Usage

- Audio encoder + adaptor: ~400MB GPU memory (single GPU)
- vLLM LLM (Qwen3-0.6B): ~1.2GB per GPU (with bf16)
- KV cache: scales with `gpu_memory_utilization` setting

## Architecture Details

```
┌─────────────────────────────────────────────────────┐
│                  FunASRNanoVLLM                       │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Audio Input                                         │
│      │                                               │
│      ▼                                               │
│  ┌──────────────┐                                    │
│  │  WavFrontend │  (fbank, LFR, CMVN)               │
│  └──────┬───────┘                                    │
│         ▼                                            │
│  ┌──────────────────────┐                            │
│  │ SenseVoiceEncoder    │  (50-layer SANM, frozen)  │
│  │ (Audio Encoder)      │                            │
│  └──────┬───────────────┘                            │
│         ▼                                            │
│  ┌──────────────────────┐                            │
│  │ Transformer Adaptor  │  (2-layer, downsample)    │
│  │ (Audio Adaptor)      │                            │
│  └──────┬───────────────┘                            │
│         ▼                                            │
│  [Audio Embeddings]  ──────────┐                     │
│                                ▼                     │
│  [Text Token Embeds] ───> [Concat] ─── EmbedsPrompt │
│  (prefix + suffix)              │                    │
│                                 ▼                    │
│                    ┌────────────────────┐            │
│                    │   vLLM Engine      │            │
│                    │   (Qwen3-0.6B)    │            │
│                    │   Tensor Parallel  │            │
│                    └────────┬───────────┘            │
│                             ▼                        │
│                    Generated Text                     │
│                                                      │
│  (Optional) CTC Forced Alignment → Timestamps       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## Supported Models

| Model | Hub ID | Languages |
|-------|--------|-----------|
| Fun-ASR-Nano | `FunAudioLLM/Fun-ASR-Nano-2512` | Chinese, English, Japanese + dialects |
| Fun-ASR-MLT-Nano | `FunAudioLLM/Fun-ASR-MLT-Nano-2512` | 31 languages |

## Troubleshooting

### "model.pt not found"
The model must be fully downloaded first. On first use, `from_pretrained()` will download it automatically from the hub.

### CUDA out of memory
- Reduce `gpu_memory_utilization` (e.g. 0.6)
- Use `tensor_parallel_size` > 1 to split across GPUs
- Reduce `max_model_len`

### Slow first inference
The first call includes vLLM's CUDA graph warmup. Subsequent calls will be faster.

### Tokenizer warnings
These are typically harmless warnings from the Qwen3 tokenizer about special tokens and can be ignored.

## Related

- **[Streaming vLLM Inference](vllm_streaming.md)** — Chunk-by-chunk streaming (SDK level)
- **[Real-time WebSocket Server](realtime_demo.md)** — Production-ready streaming service with VAD + Speaker Diarization + Hotwords (uses vLLM as backend)
