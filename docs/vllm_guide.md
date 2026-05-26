# FunASR vLLM Inference Engine Guide

FunASR integrates the vLLM high-throughput inference engine to accelerate autoregressive decoding of LLM-based ASR models. Three modes are supported: offline SDK inference, streaming SDK inference, and production-grade WebSocket real-time service.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation & Environment](#2-installation--environment)
3. [Offline SDK Inference](#3-offline-sdk-inference)
4. [Streaming SDK Inference](#4-streaming-sdk-inference)
5. [Real-time WebSocket Service](#5-real-time-websocket-service)
6. [Performance Comparison](#6-performance-comparison)
7. [API Reference](#7-api-reference)
8. [FAQ](#8-faq)

---

## 1. Overview

### Supported Models

| Model | vLLM Support | Notes |
|-------|-------------|-------|
| **FunASRNano** | ✓ | Qwen3-0.6B LLM, supports offline and streaming |
| **LLMASR** | ✓ | Whisper + Qwen/Vicuna/LLaMA |
| **GLMASR** | ✓ | GLM-ASR-Nano |
| **QwenAudioWarp** | ✓ | Qwen-Audio |
| Paraformer | ✗ | Non-autoregressive model (CIF predictor), no LLM decoding |
| SenseVoice | ✗ | Whisper-like encoder-decoder, not LLM |
| Conformer/Transformer | ✗ | CTC/attention decoding, not LLM |

### Architecture

```
┌────────────────────────────────────────────────────────┐
│                   FunASR + vLLM                         │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Audio ──→ Frontend ──→ Audio Encoder ──→ Adaptor      │
│            (fbank)      (SenseVoice)     (Transformer) │
│                                │                       │
│                                ▼                       │
│                       Audio Embeddings                  │
│                                │                       │
│  Text Prompt ──→ Tokenize ──→ Embed ──→ [Concat]      │
│  (hotwords/language/itn)                    │          │
│                                             ▼          │
│                                    ┌──────────────┐    │
│                                    │  vLLM Engine │    │
│                                    │  (Qwen3 etc) │    │
│                                    │  TP parallel │    │
│                                    └──────┬───────┘    │
│                                           ▼            │
│                                    Generated Text      │
│                                                        │
│  (Optional) CTC Forced Alignment ──→ Character-level   │
│                                      timestamps        │
└────────────────────────────────────────────────────────┘
```

### Three Usage Modes

| Mode | Entry Point | Use Case |
|------|-------------|----------|
| [Offline SDK Inference](#3-offline-sdk-inference) | `AutoModelVLLM` / `FunASRNanoVLLM` | Large-scale transcription, batch processing |
| [Streaming SDK Inference](#4-streaming-sdk-inference) | `FunASRNanoStreamingVLLM` | Real-time subtitles (SDK integration) |
| [WebSocket Real-time Service](#5-real-time-websocket-service) | `serve_realtime_ws.py` | Production deployment, multi-client access |

---

## 2. Installation & Environment

### Dependencies

```bash
pip install funasr>=1.3.0
pip install vllm>=0.12.0
pip install safetensors tiktoken websockets regex

# FunASR development mode install (requires auto_model_vllm)
cd /path/to/FunASR && pip install -e .
```

### Hardware Requirements

| Configuration | Minimum | Recommended |
|--------------|---------|-------------|
| GPU Memory | 8GB | 16GB+ |
| CUDA | 11.8 | 12.0+ |
| GPU Count | 1 | 2+ (tensor parallel) |

---

## 3. Offline SDK Inference

Best for large-scale audio transcription and offline batch processing. vLLM's batching capability provides the greatest advantage in this scenario.

### Design Principles

The offline SDK inference splits the ASR pipeline into two stages executed independently:

```
┌─────────────────────────────────────────────────────────────────────┐
│           Stage 1: Audio Encoding (PyTorch, single GPU)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Audio file list ──→ Group (batch=8) ──→ Frontend (Fbank)           │
│       │                                      │                      │
│       │                                      ▼                      │
│       │                             SenseVoice Encoder              │
│       │                                      │                      │
│       │                                      ▼                      │
│       │                             Audio Adaptor                   │
│       │                             (dim transform + LFR truncation)│
│       │                                      │                      │
│       └─── Shared text prompt encoding ─┐    ▼                      │
│            (system/hotwords/language)    │ audio_embeds              │
│                     │                   │    │                       │
│                     ▼                   │    ▼                       │
│                prefix_emb ──→ [concat: prefix | audio | suffix]     │
│                                              │                      │
│                                              ▼                      │
│                                    EmbedsPrompt (N samples)          │
└──────────────────────────────────────────────┼─────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│         Stage 2: LLM Decoding (vLLM, multi-GPU Tensor Parallel)     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  EmbedsPrompt × N ──→ vLLM Continuous Batching                      │
│                        (PagedAttention + CUDA Graph)                 │
│                              │                                      │
│                              ▼                                      │
│                     Generated token_ids × N                          │
│                              │                                      │
│                              ▼                                      │
│                     Decode + post-processing (clean special tokens)  │
│                              │                                      │
│                              ▼                                      │
│                     (Optional) CTC Forced Alignment → char timestamps│
└─────────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

1. **Weight separation**: On first run, extracts `llm.*` prefixed weights from `model.pt` and saves them in HuggingFace safetensors format for vLLM (cached in `Qwen3-0.6B-vllm/` directory)
2. **Embedding concatenation**: Text prompt is encoded via the LLM's `embed_tokens` layer, then concatenated with audio adaptor output along the sequence dimension: `[prefix_emb | audio_emb | suffix_emb]`, submitted to vLLM as `EmbedsPrompt`
3. **Low Frame Rate truncation**: Adaptor output is truncated to the correct length using: `fake_token_len = ((((fbank_len - 3 + 2) // 2 - 3 + 2) // 2) - 1) // 2 + 1`, ensuring consistency with PyTorch training
4. **Batch audio encoding**: Multiple audio files are grouped (batch_size=8) through encoder + adaptor forward pass, reducing GPU kernel launch overhead
5. **Shared text prompt**: When hotwords/language are the same within a batch, prefix_emb and suffix_emb are computed only once
6. **CTC timestamps**: Encoder output is preserved; after LLM generates text, forced alignment produces character-level timestamps

**Why faster than PyTorch generate()?**

| Dimension | PyTorch | vLLM |
|-----------|---------|------|
| KV Cache | Fixed pre-allocation (wastes memory) | PagedAttention on-demand allocation |
| Batching | Manual padding required | Continuous Batching auto-scheduling |
| CUDA | Sequential per-sample | CUDA Graph + operator fusion |
| Multi-GPU | Manual implementation | Tensor Parallel one-line config |
| Result | RTFx ~20 | **RTFx 340+** (16x speedup) |

### Universal Interface (Recommended)

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

model = AutoModelVLLM(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    hub="ms",                    # or "hf"
    tensor_parallel_size=2,      # multi-GPU parallel
    gpu_memory_utilization=0.8,
)

results = model.generate(
    ["audio1.wav", "audio2.wav"],
    language="中文",
    hotwords=["张三", "北京"],
)
for r in results:
    print(f"[{r['key']}] {r['text']}")
```

### Direct Interface

```python
from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM

engine = FunASRNanoVLLM.from_pretrained(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    tensor_parallel_size=4,
)

results = engine.generate(
    inputs="wav.scp",  # supports scp/jsonl/file list
    hotwords=["开放时间"],
    language="中文",
    max_new_tokens=512,
)
```

### Command Line

```bash
cd examples/industrial_data_pretraining/fun_asr_nano

# Single file
python demo_vllm.py --input audio.wav --language 中文

# Batch + multi-GPU
python demo_vllm.py --input wav.scp --tensor-parallel-size 4 --batch-size 32

# With hotwords + save results
python demo_vllm.py --input audio.wav --hotwords 张三 北京 --output results.jsonl
```

---

## 4. Streaming SDK Inference

Processes audio in 720ms chunks incrementally, outputting progressively stable recognition results. Suitable for SDK-integrated real-time subtitle scenarios.

### Design Principle

```
Audio stream (720ms chunks)
    │ Cumulative re-encoding (each chunk contains all audio from start to current)
    ▼
┌──────────────────────────┐
│ Stage 1: First 10 chunks │  ← No prev_text, batch generation
│ Find stable output       │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ Stage 2: Subsequent      │  ← Use stable output as prev_text
└──────────┬───────────────┘
           ▼
Each chunk: [fixed region (confirmed)] + [8-char unfixed (may change)]
```

### Usage

```python
from funasr.models.fun_asr_nano.inference_vllm_streaming import FunASRNanoStreamingVLLM

engine = FunASRNanoStreamingVLLM.from_pretrained(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    chunk_ms=720,
    rollback_chars=8,
)

for result in engine.streaming_generate("audio.wav", language="中文"):
    if result["is_final"]:
        print(f"Final: {result['text']}")
    else:
        print(f"[{result['audio_duration_ms']:.0f}ms] Confirmed: {result['fixed_text']}")
```

### Output Characteristics

| Accumulated Audio | Output Quality |
|-------------------|---------------|
| < 1.5s | Empty or noise |
| 1.5-3.0s | Partially correct |
| > 3.0s | Accurate output |

> Note: `repetition_penalty=1.3` is hardcoded internally to prevent short-chunk repetition degradation.

---

## 5. Real-time WebSocket Service

Production-grade real-time speech recognition service integrating VAD segmentation + vLLM inference + speaker diarization + hotwords.

### Inference Logic

**Core design: VAD endpoint-based segment decoding (not fixed chunks)**

```
Audio stream ──→ StreamingVAD (60ms) ──→ Endpoint detected ──→ vLLM decode segment
                                    │                              │
                                    │                              ▼
                                    │                     locked_sentences (locked)
                                    │                              │
                                    │                              ▼
                                    │                     SPK assign (speaker assignment)
                                    │
                                    └──→ Not ended ──→ vLLM partial decode (every 0.48s)
                                                              │
                                                              ▼
                                                       partial text (preview, will be overwritten)
```

**Two inference paths:**

| Path | Trigger Condition | Output |
|------|-------------------|--------|
| Confirmed segment decode | VAD detects silence endpoint | Locked to sentences, never changes |
| Partial preview | Every 0.48s + new audio ≥ 960ms | Temporary text, overwritten anytime |

**Dynamic VAD thresholds:**

| Accumulated Duration | Silence Threshold | Effect |
|---------------------|-------------------|--------|
| ≤ 5s | 2.0s | Preserve short sentences |
| 5-10s | 1.5s | Normal segmentation |
| 10-15s | 1.0s | Start tightening |
| 15-30s | 0.8s | Faster splitting |
| 30-45s | 0.4s | Prevent overly long segments |
| > 45s | 0.1s | Force split |

**STOP final processing:**
1. Feed remaining audio to VAD (is_final=True)
2. Force-end currently speaking segment
3. Global SPK re-clustering (correct speaker IDs)
4. Return `is_final: true`

### Deployment

```bash
cd examples/industrial_data_pretraining/fun_asr_nano

# Single GPU
CUDA_VISIBLE_DEVICES=0 python serve_realtime_ws.py --port 10095 --language 中文

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1 python serve_realtime_ws.py \
    --port 10095 --tensor-parallel-size 2 --language 中文

# Full parameters
python serve_realtime_ws.py \
    --port 10095 \
    --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --hub ms \
    --device cuda:0 \
    --decode-interval 0.48 \
    --hotword-file hotword_list \
    --language 中文 \
    --dtype bf16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 2048
```

### Clients

| Client | Usage |
|--------|-------|
| Browser | Open `client_mic.html` (microphone/file/hotwords/speakers) |
| Python CLI | `python client_python.py --server ws://localhost:10095 --mic` |
| Test script | `python client_test.py --server ws://localhost:10095 --file audio.wav` |

Remote access requires SSH port forwarding: `ssh -L 10095:localhost:10095 <server>`

### WebSocket Protocol

**Client → Server:**

| Message | Format | Description |
|---------|--------|-------------|
| Start session | `"START"` | Initialize session |
| Set hotwords | `"HOTWORDS:word1,word2"` | Optional |
| Set language | `"LANGUAGE:中文"` | Optional |
| Audio data | `bytes` | PCM16, 16kHz, mono |
| End session | `"STOP"` | Trigger final decoding |

**Server → Client:**

```json
// Events
{"event": "started"}
{"event": "hotwords_set", "hotwords": ["word1", "word2"]}
{"event": "language_set", "language": "中文"}
{"event": "stopped"}

// Real-time results
{
    "sentences": [{"text": "confirmed text", "start": 1700, "end": 5500, "spk": 0}],
    "partial": "currently speaking...",
    "partial_start_ms": 5800,
    "duration_ms": 7200,
    "is_final": false
}

// Final result
{
    "sentences": [{"text": "...", "start": ..., "end": ..., "spk": ...}, ...],
    "partial": "",
    "partial_start_ms": 0,
    "duration_ms": 10000,
    "is_final": true
}
```

**Interaction sequence:**

```
Client                          Server
  │── "START" ─────────────────→│
  │←─ {"event":"started"} ──────│
  │── "HOTWORDS:张三,北京" ────→│
  │←─ {"event":"hotwords_set"} ─│
  │── "LANGUAGE:中文" ─────────→│
  │←─ {"event":"language_set"} ─│
  │── [audio bytes] ───────────→│
  │←─ {sentences,partial} ──────│  (real-time push)
  │── [audio bytes] ───────────→│
  │←─ {sentences,partial} ──────│
  │── "STOP" ──────────────────→│
  │←─ {sentences,is_final:true} │  (final result)
  │←─ {"event":"stopped"} ──────│
```

### Hotword File

Default filename `hotword_list` (specified via `--hotword-file`), one word per line:

```
张三
李四
北京大学
```

Can also be set dynamically via WebSocket: `HOTWORDS:word1,word2,word3`

---

## 6. Performance Comparison

### Offline Inference

| Configuration | 5.6s Audio Latency | Relative Speedup |
|--------------|-------------------|-----------------|
| PyTorch (baseline) | 0.89s | 1x |
| vLLM 1-GPU | 0.30s | **3x** |
| vLLM 2-GPU TP | ~0.20s | **4.5x** |

### Batch Throughput

| Batch Size | 1-GPU | 2-GPU | 4-GPU |
|-----------|-------|-------|-------|
| 1 | ~1.5x | ~2x | ~2.5x |
| 16 | ~4x | ~7x | ~12x |
| 32 | ~5x | ~9x | ~15x |

### WebSocket Real-time Service

| Metric | Value |
|--------|-------|
| RTF | < 0.08 |
| First-word latency | ~480ms |
| Total time for 30s audio | ~2.3s |
| Concurrency | Multiple WebSocket connections |

### VAD + vLLM Pipeline

| Scenario | PyTorch Sequential | vLLM Batch | Speedup |
|----------|-------------------|------------|---------|
| 10 segments × 5s | ~9s | ~1.5s | **6x** |
| 20 segments × 5s | ~18s | ~2.5s | **7x** |

---

## 7. API Reference

### AutoModelVLLM

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM
```

Universal entry point that automatically detects model type and selects the corresponding vLLM implementation.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | - | Model name (hub) or local path |
| `hub` | `"ms"` | `"ms"` / `"hf"` |
| `device` | `"cuda:0"` | Audio encoder device |
| `dtype` | `"bf16"` | Precision |
| `tensor_parallel_size` | `1` | vLLM GPU parallel count |
| `gpu_memory_utilization` | `0.8` | KV Cache memory ratio |
| `max_model_len` | `4096` | Maximum sequence length |

### AutoModelVLLM.generate()

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inputs` | - | Audio path/list/numpy/tensor |
| `language` | `None` | Language hint |
| `hotwords` | `None` | Hotword list |
| `itn` | `True` | Inverse text normalization |
| `max_new_tokens` | `512` | Maximum generated tokens |
| `temperature` | `0.0` | Sampling temperature |
| `repetition_penalty` | `1.0` | Repetition penalty |

**Returns**: `[{"key": str, "text": str, "timestamps": [...]}]`

### FunASRNanoStreamingVLLM.from_pretrained()

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"FunAudioLLM/Fun-ASR-Nano-2512"` | Model name or path |
| `hub` | `"ms"` | Model source |
| `device` | `"cuda:0"` | Device |
| `dtype` | `"bf16"` | Precision |
| `tensor_parallel_size` | `1` | GPU parallel count |
| `gpu_memory_utilization` | `0.8` | Memory ratio |
| `max_model_len` | `2048` | Sequence length |
| `chunk_ms` | `720` | Chunk duration |
| `rollback_chars` | `8` | Rollback character count |

### streaming_generate()

| Parameter | Default | Description |
|-----------|---------|-------------|
| `audio_input` | - | Audio path/data |
| `chunk_ms` | `720` | Chunk size |
| `rollback_chars` | `8` | Rollback characters |
| `hotwords` | `None` | Hotwords |
| `language` | `None` | Language |
| `max_new_tokens` | `200` | Max tokens per chunk |
| `temperature` | `0.0` | Sampling temperature |

**Yields**: `{"text", "fixed_text", "is_final", "chunk_idx", "audio_duration_ms"}`

> `repetition_penalty=1.3` is hardcoded internally.

### serve_realtime_ws.py Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--port` | `10095` | WebSocket port |
| `--model` | `FunAudioLLM/Fun-ASR-Nano-2512` | Model |
| `--hub` | `ms` | Source |
| `--device` | `cuda:0` | Device |
| `--decode-interval` | `0.48` | Partial decode interval (seconds) |
| `--hotword-file` | `hotword_list` | Hotword file |
| `--language` | `None` | Language |
| `--dtype` | `bf16` | Precision |
| `--tensor-parallel-size` | `1` | GPU parallel |
| `--gpu-memory-utilization` | `0.8` | Memory ratio |
| `--max-model-len` | `2048` | Sequence length |

---

## 8. FAQ

### Q: Why is the first startup slow?
vLLM needs to initialize KV Cache and perform CUDA Graph warmup, taking approximately 60-90 seconds. Subsequent inferences respond instantly.

### Q: How to handle CUDA OOM?
- Reduce `gpu_memory_utilization` (e.g., 0.6)
- Increase `tensor_parallel_size` to distribute across multiple GPUs
- Reduce `max_model_len`

### Q: Can Paraformer use vLLM?
No. Paraformer is a non-autoregressive model (CIF predictor) where all tokens are generated in parallel without KV-cache. vLLM only accelerates autoregressive LLM decoding.

### Q: What's the difference between WebSocket service and streaming_generate?

| | WebSocket Service | streaming_generate |
|---|---|---|
| Segmentation | VAD natural endpoints | Fixed 720ms chunks |
| Inference | Decode entire VAD segment | Cumulative re-encode all audio |
| Accuracy | Higher | Lower for first 3s |
| Use case | Production deployment | SDK integration |

### Q: Why is output empty for the first few seconds of streaming?
Normal. The model needs ~3 seconds of accumulated audio to produce meaningful output. This is a model characteristic, not a vLLM limitation.

### Q: What audio formats are supported?
wav, mp3, flac and other mainstream formats. Sample rate is automatically converted to 16kHz.

### Q: Browser cannot access microphone?
Chrome requires HTTPS or localhost. For remote servers, use SSH port forwarding: `ssh -L 10095:localhost:10095 <server>`

### Q: Will multiple concurrent connections interfere with each other?
No. Each WebSocket connection has an independent session (VAD/ASR state isolation). vLLM handles scheduling internally.

---

## Appendix: DynamicStreamingVAD

`funasr.models.fsmn_vad_streaming.dynamic_vad.DynamicStreamingVAD` is a generic dynamic-threshold streaming VAD wrapper that adjusts silence splitting thresholds dynamically based on the accumulated duration of the current speech segment, built on top of fsmn-vad.

### Design Motivation

fsmn-vad uses a fixed silence threshold (800ms) by default. In practice:
- Short utterances (e.g., "okay") need longer silence before splitting, otherwise sentences get fragmented
- Long segments (e.g., 30s+ meeting speeches) need faster splitting, otherwise ASR input becomes too long and quality degrades

### Usage

```python
from funasr import AutoModel
from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD

vad_model = AutoModel(model="fsmn-vad", device="cuda:0")

# Use default threshold configuration
vad = DynamicStreamingVAD(vad_model)

# Or customize thresholds
vad = DynamicStreamingVAD(
    vad_model,
    silence_schedule=[
        (3000, 1500),       # accumulated <=3s: wait 1.5s silence
        (10000, 800),       # accumulated 3-10s: wait 0.8s
        (float('inf'), 300), # accumulated >10s: wait 0.3s
    ],
    speech_noise_thres=0.5,
)
```

#### Streaming Call

```python
import torch

for audio_chunk in audio_stream:  # real-time audio stream
    segments = vad.feed(torch.from_numpy(audio_chunk).float())
    for seg in segments:
        print(f"Speech: {seg[0]}-{seg[1]}ms")

# At the end
final_segments = vad.finalize()
```

#### Non-streaming Call

```python
segments = vad.process(full_audio_tensor)
for seg in segments:
    print(f"Speech: {seg[0]}-{seg[1]}ms")
```

### Default Threshold Configuration

| Accumulated Duration | Silence Threshold | Description |
|---------------------|-------------------|-------------|
| ≤ 5s | 2.0s | Preserve short sentences |
| 5-10s | 1.5s | Normal segmentation |
| 10-15s | 1.0s | Start tightening |
| 15-30s | 0.8s | Faster splitting |
| 30-45s | 0.4s | Prevent overly long segments |
| > 45s | 0.1s | Force split |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vad_model` | - | fsmn-vad instance loaded via FunASR AutoModel |
| `chunk_size_ms` | 60 | VAD internal processing chunk size |
| `speech_noise_thres` | 0.5 | Speech/noise discrimination threshold |
| `speech_to_sil_thres_ms` | 150 | Speech-to-silence base time |
| `silence_schedule` | See table above | Dynamic threshold config `[(upper_ms, silence_ms), ...]` |
| `sample_rate` | 16000 | Sample rate |

### Properties

| Property | Description |
|----------|-------------|
| `vad.is_speaking` | Whether currently in speech state |
| `vad.current_duration_ms` | Current segment accumulated duration |
| `vad.current_threshold_ms` | Current silence threshold in use |
