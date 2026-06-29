# FunASR vLLM Inference Engine Guide

---

## Benchmark

**Test set**: 184 files, 11,541 seconds total. Models: Fun-ASR-Nano / GLM-ASR-Nano.

| Model | Engine | VAD | RTFx | CER | Notes |
|-------|--------|-----|------|-----|-------|
| Fun-ASR-Nano | PyTorch | dynamic | 21 | 8.06% | Baseline |
| Fun-ASR-Nano | **vLLM batch** | dynamic | **340** | **8.20%** | 16x speedup |
| Fun-ASR-Nano | **Offline service (no SPK)** | dynamic | **102** | 8.14% | |
| Fun-ASR-Nano | **Offline service (+SPK)** | dynamic | **46** | 8.19% | SPK off by default |
| GLM-ASR-Nano | **vLLM batch** | fixed | **265** | 12.93% | No long-audio support |

> vLLM matches PyTorch CER exactly (delta < 0.2%) while achieving 16–340x speedup.

---

## Table of Contents

1. [Installation & Environment](#1-installation--environment)
2. [vLLM Engine Architecture](#2-vllm-engine-architecture)
3. [Offline SDK Inference](#3-offline-sdk-inference)
4. [Streaming SDK Inference](#4-streaming-sdk-inference)
5. [Offline Speech Recognition Service](#5-offline-speech-recognition-service)
6. [Streaming Speech Recognition Service](#6-streaming-speech-recognition-service)
7. [Dynamic VAD](#7-dynamic-vad)
8. [API Reference](#8-api-reference)
9. [FAQ](#9-faq)

---

## 1. Installation & Environment

```bash
pip install torch torchaudio
pip install funasr>=1.3.0
# Install vLLM separately after choosing a version compatible with your NVIDIA driver, CUDA runtime, and PyTorch wheel.
pip install safetensors tiktoken websockets regex fastapi uvicorn python-multipart

cd /path/to/FunASR && pip install -e .
```

**Hardware**: GPU ≥ 8 GB VRAM, CUDA ≥ 11.8. 16 GB+ recommended.

Install a PyTorch/torchaudio/vLLM combination that matches your NVIDIA driver and
CUDA runtime. Do not blindly keep the newest wheel if it was built for a newer
CUDA runtime than your driver supports; PyTorch can fail during CUDA
initialization with `The NVIDIA driver on your system is too old` before FunASR
starts. If that happens, reinstall compatible PyTorch, torchaudio, and vLLM
wheels for the CUDA version reported by `nvidia-smi`, or update the NVIDIA
driver first.

---

## 2. vLLM Engine Architecture

### Overall Architecture

FunASR's vLLM integration splits the ASR model into two independently running components:

```
┌──────────────────────────────────────────────────────────────┐
│                  FunASR + vLLM Inference Architecture          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────── PyTorch (single GPU) ───────────┐          │
│  │                                                │          │
│  │  Audio ──→ Frontend ──→ Audio Encoder ──→ Adaptor         │
│  │            (fbank)      (SenseVoice/     (Transformer/    │
│  │                          Whisper)         MLP)            │
│  │                              │                            │
│  │                              ▼                            │
│  │                     Audio Embeddings                      │
│  │                              │                            │
│  │  Text Prompt ──→ Tokenize ──→ Embed                      │
│  │  (system/user/                  │                         │
│  │   hotwords/language)            │                         │
│  │                                 ▼                         │
│  │                          [Concat Embeddings]              │
│  └─────────────────────────────────┼─────────────┘          │
│                                    │                         │
│                                    ▼ EmbedsPrompt            │
│  ┌─────────────── vLLM Engine ────────────────────┐          │
│  │                                                │          │
│  │   PagedAttention + Continuous Batching         │          │
│  │   KV Cache management + CUDA Graph             │          │
│  │   Tensor Parallel (multi-GPU)                  │          │
│  │                                                │          │
│  │   Qwen3-0.6B / Llama-2B (LLM decoding)        │          │
│  │                                                │          │
│  └────────────────────┬───────────────────────────┘          │
│                       │                                      │
│                       ▼                                      │
│                Generated Text                                │
│                       │                                      │
│  ┌────────────────────┼──────────────────────────┐           │
│  │  (Optional) CTC Decoder ──→ Forced Alignment  │           │
│  │           ──→ Character-level timestamps       │           │
│  └───────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────┘
```

### Why vLLM?

| Feature | PyTorch generate() | vLLM |
|---------|-------------------|------|
| KV Cache management | Fixed allocation, wastes memory | PagedAttention, on-demand allocation |
| Batching | Manual padding required | Continuous Batching, automatic scheduling |
| CUDA optimization | None | CUDA Graph + operator fusion |
| Multi-GPU parallelism | Manual implementation | Tensor Parallel with one-line config |
| Throughput | RTFx ~20 | **RTFx 340+** |

### Supported Models

| Model | LLM component | Audio encoder | vLLM speedup |
|-------|--------------|---------------|-------------|
| **Fun-ASR-Nano** | Qwen3-0.6B | SenseVoice | ✓ 21.7x |
| **GLM-ASR-Nano** | Llama-2B | Whisper-like | ✓ 7.6x |
| LLMASR | Qwen/Vicuna | Whisper | ✓ |
| Paraformer | No LLM | — | ✗ Non-autoregressive |
| SenseVoice | No LLM | — | ✗ Encoder-decoder |

### Key Implementation Details

1. **Weight separation**: LLM weights are extracted from `model.pt` and converted to HuggingFace format for vLLM loading
2. **EmbedsPrompt**: Audio embeddings and text embeddings are concatenated and fed to vLLM as a single prompt embedding
3. **use_low_frame_rate**: Fun-ASR-Nano's adaptor output must be truncated to the correct token count via a formula (critical for consistency)
4. **Batch encode**: Multiple audio files pass through `extract_fbank` → `audio_encoder` → `audio_adaptor` in a single forward pass
5. **CTC timestamps**: Encoder output is retained; after text generation, forced alignment yields character-level timing

---

## 3. Offline SDK Inference

Best suited for large-scale audio transcription and offline batch processing. vLLM's batching capability provides the greatest advantage in this scenario.

### Design Principles

Offline SDK inference splits the ASR pipeline into two stages executed independently:

```
┌─────────────────────────────────────────────────────────────────────┐
│            Stage 1: Audio Encoding (PyTorch, single GPU)             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Audio file list ──→ Group (batch of 8) ──→ Frontend (Fbank)        │
│       │                                          │                  │
│       │                                          ▼                  │
│       │                                 SenseVoice Encoder          │
│       │                                          │                  │
│       │                                          ▼                  │
│       │                                 Audio Adaptor               │
│       │                                 (dim transform + LFR trunc) │
│       │                                          │                  │
│       └─── Shared text prompt encoding ────┐     ▼                  │
│            (system/hotwords/language)       │  audio_embeds          │
│                     │                      │     │                  │
│                     ▼                      │     ▼                  │
│                prefix_emb ──→ [concat: prefix | audio | suffix]     │
│                                                  │                  │
│                                                  ▼                  │
│                                        EmbedsPrompt (N samples)     │
└──────────────────────────────────────────────────┼─────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│        Stage 2: LLM Decoding (vLLM, multi-GPU Tensor Parallel)      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  EmbedsPrompt × N ──→ vLLM Continuous Batching                      │
│                        (PagedAttention + CUDA Graph)                 │
│                              │                                      │
│                              ▼                                      │
│                     Generated token_ids × N                          │
│                              │                                      │
│                              ▼                                      │
│                     Decode + post-processing (strip special tokens)  │
│                              │                                      │
│                              ▼                                      │
│                     (Optional) CTC Forced Alignment → char timestamps│
└─────────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

1. **Weight separation**: On first run, weights with the `llm.*` prefix are extracted from `model.pt` and saved in HuggingFace safetensors format for vLLM (cached in the `Qwen3-0.6B-vllm/` directory)
2. **Embedding concatenation**: The text prompt is encoded through the LLM's `embed_tokens` layer into embeddings, then concatenated with the audio adaptor output along the sequence dimension: `[prefix_emb | audio_emb | suffix_emb]`, and submitted to vLLM as an `EmbedsPrompt`
3. **Low Frame Rate truncation**: Adaptor output must be truncated to the correct length using: `fake_token_len = ((((fbank_len - 3 + 2) // 2 - 3 + 2) // 2) - 1) // 2 + 1`, ensuring consistency with the PyTorch training pipeline
4. **Batch audio encoding**: Multiple audio files are grouped in batches of 8 through the encoder + adaptor forward pass, reducing GPU kernel launch overhead
5. **Shared text prompt**: When hotwords and language are identical within a batch, prefix_emb and suffix_emb are computed only once
6. **CTC timestamps**: Encoder output is preserved; after LLM text generation, forced alignment produces character-level timestamps

**Why faster than PyTorch generate()?**

| Dimension | PyTorch | vLLM |
|-----------|---------|------|
| KV Cache | Fixed pre-allocation (wastes memory) | PagedAttention on-demand allocation |
| Batching | Manual padding alignment | Continuous Batching auto-scheduling |
| CUDA | Sequential per-sample execution | CUDA Graph + operator fusion |
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
    inputs="wav.scp",  # supports scp/jsonl/file lists
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

Processes audio in 720 ms chunks incrementally, outputting progressively stable recognition results. Suited for SDK-integrated real-time subtitle scenarios.

### Design Principles

```
Audio stream (720 ms chunks)
    │ Cumulative re-encoding (each chunk covers all audio from the start)
    ▼
┌──────────────────────────┐
│ Stage 1: First 10 chunks │  ← No prev_text; batch generation
│ Identify stable output   │
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

| Accumulated audio | Output quality |
|-------------------|---------------|
| < 1.5 s | Empty or noise |
| 1.5–3.0 s | Partially correct |
| > 3.0 s | Accurate output |

> Note: `repetition_penalty` defaults to `1.0` for Fun-ASR-Nano vLLM paths. Tune it explicitly only after validating that it improves your workload.

---

## 5. Offline Speech Recognition Service

### 5.1 Service Architecture

```
Client                                  serve_vllm.py
  │                                        │
  │── HTTP / OpenAI / WebSocket ─────────→│
  │                                        │
  │                                   ┌────┴────────────────────────┐
  │                                   │ 1. Receive complete audio    │
  │                                   │ 2. Dynamic VAD (≤60 s/seg)  │
  │                                   │ 3. vLLM batch all segments  │
  │                                   │ 4. CTC timestamps (per-char)│
  │                                   │ 5. Speaker diarization (opt)│
  │                                   └────┬────────────────────────┘
  │                                        │
  │←── JSON result ───────────────────────│
```

**Characteristics**:
- Processes audio only after it arrives in full — ideal for file transcription
- Dynamic VAD preserves long segments (≤60 s), reducing boundary-cut losses
- Batch inference over all VAD segments maximizes throughput
- Automatically outputs character-level timestamps
- Speaker diarization is off by default; clients can enable it

### 5.2 Starting the Service

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_vllm.py \
    --port 8899 \
    --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --gpu-memory-utilization 0.5
```

### 5.3 Protocol 1: HTTP REST — `POST /asr`

The most feature-complete interface, supporting speaker diarization, timestamps, and hotwords.

**Request**: `multipart/form-data`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file (wav/mp3/flac) |
| `language` | string | None | Language ("中文" / "English" / ...), None for auto |
| `hotwords` | string | "" | Hotwords, comma-separated |
| `spk` | bool | false | Enable speaker diarization |
| `timestamp` | bool | true | Output character-level timestamps |

**Response**:

```json
{
    "text": "Full transcription text",
    "segments": [
        {
            "text": "Segment text",
            "start": 1.7,
            "end": 14.8,
            "speaker": "SPK0",
            "words": [
                {"word": "砸", "start": 2.02, "end": 2.08},
                {"word": "了", "start": 2.26, "end": 2.32}
            ]
        }
    ],
    "duration": 227.4,
    "processing_time": 3.422,
    "rtf": 0.015
}
```

**Client examples**:

```bash
# cURL
curl -X POST http://localhost:8899/asr \
    -F "file=@meeting.wav" -F "language=中文" -F "spk=true"
```

```python
# Python requests
import requests
resp = requests.post("http://localhost:8899/asr",
    files={"file": open("audio.wav", "rb")},
    data={"language": "中文", "spk": "true"})
result = resp.json()
```

```javascript
// JavaScript fetch
const form = new FormData();
form.append("file", audioBlob, "audio.wav");
form.append("language", "中文");
form.append("spk", "true");
const resp = await fetch("http://localhost:8899/asr", { method: "POST", body: form });
const result = await resp.json();
```

### 5.4 Protocol 2: OpenAI Whisper Compatible — `POST /v1/audio/transcriptions`

Compatible with the OpenAI Whisper API standard; works directly with the OpenAI SDK.

**Request**: `multipart/form-data`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file |
| `model` | string | "fun-asr-nano" | Model name (compatibility field) |
| `language` | string | None | Language |
| `response_format` | string | "json" | "json" / "text" / "verbose_json" |
| `timestamp_granularities` | string | "word" | "word" / "segment" |
| `spk` | bool | false | Speaker diarization (FunASR extension) |

**Response** (`verbose_json`):

```json
{
    "task": "transcribe",
    "language": "zh",
    "duration": 5.17,
    "text": "我一直没有照顾孩子，但是我想要抚养权。",
    "segments": [
        {
            "id": 0, "start": 0.0, "end": 5.15,
            "text": "我一直没有照顾孩子，但是我想要抚养权。",
            "words": [{"word": "我", "start": 0.42, "end": 0.48}, ...]
        }
    ]
}
```

**Client examples**:

```python
# OpenAI SDK (recommended)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8899/v1", api_key="none")
result = client.audio.transcriptions.create(
    model="fun-asr-nano",
    file=open("audio.wav", "rb"),
    response_format="verbose_json",
)
print(result.text)
```

```bash
# cURL
curl -X POST http://localhost:8899/v1/audio/transcriptions \
    -F "file=@audio.wav" -F "model=fun-asr-nano" -F "response_format=verbose_json"
```

### 5.5 Protocol 3: WebSocket — `ws://host:port/ws`

WebSocket interface for the offline service. Send complete audio, then receive results. Speaker clustering is performed automatically on STOP, and results include the `spk` field.

**Client → Server**:

| Message | Description |
|---------|-------------|
| `"START"` | Begin session |
| `"LANGUAGE:中文"` | Set language (optional) |
| `"HOTWORDS:word1,word2"` | Set hotwords (optional) |
| `[binary]` | PCM16 16 kHz mono audio data |
| `"STOP"` | End session; request recognition result |

**Server → Client**:

```json
{"event": "started"}
{"event": "language_set", "language": "中文"}
{"sentences": [{"text":"...","start":..,"end":..}], "is_final": true, "duration_ms": 5170}
{"event": "stopped"}
```

**Client example**:

```python
import asyncio, websockets, json, numpy as np, soundfile as sf

async def offline_ws(audio_path):
    audio, sr = sf.read(audio_path)
    pcm = (audio * 32768).astype(np.int16)

    async with websockets.connect("ws://localhost:8899/ws") as ws:
        await ws.send("START")
        await ws.recv()
        await ws.send("LANGUAGE:中文")
        await ws.recv()

        # Send complete audio
        await ws.send(pcm.tobytes())
        await ws.send("STOP")

        # Receive result
        async for msg in ws:
            data = json.loads(msg)
            if data.get("is_final"):
                for s in data["sentences"]:
                    print(f"[{s['start']/1000:.1f}s] {s['text']}")
                break

asyncio.run(offline_ws("audio.wav"))
```

---

## 6. Streaming Speech Recognition Service

### 6.1 Service Architecture

```
Client (microphone / audio stream)     serve_realtime_ws.py
  │                                      │
  │── WebSocket PCM16 16 kHz ──────────→│
  │   (~100 ms per frame, continuous)    │
  │                                      │
  │                                 ┌────┴─────────────────────────┐
  │                                 │ Real-time loop:               │
  │                                 │  ├─ Dynamic VAD (60 ms chunk) │
  │                                 │  ├─ Endpoint → vLLM decode    │
  │                                 │  ├─ No endpoint → partial     │
  │                                 │  └─ Streaming SPK assignment  │
  │                                 └────┬─────────────────────────┘
  │                                      │
  │←── JSON real-time push ─────────────│
```

**Characteristics**:
- Audio arrives frame by frame; processing starts immediately
- Natural sentence segmentation based on VAD endpoints
- Confirmed segment text is locked and never changes; partial text updates in real time
- Streaming speaker assignment + global re-clustering on STOP
- First-word latency ~480 ms

### 6.2 Starting the Service

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py \
    --port 10095 --language 中文 --hotword-file hotword_list
```

### 6.3 WebSocket Protocol

**Connection**: `ws://host:10095`

**Client → Server**:

| Message | Format | Description |
|---------|--------|-------------|
| Start | `"START"` | Initialize session |
| Hotwords | `"HOTWORDS:word1,word2"` | Optional |
| Language | `"LANGUAGE:中文"` | Optional |
| Audio | `binary` | PCM16 16 kHz mono |
| End | `"STOP"` | Final decode + SPK re-clustering |

**Server → Client**:

```json
{"event": "started"}
{"sentences": [{"text":"你好","start":300,"end":1200,"spk":0}], "partial": "世界", "is_final": false}
{"sentences": [...], "is_final": true}
{"event": "stopped"}
```

**Fields**: `sentences[]` = locked segments, `partial` = text being spoken (may change), `is_final` = true after STOP.

**Sequence diagram**:
```
Client              Server
  │── START ───────→│
  │←─ started ──────│
  │── [audio] ─────→│
  │←─ {partial} ────│
  │── [audio] ─────→│
  │←─ {sentences+partial} ─│  (VAD cut a sentence)
  │── STOP ────────→│
  │←─ {is_final:true} ────│
  │←─ stopped ─────│
```

### 6.4 Client Usage

**Python CLI**:
```bash
python client_python.py --server ws://localhost:10095 --mic
python client_python.py --server ws://localhost:10095 --file audio.wav
```

**Browser**: Open `client_mic.html`

**Custom Python**:
```python
import asyncio, websockets, numpy as np, json

async def stream(audio_path):
    import soundfile as sf
    audio, sr = sf.read(audio_path)
    pcm = (audio * 32768).astype(np.int16)

    async with websockets.connect("ws://localhost:10095") as ws:
        await ws.send("START")
        await ws.recv()

        for i in range(0, len(pcm), 1600):
            await ws.send(pcm[i:i+1600].tobytes())
            await asyncio.sleep(0.05)

        await ws.send("STOP")
        async for msg in ws:
            data = json.loads(msg)
            if data.get("is_final"):
                for s in data["sentences"]:
                    print(f"[{s['start']/1000:.1f}s] {s['text']}")
                break

asyncio.run(stream("audio.wav"))
```

---

## 7. Dynamic VAD

fsmn-vad enables dynamic silence thresholds by default. Offline and streaming modes use different configurations.

| Accumulated duration | Offline (preserve long segs ≤60 s) | Streaming (balance latency) |
|---------------------|-----------------------------------|-----------------------------|
| ≤ 5 s | 2000 ms | 2000 ms |
| 5–10 s | 2000 ms | 1500 ms |
| 10–15 s | 1000 ms | 1000 ms |
| 15–20 s | 1000 ms | 800 ms |
| 20–30 s | 800 ms | 800 ms |
| 30–45 s | 600 ms | 400 ms |
| 45–60 s | 200–400 ms | 100 ms |
| > 60 s | 100 ms | 100 ms |

Offline mode favors longer segments to reduce boundary-cut losses; streaming mode tightens faster to reduce latency.

### Customization

```python
model.generate(input="audio.wav", silence_schedule=[(5000,1500), (20000,800), (float('inf'),300)])
```

> GLM-ASR does not support long-segment inference; pass `dynamic_silence=False` when using it.

---

## 8. API Reference

| Parameter | AutoModelVLLM | serve_vllm.py | serve_realtime_ws.py |
|-----------|--------------|---------------|---------------------|
| model | ✓ | --model | --model |
| gpu_memory_utilization | ✓ | --gpu-memory-utilization | --gpu-memory-utilization |
| tensor_parallel_size | ✓ | — | --tensor-parallel-size |
| max_model_len | ✓ | --max-model-len | --max-model-len |
| language | generate() param | API param | --language / LANGUAGE: |
| hotwords | generate() param | API param | --hotword-file / HOTWORDS: |

---

## 9. FAQ

**Q: Offline or streaming?**
Complete files → offline (high throughput). Microphone / live stream → streaming (low latency).

**Q: Can GLM-ASR use dynamic VAD?**
It does not support long-segment inference. Use `dynamic_silence=False`.

**Q: Performance impact of SPK?**
RTFx drops from 102 to 46. CER is unchanged. Disabled by default.

**Q: Entry points for custom development?**
Offline: `serve_vllm.process_audio()` / `FunASRNanoVLLM.generate()`
Streaming: `serve_realtime_ws.RealtimeASRSession`

**Q: Slow first startup?**
vLLM initialization takes 60–90 s (KV Cache + CUDA Graph warmup). Subsequent inferences are instant.
