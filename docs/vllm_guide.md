# FunASR vLLM Inference Engine Guide

---

## Benchmark

**Test set**: 184 files, 11,541 seconds total. Models: Fun-ASR-Nano / GLM-ASR-Nano. See [Benchmark RTF and Reproducibility Notes](./benchmark/rtf_reproducibility.md) for the `RTFx` definition, timing scope checklist, and fields required for comparable reports.

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

Install vLLM first, choosing a version compatible with your NVIDIA driver's CUDA. vLLM pins and installs a matching torch / torchaudio / torchvision trio automatically, so do not install torch/torchaudio yourself — the three are ABI-locked, i.e. they must be the matching set built against each other (e.g. torch 2.10.0 ↔ torchaudio 2.10.0 ↔ torchvision 0.25.0). 

```bash
# 1) Install vLLM first. Pick the version by the CUDA version shown in `nvidia-smi`
#    (the driver's max CUDA), NOT the runtime CUDA. vLLM brings a matching torch/torchaudio/torchvision.
#    driver CUDA 12.x  -> pip install vllm==0.19.1   (ships torch 2.10 / cu128)
#    driver CUDA >= 13 -> pip install vllm           (latest; ships torch 2.11 / cu130)
pip install "vllm==0.19.1"   # adjust to your driver CUDA; see note below

# 2) Then FunASR and the rest.
pip install "funasr>=1.3.19"

cd /path/to/FunASR && pip install -e .
```

**Hardware**: GPU ≥ 8 GB VRAM, CUDA ≥ 11.8. 16 GB+ recommended.

Why not pip install torch torchaudio? The torch/torchaudio/torchvision versions are determined by the vLLM release — each major vLLM version bumps them together (see vLLM's requirements/cuda.txt). Installing them by hand pulls the newest wheel, which may be built for a newer CUDA runtime than your driver supports; PyTorch then fails during CUDA initialization with The NVIDIA driver on your system is too old before FunASR even starts. Letting vLLM own the trio avoids this. If you still hit a driver-too-old error, install a vLLM version whose CUDA build matches the CUDA reported by nvidia-smi (e.g. vllm==0.19.1 for CUDA 12.x), or update the NVIDIA driver first.

---

## 2. vLLM Engine Architecture

### Overall Architecture

FunASR's vLLM integration splits the ASR model into two independently running components:

```
┌──────────────────────────────────────────────────────────────┐
│                  FunASR + vLLM Inference Architecture        │
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
│  │  Text Prompt ──→ Tokenize ──→ Embed                       │
│  │  (system/user/                  │                         │
│  │   hotwords/language)            │                         │
│  │                                 ▼                         │
│  │                          [Concat Embeddings]              │
│  └─────────────────────────────────┼─────────────┘           │
│                                    │                         │
│                                    ▼ EmbedsPrompt            │
│  ┌─────────────── vLLM Engine ────────────────────┐          │
│  │                                                │          │
│  │   PagedAttention + Continuous Batching         │          │
│  │   KV Cache management + CUDA Graph             │          │
│  │   Tensor Parallel (multi-GPU)                  │          │
│  │                                                │          │
│  │   Qwen3-0.6B / Llama-2B (LLM decoding)         │          │
│  │                                                │          │
│  └────────────────────┬───────────────────────────┘          │
│                       │                                      │
│                       ▼                                      │
│                Generated Text                                │
│                       │                                      │
│  ┌────────────────────┼──────────────────────────┐           │
│  │  (Optional) CTC Decoder ──→ Forced Alignment  │           │
│  │           ──→ Character-level timestamps      │           │
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
2. **EmbedsPrompt**: a vLLM input mode that feeds **precomputed embedding vectors** (rather than the usual token IDs) directly as the prompt (enabled via `enable_prompt_embeds=True`). Fun-ASR-Nano requires it because the audio, after the adaptor, is a sequence of continuous vectors — not tokens — so the audio embeddings and text embeddings are concatenated along the sequence dimension and submitted to vLLM as a whole
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
│            Stage 1: Audio Encoding (PyTorch, single GPU)            │
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
│            (system/hotwords/language)      │  audio_embeds          │
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
│                        (PagedAttention + CUDA Graph)                │
│                              │                                      │
│                              ▼                                      │
│                     Generated token_ids × N                         │
│                              │                                      │
│                              ▼                                      │
│                     Decode + post-processing (strip special tokens) │
│                              │                                      │
│                              ▼                                      │
│                    (Optional) CTC Forced Alignment → char timestamps│
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

> **Note: `repetition_penalty` cannot be used with EmbedsPrompt.** Here the prompt is a sequence of embedding vectors with no corresponding token IDs, whereas `repetition_penalty` needs the prompt's token IDs to down-weight already-seen tokens in the logits; applying it under EmbedsPrompt **indexes out of bounds and triggers a CUDA device-side assert**. 

### Production API Stability Checklist

When wrapping `AutoModelVLLM` in a long-running API service, keep request state isolated and pin safe decoding defaults:

```python
common = dict(
    language="auto",
    temperature=0.0,
    repetition_penalty=1.0,
    max_new_tokens=200,
)

for _ in range(2):
    results = model.generate(["vad_segment_01.wav", "vad_segment_02.wav"], **common)
    print([r["text"] for r in results])
```

If the same audio is normal on the first request but repeats on the second request:

1. Run the minimal script above outside the API layer with the same VAD segments.
2. If the script is stable, check whether the API wrapper reuses per-request variables, previous VAD segment lists, previous `results`, or accumulated text across requests.
3. If the script also repeats, capture the exact `funasr`, `vllm`, and `torch` versions, plus the first and second outputs, before tuning any decoding parameter.

Do not increase `repetition_penalty` to suppress repeats on Fun-ASR-Nano vLLM. The prompt-embeds path should stay at the neutral value `1.0`.

---

## 5. Offline Speech Recognition Service

### 5.1 Service Architecture

```
Client                                  serve_vllm.py
  │                                        │
  │── HTTP / OpenAI / WebSocket ─────────→│
  │                                        │
  │                                   ┌────┴────────────────────────┐
  │                                   │ 1. Receive complete audio   │
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

> **About [`CUDA_VISIBLE_DEVICES`](https://docs.vllm.ai/en/v0.4.3/serving/env_vars.html)**:  the `=0` in the examples is just an illustrative value ("use GPU 0"), **not a fixed requirement**. It selects which GPUs are visible to this process (indexed as in `nvidia-smi`), a single GPU machine does not need to set it.
>
> - **Single GPU**: small models like 0.6B / 1.7B can run several instances on one card — point multiple processes at the same GPU (e.g. all `=0`) sharing it via MPS, or split across cards with process A `=0`, B `=1` (see §6.7).
>

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
  │── WebSocket PCM16 16 kHz ───────────→│
  │   (~100 ms per frame, continuous)    │
  │                                      │
  │                                 ┌────┴──────────────────────────┐
  │                                 │ Real-time loop:               │
  │                                 │  ├─ Dynamic VAD (60 ms chunk) │
  │                                 │  ├─ Endpoint → vLLM decode    │
  │                                 │  ├─ No endpoint → partial     │
  │                                 │  └─ Streaming SPK assignment  │
  │                                 └────┬──────────────────────────┘
  │                                      │
  │←── JSON real-time push ──────────────│
```

**Characteristics**:
- Audio arrives frame by frame; processing starts immediately
- Natural sentence segmentation based on VAD endpoints
- Confirmed segment text is locked and never changes; partial text updates in real time
- Optional streaming speaker assignment (`--enable-spk`) + global re-clustering on STOP
- First-word latency ~480 ms

### 6.2 Starting the Service

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py \
    --port 10095 --language 中文 --hotword-file hotword_list
```

For multi-client or long continuous-speech workloads, start by bounding partial previews and lowering the refresh rate:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py \
    --port 10095 --language 中文 \
    --partial-window-sec 8 --decode-interval 0.8
```

Speaker diarization is disabled by default; add `--enable-spk` only when the `spk` field is required.

For long-lived microphone sessions behind Docker, nginx, or a cloud load
balancer, keep WebSocket ping/pong enabled and tune the timeout to be longer
than short network stalls:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py \
    --port 10095 --language 中文 \
    --ws-ping-interval 20 --ws-ping-timeout 60
```

Set `--ws-ping-interval 0` only when an external gateway already owns
keepalive/reconnect policy.

For long-session debugging, especially with `--enable-spk`, enable periodic
session-state logs:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py \
    --port 10095 --language 中文 --enable-spk \
    --log-session-stats-interval 30
```

This prints a `Session stats:` line every 30 seconds. Include the last few
lines in issue reports together with tail RTF, process RSS, GPU memory, and
the final disconnect log.

### 6.3 WebSocket Protocol

**Connection**: `ws://host:10095`

**Client → Server**:

| Message | Format | Description |
|---------|--------|-------------|
| Start | `"START"` | Initialize session |
| Hotwords | `"HOTWORDS:word1,word2"` | Optional |
| Language | `"LANGUAGE:中文"` | Optional |
| Audio | `binary` | PCM16 16 kHz mono |
| End | `"STOP"` | Final decode; SPK re-clustering only when `--enable-spk` is enabled |

**Server → Client**:

```json
{"event": "started"}
{"sentences": [{"text":"你好","start":300,"end":1200}], "partial": "世界", "is_final": false}
{"sentences": [...], "is_final": true}
{"event": "stopped"}
```

**Fields**: `sentences[]` = locked segments, `partial` = text being spoken (may change), `partial_start_ms` = where the current provisional `partial` begins, `is_final` = true after STOP. When `--enable-spk` is enabled, `sentences[]` also includes `spk`.

**Sequence diagram**:
```
Client              Server
  │── START ───────→│
  │←─ started ──────│
  │── [audio] ─────→│
  │←─ {partial} ────│  #refer to 6.5
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

**Realtime benchmark**:
```bash
python examples/industrial_data_pretraining/fun_asr_nano/realtime_ws_benchmark.py \
    audio_16k_mono_pcm16.wav --server ws://localhost:10095 --clients 4 \
    --output-jsonl realtime_ws_4c.jsonl
```

For metric definitions and reporting fields, see [Realtime WebSocket Benchmark](./benchmark/realtime_ws_benchmark.md).

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

### 6.5 Partial preview mechanism and long-sentence behavior

**What partial is and how it's produced**
While the user is speaking, the streaming service periodically (default `decode_interval≈0.48s` in `serve_realtime_ws.py`) decodes "the current sentence from its start up to now," emitting **provisional text** (the `partial` field in the protocol, which may be overwritten by later refreshes), until VAD detects the sentence end and locks it into `sentences`. This lets the user see text as they speak.

> Note: `serve_vllm.py`'s `/ws` (§5) has **no partial** and only returns at sentence end; use `serve_realtime_ws.py` for live preview.

**Frontend rendering rule**
Treat `partial` as a replaceable preview, not as text to append. A good UI keeps locked text and preview text separate:

```js
const committed = data.sentences.map((s) => s.text).join("");
const preview = data.partial || "";
render(committed + preview);
```

If `partial_start_ms` moves forward because `--partial-window-sec` is active, the preview only describes the current bounded decode window. Replace the preview area on each message; append only VAD-locked `sentences` or the final `is_final=true` result.

**Principle: why each partial re-encodes the whole segment from the start**
Fun-ASR-Nano's acoustic encoder (SenseVoice) is a **full-context, non-streaming** encoder — each frame's representation depends on the context of the entire segment. When the sentence continues and the audio grows, the context of the earlier frames changes, so **the previously computed encoding no longer holds**. It therefore cannot cache history and encode only the new frames the way a streaming / causal encoder would; it must run the whole "start → now" segment through the encoder again.

**Resulting behavior: partial gets slower on long sentences (O(L²))**
Because each refresh re-encodes from the sentence start, the longer a sentence, the longer each partial's audio and the more refreshes occur — so **total encoding work grows quadratically with sentence length**. In practice a ~29 s continuous utterance is fully re-encoded a dozen-plus times, with single-pass encoder time climbing from tens to hundreds of milliseconds. (The §4 SDK streaming "each chunk contains all audio from the start to now" is the same mechanism; long files behave the same way.)

**Usage guidance**
- Normal conversational speech has natural pauses, so VAD splits it into relatively short utterances and each partial's cost is naturally bounded — **usually nothing to worry about**.
- Only **very long, pauseless continuous speech** (e.g. reading aloud) makes a single utterance keep growing and the partial preview progressively slower. `serve_realtime_ws.py` bounds provisional previews with `--partial-window-sec 15` by default; for multi-client or continuous-monologue load tests, try `8-10` and raise `--decode-interval` to `0.8-1.0`. This only affects provisional `partial`; VAD-locked sentences and STOP final output still run on the full audio.

### 6.6 Cost of speaker diarization (SPK) and how to enable it

`serve_realtime_ws.py` **does not load** the SPK model by default. It loads `--spk-model` (default `iic/speech_eres2netv2_sv_zh-cn_16k-common`) only when started with `--enable-spk`, then runs speaker assignment for each VAD-completed sentence during streaming. Note:

- **SPK is of limited effectiveness on Fun-ASR-Nano** (see #2944); most real-time ASR scenarios do not need speaker separation.
- **Streaming SPK is expensive and grows with the session**: each sentence re-clusters **all historical embeddings** (**O(N²)**, more expensive per sentence as the session grows) and **synchronously blocks the event loop**; the session also **re-clusters everything again** at the end, so the per-sentence clustering during streaming is overwritten by the final result — redundant as far as the final output is concerned. This is especially pronounced under long sessions + high concurrency.
- **Recommendation**: keep the default off for multi-client live ASR; if diarization is required, add `--enable-spk` and treat the final STOP-time labels as authoritative.
- **Long-session diagnostics**: when a session still slows down or disconnects, rerun with `--log-session-stats-interval 30` and check whether `audio_buffer_samples`, `locked_sentences`, `speaker_history_chunks`, `speaker_history_embeddings`, and `speaker_centers` stay bounded. If those counters stay near their limits while RTF keeps rising, the remaining bottleneck is more likely model inference, response payload size, or environment scheduling rather than retained session state.

### 6.7 Production concurrency and multi-process deployment

`serve_realtime_ws.py` is a **single-asyncio-event-loop** service: both `decode()` (timed partial) and `add_audio()` (decode triggered at VAD sentence end) **synchronously block** the entire event loop — while any one connection is decoding, all others pause sending/receiving. Therefore:

- **The single-process concurrency ceiling comes from event-loop serialization, not GPU compute.** Under high concurrency GPU utilization stays low and the encoder runs at ~86× real time; mistaking this for insufficient GPU and adding cards or tensor parallelism yields little (TP only splits the LLM, not the standalone encoder).
- **The right way to scale (currently) = multiple independent processes on one card + CUDA MPS + nginx round-robin**: each process has its own GIL and CUDA context, sidestepping the single-loop serialization; MPS lets the processes truly share the GPU concurrently and fill the idle compute; nginx round-robins across the WebSocket backends. Beyond a single card's headroom, scale out horizontally (one instance per card + a load balancer).
- **vLLM is not always more efficient for small real-time streams than several PyTorch processes.** vLLM helps most when requests can be batched or when LLM token decoding dominates. The current real-time WebSocket path submits many small, synchronous per-connection decode calls through one event loop, so it may reserve much more memory while still leaving GPU utilization modest. For a small number of continuous microphone streams, several lighter PyTorch processes can be easier to pack on one card. For vLLM, benchmark with the real traffic shape and start with lower `--gpu-memory-utilization` plus multiple service processes instead of assuming one vLLM process should carry every stream.
- **Sustainable concurrency has no universal "supports N connections" number.** The ceiling is set not by the number of connections but by **how many are speaking at the same moment** — each speaking connection triggers a partial decode roughly once per second, all serialized on that single event loop. It mainly varies with: **① silence ratio** — in real turn-taking users spend most of the time listening, so far fewer are decoding simultaneously than are connected, whereas a continuous monologue keeps nearly every connection decoding; **② sentence length** — longer sentences make each partial encode more expensive (see 6.5's O(L²)), raising load at the same connection count. So the same "single L20 + multi-process + MPS" setup can sustain dozens of connections under turn-taking-like load but markedly fewer under long, pauseless speech. **Any "supports X connections" figure holds only for the traffic profile it was measured under** — benchmark with your own real traffic (sentence length, pauses, continuous or not) rather than treating someone else's number as your spec.

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

**Q: vLLM returns repeated punctuation such as `!!!!!!!!` but PyTorch/HF generate is normal. What should I check?**
This usually means the audio frontend and checkpoint can work, but the vLLM
prompt-embedding path or decoding parameters differ from the upstream runner.
Check these items before changing the model:

- Pass prompt embeddings to vLLM as float32:
  `EmbedsPrompt(prompt_embeds=input_embeds.float())`.
- Use ASR-style deterministic decoding. The Fun-ASR-Nano vLLM path defaults to
  `temperature=0.0`, `top_p=1.0`, and `skip_special_tokens=True`. In
  prompt-embeds mode, keep `repetition_penalty` at the neutral `1.0` unless you
  are using a token-prompt path; other values are normalized by FunASR's vLLM
  helpers to avoid vLLM CUDA scatter errors.
- Verify that `model_dir` and `vllm_model_dir` are the matching Fun-ASR-Nano
  pair. If clearing `vllm_model_dir` makes the same audio work through HF
  generate, keep debugging the vLLM path rather than the audio file.
- Log vLLM `finish_reason`, generated token ids, prompt embedding dtype, and
  prompt embedding shape for one failing sample. Repeated punctuation with
  `finish_reason="length"` usually points to decode/prompt mismatch rather than
  VAD or audio loading.
