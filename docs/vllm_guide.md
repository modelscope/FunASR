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

> In the reported table, Fun-ASR-Nano batch throughput is `340 / 21 = 16.2` times the PyTorch baseline when timing scopes match. `RTFx 340` means 340 times realtime, not 340 times faster than PyTorch. CER changes from `8.06%` to `8.20%` (+0.14 percentage points), not identical accuracy. These historical measurements are not a guarantee for other hardware, workloads, or configurations.

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

The SDK, offline service, and WebSocket service in this guide use the **FunASR split engine**. Keep its environment separate from the native `vllm serve` validation described below. Choose a vLLM release and GPU wheel before installing into a dedicated virtual environment. The CUDA number in `nvidia-smi` is the driver's supported upper bound, not the installed CUDA runtime; a broad label such as "12.x" or "13.x" is not sufficient to establish wheel compatibility.

The following starting point uses the split-engine version `vllm==0.19.1` and a fixed FunASR source commit. It pins those two projects, but **is not a complete dependency lockfile or a clean-install validation for every GPU**. Check GPU builds and driver requirements in the [versioned vLLM installation documentation](https://github.com/vllm-project/vllm/blob/v0.19.1/docs/getting_started/installation/gpu.md).

```bash
python3.12 -m venv .venv-funasr-vllm
source .venv-funasr-vllm/bin/activate
python -m pip install "vllm==0.19.1"

# Service scripts come from source. Use a new directory, not an existing checkout.
git clone https://github.com/modelscope/FunASR.git FunASR-vllm
cd FunASR-vllm
git checkout --detach e42443f55971d0c804dcf2973fdd2e6e09bd5611
python -m pip install -e .
python -m pip install safetensors tiktoken websockets regex fastapi uvicorn python-multipart
python -m pip check
python -m pip freeze > environment.txt
```

Save the GPU/driver, Python version, source commit, model revision, and `environment.txt`, then validate single requests, actual WebSocket sessions, and the intended concurrent workload. `pip check` checks declared dependency relationships only; it does not verify CUDA, audio operators, or end-to-end serving. Installing the PyPI package alone does not provide the repository service scripts referenced here.

### Choose the model path before you start

There are two different Fun-ASR-Nano vLLM integrations. Their checkpoints and
APIs are not interchangeable:

#### A. FunASR split-engine integration (this guide)

Use the official `FunAudioLLM/Fun-ASR-Nano-2512` checkpoint from
[ModelScope](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) or
[Hugging Face](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512). The
current Hugging Face `model.pt` has incomplete CTC weights, so it remains
usable for transcription but FunASR disables the affected CTC path rather than
returning unreliable timestamps or speaker diarization. Use the ModelScope
checkpoint when your deployment requires timestamps or speaker diarization;
the publication repair is tracked in [#3496](https://github.com/modelscope/FunASR/issues/3496).
The `Qwen3-0.6B/` subdirectory intentionally contains only the LLM config and
tokenizer; it is not a standalone model download.

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

# Choose one official hub. ModelScope is the default.
model = AutoModelVLLM(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    hub="ms",
)
# model = AutoModelVLLM(
#     model="FunAudioLLM/Fun-ASR-Nano-2512",
#     hub="hf",
# )
```

On first load, FunASR calls `prepare_vllm_model_dir()` automatically. It copies
the config/tokenizer from `Qwen3-0.6B/`, extracts the `llm.*` tensors from the
root `model.pt`, and writes
`Qwen3-0.6B-vllm/model.safetensors`. Do not point `model` at `Qwen3-0.6B/` and
do not try to serve that config-only directory directly with vLLM.

#### B. Native vLLM transcription integration

Native `FunASRForConditionalGeneration` uses a full native checkpoint, not the
split `model.pt` layout in path A. The official native checkpoint is
[`FunAudioLLM/Fun-ASR-Nano-2512-vllm`](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-vllm).
At the 2026-09-07 audit, the v0.28.0 model registry still referenced
[`allendou/Fun-ASR-Nano-2512-vllm`](https://huggingface.co/allendou/Fun-ASR-Nano-2512-vllm),
a community-converted full checkpoint hosted outside the official FunAudioLLM
organization, for its native `FunASRForConditionalGeneration` architecture.
Main and released versions need not use the same reference; the official
validation record below identifies the versions checked, rather than treating
all vLLM model lists as interchangeable.
Use either native checkpoint only when you intentionally choose vLLM's native
transcription API. Do not substitute either one for the official checkpoint in
the FunASR `AutoModelVLLM` examples below or pass it to
[serve_realtime_ws.py](../examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py).
Those services expect `model.pt`, `config.yaml`, and `Qwen3-0.6B/`, not the
native layout.

Both native paths use `vllm serve` for non-realtime request/response
transcription at `/v1/audio/transcriptions`; they do not register
`/v1/realtime`. vLLM registers that WebSocket endpoint only for models that
declare the realtime task, and `FunASRForConditionalGeneration` is not in its
[Realtime Transcription table](https://docs.vllm.ai/en/latest/models/supported_models/#realtime-transcription).
For realtime streaming, use the FunASR streaming SDK inference or streaming ASR
service. The `AutoModelVLLM` examples in path A are offline inference as well.

The native HTTP API does not inherit the FunASR WebSocket service's VAD, partial
previews, session state, or SPK processing. A WebSocket handshake rejection
status is not a stable API contract or a way to probe model realtime support.

See the [official native validation record](./vllm_official_native_validation.md)
for the pinned model revision, launch arguments, and recorded output. It covers
file transcription with vLLM 0.27.1 in an existing H100 environment, not a clean
installation, accuracy, capacity, long-audio, realtime streaming, or speaker
diarization validation. Do not substitute that environment for the 0.19.1
split-engine installation above.

**Hardware**: follow the requirements of the selected vLLM GPU build. Memory
usage also depends on the model, precision, KV cache, batch size, and active
sessions. This guide does not promise a universal minimum VRAM or concurrency
capacity.

Do not upgrade `torch` or `torchaudio` individually inside a validated environment.
Dependency constraints change between vLLM releases; there is no universal
"automatically matching trio" guarantee. For example, the
[0.19.1 release metadata](https://pypi.org/pypi/vllm/0.19.1/json) declares
`torch==2.10.0`, `torchaudio==2.10.0`, and `torchvision==0.25.0`, whereas the
[0.27.1 release metadata](https://pypi.org/pypi/vllm/0.27.1/json) declares
`torch==2.13.0`, `torchaudio==2.11.0`, and `torchvision==0.28.0`.
These are declared constraints, not proof of successful installation or ABI
compatibility. Upgrade in a new environment and validate again. For a
driver-too-old error, check the actual wheel's CUDA build and driver requirements
instead of blindly installing the latest release.

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
| Reported batch throughput | RTFx 21 | RTFx 340; see Benchmark scope |

### Supported Models

| Model | LLM component | Audio encoder | Integration |
|-------|--------------|---------------|-------------|
| **Fun-ASR-Nano** | Qwen3-0.6B | SenseVoice | Specialized split engine |
| **GLM-ASR-Nano** | Llama-2B | Whisper-like | Specialized split engine |
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
| Reported batch result | RTFx 21 | RTFx 340; hardware and timing scope must match |

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
    inputs=["audio1.wav", "audio2.wav"],  # existing audio files, not manifests
    hotwords=["开放时间"],
    language="中文",
    max_new_tokens=512,
)
```

The direct engine accepts an audio path, a list of audio paths, or 16 kHz waveform arrays/tensors. It does not expand SCP/JSONL manifests. Use [demo_vllm.py](../examples/industrial_data_pretraining/fun_asr_nano/demo_vllm.py) below for manifests: SCP rows contain a path or `key path`; JSONL rows contain a `source` audio-path field. Paths must resolve from the process working directory.

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

Illustrative values, not a measured run. HTTP timestamps and duration are in seconds. Optional `words` and `speaker` fields require their respective processing paths; see [the serializer and service](../examples/industrial_data_pretraining/fun_asr_nano/serve_vllm.py).

```json
{
    "text": "你好",
    "segments": [
        {
            "text": "你好",
            "start": 0.3,
            "end": 1.2,
            "speaker": "SPK0",
            "words": [
                {"word": "你", "start": 0.3, "end": 0.6},
                {"word": "好", "start": 0.6, "end": 1.2}
            ]
        }
    ],
    "duration": 2.0,
    "processing_time": 0.1,
    "rtf": 0.05
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

Illustrative complete response; timestamps are in seconds and are not a recognition-quality guarantee.

```json
{
    "task": "transcribe",
    "language": "zh",
    "duration": 2.0,
    "text": "你好",
    "segments": [
        {
            "id": 0, "start": 0.3, "end": 1.2,
            "text": "你好",
            "words": [
                {"word": "你", "start": 0.3, "end": 0.6},
                {"word": "好", "start": 0.6, "end": 1.2}
            ]
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

Each line below represents a separate JSON WebSocket message, not one JSON document. Values are illustrative; WebSocket offsets are in milliseconds.

```text
{"event": "started"}
{"event": "language_set", "language": "中文"}
{"sentences": [{"text": "你好", "start": 300, "end": 1200}], "is_final": true, "duration_ms": 2000}
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
- The partial-decode interval is a scheduling setting, not a first-word latency guarantee; measure end-to-end latency on your workload

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

The server sends WebSocket pings every 20 seconds by default, while ping-timeout
closure is disabled. Under concurrent long-audio load, model and VAD work can
delay control-frame handling even though the connection is healthy. A fixed
timeout can therefore close valid sessions during compute or queue backpressure.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py \
    --port 10095 --language 中文
```

Set a positive `--ws-ping-timeout` only after measuring the worst-case decode
and queue delay for the production traffic shape; keep it above that delay and
coordinate it with the gateway idle-timeout policy. The `websockets` library's
receive high-water mark (`max_queue`) can pause socket reads when its queue fills;
this also delays processing Ping frames even if inference runs off the event loop.
Increasing that mark only moves the overload threshold. It isn't a throughput
fix. Set `--ws-ping-interval 0` only when an external gateway already owns
keepalive/reconnect policy.

The source server receives messages independently of session inference, with a
bounded application FIFO: `--ws-receive-max-messages 128` and
`--ws-receive-max-bytes 16777216`. Both limits must be positive. They count queued
messages and payload bytes (UTF-8 bytes for text commands), not total process
memory: protocol buffering, an in-flight message and session audio add to it.
An overflow closes the connection with **1013**, not a successful final result.
Do not automatically replay a partially processed session without an
application-level recovery policy.

Audio and commands remain ordered. When the next queued message is more audio,
an otherwise-due provisional decode is deferred until that backlog is consumed;
audio itself isn't dropped or combined. VAD completed-segment decoding and
`COMMIT`/`STOP` final decoding still process their full input. Preview cadence and
context/fallback observations may change under load, so identical transcript
text or improved hardware throughput isn't guaranteed. Use
`--log-decode-profile` to record `peak_messages`, `peak_bytes` and
`skipped_partials` alongside the existing engine profile.

These source changes are not in the published `funasr==1.4.15` package. Transport
tests with synthetic decoding do not replace L20 or production-traffic
acceptance; see [the benchmark contract](benchmark/realtime_ws_benchmark.md).

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

Illustrative message sequence, one JSON object per WebSocket message. Offsets are in milliseconds; this is not a measured latency trace.

```text
{"event": "started"}
{"sentences": [{"text": "你好", "start": 300, "end": 1200}], "partial": "世界", "is_final": false}
{"sentences": [{"text": "你好", "start": 300, "end": 1200}, {"text": "世界", "start": 1500, "end": 2200}], "is_final": true}
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
- Only **very long, pauseless continuous speech** (e.g. reading aloud) makes a single utterance keep growing and the partial preview progressively slower. `serve_realtime_ws.py` bounds provisional previews with `--partial-window-sec 8` by default; raise the window only after measuring headroom for multi-client continuous-monologue load. This only affects provisional `partial`; VAD-locked sentences and STOP final output still run on the full audio. See the measured L20 starting point in §6.7.

### 6.6 Cost of speaker diarization (SPK) and how to enable it

`serve_realtime_ws.py` **does not load** the SPK model by default. It loads `--spk-model` (default `iic/speech_eres2netv2_sv_zh-cn_16k-common`) only when started with `--enable-spk`, then runs speaker assignment for each VAD-completed sentence during streaming. Note:

- **SPK is of limited effectiveness on Fun-ASR-Nano** (see #2944); most real-time ASR scenarios do not need speaker separation.
- **Streaming SPK is expensive and grows with the session**: each sentence re-clusters **all historical embeddings** (**O(N²)**, more expensive per sentence as the session grows) in that session's worker; the session also **re-clusters everything again** at the end, so the per-sentence clustering during streaming is overwritten by the final result — redundant as far as the final output is concerned. This is especially pronounced under long sessions + high concurrency.
- **Recommendation**: keep the default off for multi-client live ASR; if diarization is required, add `--enable-spk` and treat the final STOP-time labels as authoritative.
- **Long-session diagnostics**: when a session still slows down or disconnects, rerun with `--log-session-stats-interval 30` and check whether `audio_buffer_samples`, `locked_sentences`, `speaker_history_chunks`, `speaker_history_embeddings`, and `speaker_centers` stay bounded. If those counters stay near their limits while RTF keeps rising, the remaining bottleneck is more likely model inference, response payload size, or environment scheduling rather than retained session state.

### 6.7 Production concurrency and multi-process deployment

`serve_realtime_ws.py` keeps network I/O on one asyncio loop, but runs each connection's blocking session work in worker threads. Compatible ASR requests that arrive together are collected for up to `--decode-batch-wait-ms` (10 ms by default), flattened into one `AutoModelVLLM.generate()` call, and capped by `--decode-max-batch-size` (16 audio segments by default). The shared vLLM engine still has one controlled caller, while audio ingestion and session state are no longer held behind a process-wide lock.

- **Streaming VAD defaults to CPU.** Each connection gets an isolated FSMN-VAD instance using `--vad-device cpu --vad-ncpu 1`, avoiding shared mutable cache state and per-frame CUDA allocator synchronization. Override the device only after benchmarking the full WebSocket path; multiplying CPU threads by the number of active sessions can also oversubscribe a host.
- **Tune batching for the traffic profile.** The 10 ms default is small relative to the 480 ms partial-decode interval and gives simultaneous speakers a chance to share encoder and vLLM batches. Lower it toward `0` for the lowest single-stream queue latency; raise it cautiously when throughput matters more than a few milliseconds of added queue time. Increase the maximum batch only after measuring GPU memory and tail latency.
- **One process is the first scaling unit.** Benchmark a single process with the built-in batching path before adding replicas. Use multiple processes or one instance per GPU only after a single process reaches its measured GPU, CPU, or tail-latency limit; each extra process duplicates model memory and may reduce batching opportunities.
- **vLLM benefits depend on requests arriving together.** Turn-taking traffic may have many connected clients but only a few simultaneous decodes, while replaying the same continuous monologue across every client creates deliberately synchronized batches. Report both traffic shape and batching flags with every result.
- **Sustainable concurrency has no universal "supports N connections" number.** It depends mainly on simultaneous speakers, silence ratio, utterance length, partial refresh interval, speaker diarization, batch wait, and GPU/CPU capacity. Long pauseless speech still costs more because provisional windows are repeatedly encoded (see §6.5). Benchmark your own workload instead of adopting another deployment's connection count as a specification.
- **Historical L20 starting point, not a capacity guarantee.** In one historical measurement from [#3528](https://github.com/modelscope/FunASR/issues/3528), one L20 running 16 synchronized clients on a 47-second continuous utterance, with SPK and client ping disabled, performed best at `--partial-window-sec 8 --decode-interval 2.0`: 408 decode requests, 3,072.1 seconds of encoded audio, 51.18-second p50 completion, 4.5-second output lag, 14.2x aggregate realtime, and 1.31-second first text. The then-default 15-second window did not complete that exact 16-client workload; the current source default is 8 seconds. This observation does not replace results from later versions or different keepalive settings. Use `8 / 2.0` only as a workload-specific tuning starting point, recording the version, window, keepalive, first-text latency, output lag, final completion, request count, and encoded-audio total. A successful Git installation is not resolution of the concurrency issue, which requires separate acceptance evidence.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py \
  --partial-window-sec 8 --decode-interval 2.0 --log-decode-profile
```

---

## 7. Dynamic VAD

Dynamic silence is a **VAD-stage** option, not an ASR decoder option. [FSMN-VAD](../funasr/models/fsmn_vad_streaming/model.py) reads `dynamic_silence` and `silence_schedule`; an explicit `max_end_silence_time` disables the dynamic default unless overridden. `AutoModelVLLM.generate(inputs, **kwargs)` forwards to the ASR engine and does not run VAD.

The table samples the SDK's `DEFAULT_SILENCE_SCHEDULE` and `STREAMING_SILENCE_SCHEDULE` constants at boundary durations. Each schedule selects the first entry whose duration limit is greater than or equal to accumulated speech; these are silence thresholds, not maximum utterance lengths.

| Accumulated speech sample | DEFAULT_SILENCE_SCHEDULE | STREAMING_SILENCE_SCHEDULE |
| --- | --- | --- |
| 5000 ms | 2000 ms | 2000 ms |
| 10000 ms | 2000 ms | 1500 ms |
| 15000 ms | 1000 ms | 1000 ms |
| 20000 ms | 1000 ms | 800 ms |
| 30000 ms | 800 ms | 800 ms |
| 40000 ms | 600 ms | 400 ms |
| 45000 ms | 400 ms | 400 ms |
| 50000 ms | 400 ms | 100 ms |
| 60000 ms | 200 ms | 100 ms |
| 60001 ms | 100 ms | 100 ms |

The streaming-named constant is not automatically selected just because audio arrives in chunks. Service wrappers can choose their own policy; for example [DynamicStreamingVAD](../funasr/models/fsmn_vad_streaming/dynamic_vad.py) maintains its own schedule and calls the underlying VAD with `dynamic_silence=False`. Do not treat this SDK table as the configuration of every server.

### Customization

```python
from funasr import AutoModel

vad = AutoModel(model="fsmn-vad", device="cpu", disable_update=True)
segments = vad.generate(
    input="audio.wav",
    cache={},
    dynamic_silence=True,
    silence_schedule=[(5000, 1500), (20000, 800), (float("inf"), 300)],
)
print(segments[0]["value"])
```

Replace `audio.wav` with an existing recording. This standalone VAD call returns `[start_ms, end_ms]` intervals in `value`, not transcripts. To feed an ASR engine, load/resample audio to its required sample rate, slice by these millisecond offsets, and pass the waveform list as `model.generate(inputs=audio_segments)`. Preserve the offsets when merging results. This example only demonstrates segmentation; it does not automatically connect VAD to vLLM.

> For GLM-ASR, pre-segment and validate duration limits for the selected checkpoint. To request fixed silence, pass `dynamic_silence=False` to the **VAD** call, not the ASR engine. Fixed silence alone does not cap segment duration.

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
Pre-segment long recordings and validate segment lengths with the chosen GLM checkpoint. `dynamic_silence=False` configures the separate FSMN-VAD stage, not `AutoModelVLLM`; disabling dynamic silence alone does not enforce an ASR-safe maximum segment length.

**Q: Performance impact of SPK?**
In the reported offline-service table, RTFx is 102 without SPK and 46 with SPK; CER is 8.14% and 8.19%, respectively. SPK is disabled by default. These measurements do not predict the cost or accuracy of another deployment.

**Q: Entry points for custom development?**
Offline: `serve_vllm.process_audio()` / `FunASRNanoVLLM.generate()`
Streaming: `serve_realtime_ws.RealtimeASRSession`

**Q: Slow first startup?**
Model loading, KV-cache allocation, and optional CUDA Graph warmup contribute to startup time. Measure cold startup and warm inference separately; neither has a fixed duration or an instant-response guarantee.

**Q: What happens when Fun-ASR-Nano vLLM uses `dtype="fp16"`?**
The audio frontend and adaptor remain float16, but FunASR runs the Qwen3 decoder
in bfloat16 because vLLM float16 decoding can produce degraded repeated output.
This is automatic and keeps the two-byte decoder weight footprint. On hardware
without BF16 support, use `dtype="fp32"`; the vLLM path does not claim an
end-to-end FP16 decoder mode.

**Q: vLLM returns repeated punctuation such as `!!!!!!!!` but PyTorch/HF generate is normal. What should I check?**
This usually means the audio frontend and checkpoint can work, but the vLLM
prompt-embedding path or decoding parameters differ from the upstream runner.
Check these items before changing the model:

- Pass prompt embeddings to vLLM as float32:
  pass `input_embeds.float()` to the `prompt_embeds` argument of `EmbedsPrompt`.
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
