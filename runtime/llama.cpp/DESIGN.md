# FunASR on llama.cpp / GGUF — Design Document

This document describes the design of the `runtime/llama.cpp` directory: a C++ /
ggml runtime that runs FunASR models (Fun-ASR-Nano, SenseVoiceSmall, Paraformer)
without PyTorch, on CPU and edge devices, with quantized GGUF weights. It is the
counterpart of [whisper.cpp](https://github.com/ggml-org/whisper.cpp) for FunASR.

It is written to be read without the source: it explains *why* the runtime exists,
*how* each model maps onto ggml, the GGUF weight format, the numerical-fidelity and
validation methodology, the non-obvious gotchas discovered during the port, and the
roadmap.

> This is the **shared design document** for the FunASR-on-llama.cpp effort and is
> kept identical across the FunASR family repos (modelscope/FunASR, Fun-ASR,
> SenseVoice). The three models share one ggml SAN-M encoder / FSMN / fbank
> foundation, so the design is documented once here in full; a single-model repo
> ships only the relevant model directory (§2) but the shared design still applies.

---

## 1. Motivation

FunASR's reference inference runs on PyTorch (and vLLM for the LLM-based models) on
GPU. That is the right tool for a server that batches many requests and wants to
saturate a GPU. It is the wrong tool when there is **no GPU and no Python**: a
laptop, a phone, a Raspberry Pi, an embedded C/C++ application, an offline desktop
app. There, you want a single self-contained binary, a few hundred MB of quantized
weights, and CPU SIMD.

[llama.cpp](https://github.com/ggml-org/llama.cpp) / ggml is the de-facto runtime
for that world (Ollama, LM Studio, whisper.cpp all build on it). Porting FunASR to
ggml + GGUF makes FunASR run anywhere llama.cpp runs, dramatically widening the
deployment surface.

| | PyTorch / vLLM (existing) | this runtime (llama.cpp) |
|---|---|---|
| target | GPU server, high QPS | CPU / edge / embedded |
| deps | Python + CUDA + PyTorch | none (C/C++ single binary) |
| weights | HF fp16/bf16 safetensors | GGUF, 2–8 bit quantization |
| key tech | PagedAttention, continuous batching | quantization, mmap, CPU SIMD |
| best for | online service, batch eval | offline, on-device, embedded |

These are complementary, not competing: cloud serving stays on vLLM; this runtime
covers the on-device / offline case.

---

## 2. System overview

Three models are supported. They share more than they differ — all three use the
same **SAN-M encoder**, the same **FSMN memory block**, the same **kaldi-compatible
fbank front end**, and the same ggml building blocks.

```
                            ┌─────────────────────── shared C++ / ggml ───────────────────────┐
 audio.wav (16k mono) ──►  kaldi 80-mel fbank + LFR(7/6)  ──►  SAN-M encoder (50 layers, ggml)
                            └──────────────────────────────────────────────────────────────────┘
                                                  │ encoder_out [T, 512]
              ┌───────────────────────────────────┼───────────────────────────────────┐
   Fun-ASR-Nano                           SenseVoiceSmall                          Paraformer
   adaptor → audio embeds                 + 4 query tokens                         CIF predictor (host)
   → inject into Qwen3-0.6B               CTC head → greedy CTC                    → SAN-M decoder (cross-attn)
   (llama_decode embd path)               → SentencePiece                          → argmax → tokens.json
   → text                                 → text                                   → text
```

| model | head / decoder | autoregressive? | output units |
|---|---|---|---|
| Fun-ASR-Nano | adaptor + Qwen3-0.6B LLM | yes (LLM) | Qwen3 BPE |
| SenseVoiceSmall | CTC | no | spectok BPE (25055) |
| Paraformer | CIF + SAN-M decoder | no (parallel) | char/BPE (8404) |

Directory layout:
```
runtime/llama.cpp/
  README.md            overview
  DESIGN.md            this document
  fun-asr-nano/        funasr-cli, funasr-encoder, funasr-embd, export_encoder_gguf.py
  sensevoice/          funasr-sensevoice, export_sensevoice_gguf.py, detok.py
  paraformer/          funasr-paraformer, export_paraformer_gguf.py, detok_paraformer.py
```
Each model dir holds the llama.cpp example sources (drop-in under `examples/`), a
GGUF export script, and a model-specific README.

---

## 3. Audio front end (kaldi fbank in C++)

All models use FunASR's `WavFrontend`: kaldi-compatible 80-bin log-mel fbank with a
hamming window (25 ms / 10 ms), pre-emphasis 0.97, DC removal, 512-pt FFT, then
**Low-Frame-Rate (LFR)** stacking of 7 frames with stride 6 → a 560-dim feature
per output frame.

The C++ implementation (`compute_fbank`) reproduces this exactly:
1. upscale the waveform by 32768 (FunASR feeds int16-range samples to kaldi),
2. per frame: remove DC offset, pre-emphasis, hamming window, zero-pad to 512,
3. radix-2 FFT, power spectrum, 80 triangular mel filters (kaldi mel: `1127·ln(1+f/700)`,
   low 20 Hz, high 8000 Hz), log floor `FLT_EPSILON`,
4. LFR: left-pad 3 copies of frame 0, stack 7 frames stride 6 → 560-dim.

**Validation:** vs torchaudio kaldi.fbank (dither=0), cosine **1.000000**,
max_abs_diff 1.75e-3.

**Gotcha — dither.** FunASR's frontend uses `dither=1.0` by default, which adds
random noise per sample, so the fbank (and everything downstream) is *non-deterministic*
in the reference. The C++ front end uses dither=0 (deterministic). The model is
robust to this; it accounts for the small (<1%) cosine gap seen when comparing
against a dithered reference.

---

## 4. The SAN-M encoder in ggml

The SenseVoice/Paraformer encoder is a 50-layer (Paraformer) or 50+20-layer
(SenseVoice, with extra `tp_encoders`) **SAN-M** stack. Each layer is pre-norm:

```
x → LN → SAN-M self-attention → +residual → LN → FFN(relu) → +residual
```

SAN-M self-attention = standard multi-head attention **plus** an FSMN memory branch
that runs in parallel on the value projection and is added to the attention output:

```
q,k,v = split(linear_q_k_v(x))           # one fused projection
fsmn  = FSMN(v)                           # depthwise conv over time + residual
attn  = softmax(qkᵀ/√d)·v → linear_out
out   = attn + fsmn
```

### 4.1 FSMN as an exact f32 shift-accumulate (design decision)

FSMN is a per-channel (depthwise) 1-D convolution over time with a symmetric
kernel (size 11). ggml has `ggml_conv_1d_dw`, but it (a) requires the kernel in
F16 and (b) is flagged as "very likely wrong for some cases" upstream. Both are
unacceptable for a faithful port.

Instead FSMN is implemented as an **exact f32 shift-accumulate**: the kernel is
exported as `[K, D]`, the value tensor is zero-padded by `(K-1)/2` on each side
along time, and the output is `Σ_j kernel[:,j] ⊙ pad(v)[:, t+j]` plus the residual.
This is 11 element-wise multiply-adds — exact in f32, no F16 rounding, no dependence
on the questionable conv kernel. It dropped the full-encoder max_abs_diff vs PyTorch
from 2.93 to **0.0052**.

### 4.2 Position encoding & input scaling

Input is pre-scaled by `√(d_model)=√512` then a sinusoidal position encoding is
added, with **depth = the input feature dim (560)** and **positions starting at 1**
(not 0) — both quirks of the FunASR encoder that must be matched exactly.

### 4.3 LayerNorm

eps = 1e-5 everywhere.

**Validation:** first layer cosine 1.0 (max_abs_diff 1.8e-4); full encoder cosine
**1.000000**, max_abs_diff 5.2e-3 (f32).

---

## 5. Per-model design

### 5.1 Fun-ASR-Nano (encoder + adaptor + LLM)

Pipeline: `fbank → encoder → adaptor → audio embeds [T', 1024] → inject into
Qwen3-0.6B → text`.

- **LLM half is native.** Qwen3 is supported by llama.cpp, so the extracted
  Qwen3-0.6B converts to GGUF with the stock `convert_hf_to_gguf.py` and runs
  unchanged.
- **Embedding injection.** The audio embeddings are fed into the LLM through
  `llama_decode`'s embedding-input path — exactly how llava/mtmd inject vision
  embeddings. The integrated CLI builds the prompt as a *mixed* sequence:
  `[prefix tokens | audio embeds | suffix tokens]`, where prefix/suffix are fed as
  token ids (llama.cpp embeds them internally; `llama_tokenize(parse_special=true)`
  reproduces the exact 18-token prefix) and the audio slot is fed as embeddings.
- **Low-frame-rate truncation (critical).** The adaptor emits `T'` frames, but the
  model only uses the first `fake_token_len` of them as audio tokens, where
  `fake_token_len` derives from the fbank length by a 3-stage `÷2` formula
  (≈ T'/8). Feeding all `T'` frames is out-of-distribution and makes the LLM loop.
- **Chunking.** Decoding a long (e.g. 60 s) clip as one segment is OOD and triggers
  greedy repetition; the CLI's `--chunk 15` splits into windows with a fresh KV per
  window, dropping micro-CER from ~29% to ~9.5%.
- **Numerics.** The adaptor output has large magnitude (std ≈ 28, |max| ≈ 1187), so
  fp16 can overflow; the runtime uses f32/f16 weights with f32 activations.

### 5.2 SenseVoiceSmall (encoder + CTC)

Pipeline: `fbank → prepend 4 query tokens → encoder → CTC head → greedy CTC →
SentencePiece`.

- **Query tokens.** Four learned embeddings are prepended: `[language(auto), event,
  emotion, textnorm]` (indices `[0,1,2,15]` for auto/woitn). They are 560-dim and
  prepended *before* the encoder's `√512` scaling and position encoding.
- **CTC decode.** `argmax → collapse consecutive → drop blank(0)` → ids → SentencePiece.
- **Gotcha — no CMVN at inference.** SenseVoice's `inference()` feeds the **raw**
  log-mel fbank to the encoder; it does **not** apply `am.mvn` CMVN (that code path
  is unused at inference). Applying CMVN makes the model predict `<|nospeech|>`.

**Validation:** CTC token ids **identical** to PyTorch (108/108 on a clip); text
matches `AutoModel` exactly.

### 5.3 Paraformer (encoder + CIF + decoder, non-autoregressive)

Pipeline: `fbank → CMVN → encoder → CIF predictor → acoustic embeds [N, 512] →
SAN-M decoder (cross-attn to encoder) → argmax → tokens.json`.

- **CMVN IS applied** here (unlike SenseVoice): `(fbank + shift)·scale`, per-dim
  (560), from `am.mvn`.
- **CIF predictor (runs on host).** Continuous Integrate-and-Fire: a 1-D conv
  (k=3) + residual + relu + linear → sigmoid → per-frame weight α; then a sequential
  integrate-and-fire loop emits one acoustic embedding each time the running α-sum
  crosses 1.0. This both decides the token count and produces the decoder input. It
  is inherently sequential, so it runs in plain C++ (cheap: ~0.5 G MACs); the
  encoder and decoder run in ggml.
- **SAN-M decoder (ggml).** 16 layers, each: `FFN → FSMN self-attention →
  cross-attention to the encoder output`. The self-attention is **FSMN-only** (no
  QK attention); cross-attention has q from the decoder slots and k,v from a fused
  `linear_k_v` of the encoder output. A 17th `decoders3` layer is FFN-only. The
  decoder FFN has an internal LayerNorm and the second linear has no bias. The
  layer ordering (FFN *before* the attention inside the residual) is unusual and is
  matched exactly.

**Validation:** decoded text **identical** to `AutoModel`; CIF token count exact
(105/105). Encoder cosine 0.997 (residual is the reference's random dither).

**Gotcha — `am.mvn` has three bracketed blocks.** `[Splice idx]`, `[AddShift=shift]`,
`[Rescale=scale]`. The shift/scale are the two 560-length vectors; naively taking
the first two blocks grabs `[0]` as the shift and mis-scales everything, which makes
CIF emit ~4× too few tokens. Parse by length.

---

## 6. GGUF conversion & weight layout

Each model has an `export_*_gguf.py` that packs weights + architecture metadata into
a single GGUF.

- Tensor names are kept verbatim from the checkpoint (e.g.
  `encoder.encoders.3.norm1.weight`); the C++ looks them up by name.
- **FSMN kernels** are transposed from `(D,1,K)` to `[K,D]` at export so the C++
  shift-accumulate can take a contiguous per-tap `[D]` vector.
- **CMVN** (`am.mvn`) is parsed to `cmvn.shift` / `cmvn.scale` tensors (Paraformer
  uses them; SenseVoice ships them but the runtime ignores them).
- **Quantization.** `--wtype f16` stores the 2-D matmul weights as F16 (norms,
  biases, FSMN kernels stay f32), halving the encoder GGUF (e.g. 935 → 469 MB) with
  cosine 0.999999. The Qwen3 LLM uses the standard llama.cpp quantizer (Q8_0 / Q4_K_M).

| file | model | dtype | size |
|---|---|---|---|
| funasr-encoder.gguf | Nano | f32 / f16 | 935 / 469 MB |
| qwen3-0.6b-q8_0.gguf | Nano LLM | Q8_0 | 805 MB |
| sensevoice-small.gguf | SenseVoice | f32 | 936 MB |
| paraformer.gguf | Paraformer | f32 | 863 MB |

---

## 7. Numerical fidelity & validation methodology

The port is validated **stage by stage** against the PyTorch reference, using golden
dumps (fbank, encoder output, adaptor/CIF output, logits/ids) compared by cosine
similarity and max-abs-diff, then end-to-end by transcription text / CER.

Summary of results (benchmark clip / set):

| stage | metric |
|---|---|
| kaldi fbank vs torchaudio | cosine 1.000000 |
| SAN-M encoder (full) vs PyTorch | cosine 1.000000, max_abs_diff 5e-3 (f32) |
| SenseVoice CTC ids | identical (108/108) |
| Paraformer text / token count | identical / 105 = 105 |
| Fun-ASR-Nano end-to-end CER (same conditions) | C++ 11.68% vs PyTorch 11.70% (Δ0.02%) |

**Why not bit-exact tokens everywhere?** Greedy decoding is chaotic: a ~5e-3
difference (from ggml-CPU vs torch-GPU matmul summation order) can flip a token on
a borderline frame, and over a long sequence the paths diverge — *this also happens
between PyTorch's own GPU and CPU*. What is faithful and what we verify is (a) the
per-tensor numerics (cosine 1.0) and (b) the **aggregate CER**, which matches the
reference under identical conditions.

**fp16 caution.** The Fun-ASR-Nano adaptor output magnitude (std ≈ 28) can overflow
fp16; the audio path is kept in f32 (weights may be f16, activations f32).

---

## 8. Performance

CPU, 8 threads, a 44 s clip:
- Encoder (50 layers): ~1.2 s.  Paraformer decoder: ~0.5 s.
- Fun-ASR-Nano end-to-end (with LLM): ~7 s.
- Fully-quantized footprint (f16 encoder + Q8 LLM) ≈ 1.3 GB.

These are first-correctness numbers; quantizing the encoder and threading/batching
the front end are open optimizations.

---

## 9. Design decisions & trade-offs

- **ggml for encoder/decoder, host C++ for CIF.** The neural matmul-heavy parts run
  in ggml (SIMD, future GPU backends); CIF is a sequential scalar loop with data-
  dependent control flow, so it is clearer and not slower in plain C++.
- **Exact f32 FSMN instead of `ggml_conv_1d_dw`.** Correctness and f32 precision
  over reusing a flagged, F16-only op (§4.1).
- **Prompt as tokens, not a Python embedding table.** The integrated CLI tokenizes
  the prompt with `llama_tokenize` and lets llama.cpp embed it, so no embedding
  matrix needs to be shipped or matched (Fun-ASR-Nano).
- **f32 by default, f16/Q8 opt-in.** f32 is the faithful default; quantization is a
  size/latency lever the user opts into. (Interestingly, Q8 on the LLM slightly
  *helps* greedy stability by regularizing away from repetition loops.)
- **Per-model self-contained example dirs.** Mirrors llama.cpp's `examples/` layout
  so each builds as a drop-in target; the shared code is duplicated rather than
  factored to keep each example independently buildable.

---

## 10. Limitations & roadmap

- **WAV input** assumes 16 kHz mono PCM16; arbitrary formats / resampling are TODO.
- **VAD.** Long audio needs segmentation; today Fun-ASR-Nano uses fixed `--chunk`
  windows. A real FSMN-VAD front end would close the last ~1.3% CER gap to the
  production VAD-segmented number and is the highest-value next step.
- **Single packaged GGUF** (encoder + adaptor + LLM in one file) and a one-command
  converter.
- **Encoder/decoder quantization** (Q8 via gguf-py quants), streaming, timestamps
  (Paraformer CIF peaks give alignment; SenseVoice/Nano via CTC).
- **Upstream.** The example sources are drop-in for llama.cpp; upstreaming the
  runtime to ggml-org/llama.cpp (as whisper.cpp-style tools) is a separate track.

---

## 11. Reproducing the validation

Each model dir's README has the build + convert + run quickstart. The export
scripts read a standard FunASR checkpoint (`model.pt` + `config.yaml` + `am.mvn` /
tokenizer). To reproduce a stage comparison, dump the corresponding PyTorch tensor
(`model.encode`, `model.calc_predictor`, …) and compare with cosine / max-abs-diff;
the numbers in §7 should reproduce within dither noise.
