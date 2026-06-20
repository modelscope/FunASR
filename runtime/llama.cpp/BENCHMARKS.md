# FunASR (llama.cpp / GGUF) vs whisper.cpp — CPU benchmark

How does the FunASR llama.cpp runtime compare with [whisper.cpp](https://github.com/ggml-org/whisper.cpp),
the de-facto on-device ASR runtime, on **Chinese** speech? This page reports a
head-to-head on identical hardware and audio.

**TL;DR — for Chinese ASR on CPU, FunASR is ~2.7× more accurate than whisper.cpp at
every model tier, and faster.**

## Results

Dataset: **184 real Mandarin clips with human references** (the standard FunASR
benchmark set). Metric: **micro-CER** with `normalize_zh` (lower is better). Speed:
real-time factor on **CPU, 8 threads** (model-load excluded). whisper forced to
Chinese (`-l zh`).

| system | **CER** (micro, normalize_zh) ↓ | speed ↑ | size |
|---|---|---|---|
| **FunASR Fun-ASR-Nano** | **8.06** (fp32 ref) / **8.42** (Q8 runtime) | LLM decode¹ | enc + Qwen3-0.6B GGUF |
| **FunASR SenseVoiceSmall** | **7.81** (fp32 ref) / **8.17** (Q8 runtime) | **~20× real-time** | 449 MB (f16) |
| **FunASR Paraformer** | **10.18** (fp32 ref) / **9.89** (Q8 runtime) | **~21× real-time** | 401 MB (f16) |
| whisper.cpp base | 31.33 | 9.9× | 142 MB |
| whisper.cpp small | 22.12 | 4.6× | 466 MB |
| whisper.cpp large-v3-turbo | 23.15 | 3.2× | 1.6 GB |

**Each FunASR row shows two numbers:** the published **fp32 reference** (PyTorch,
the number on funasr.com / the model cards) and the **Q8 llama.cpp CPU runtime**
measured here. The ~0.3 % gap is normal int8 quantization + VAD segment boundaries;
Q8 is the real CPU/edge deployment config. Either way, **FunASR ~8–10 % vs
whisper.cpp 22–31 % — a 2.7×+ accuracy gap that holds at every tier.**

¹ Fun-ASR-Nano runs an autoregressive 0.6B LLM decoder (slower than the encoder-only
SenseVoice/Paraformer; it is the accuracy leader). A clean RTF lands once the CLI
separates model-load from compute.

### Transparency / segmentation (read this before quoting numbers)

- **Segmentation differs by system, each using its natural strategy:** FunASR uses an
  `fsmn-vad` front end (segments → ASR → concatenate); whisper.cpp uses its built-in
  30 s windowing. This is a fair system-level comparison.
- **Engine-internal VAD is now implemented** — a native ggml FSMN-VAD built into the
  binaries (`--vad fsmn-vad.gguf`). The **bare binary, with no Python front end**, now
  reaches the reference end-to-end: SenseVoiceSmall **8.01 %**, Paraformer **9.85 %**,
  Fun-ASR-Nano **8.30 %** (micro, normalize_zh, full 184). The built-in C++ VAD matches
  the PyTorch `fsmn-vad` front end (segment boundaries within ~10 ms, slightly better
  CER), so the runtime is now fully self-contained.
- For full disclosure, **bare binary with no VAD at all (whole-clip)** is higher —
  SenseVoiceSmall 9.99 %, Paraformer 12.82 % — because long clips decoded as one segment
  are out-of-distribution; that is exactly what the built-in VAD fixes.

## Why FunASR wins on Chinese

1. **Training data.** SenseVoice / Paraformer / Fun-ASR-Nano are trained primarily on
   large-scale Mandarin; Whisper is a general multilingual model where Chinese is a
   small slice. On Chinese homophones Whisper makes substitution errors the FunASR
   models do not (example below).
2. **Architecture → speed.** Paraformer is non-autoregressive (CIF predictor + one
   decoder pass) and SenseVoiceSmall is encoder + CTC (one forward pass); Whisper is
   autoregressive (one step per output token).

## Qualitative example (clip 002)

| system | output (excerpt) |
|---|---|
| ground truth | 我想问，我在**滨海新区**有房…所以我必须拿到**抚养权** |
| FunASR (Nano / SenseVoice / Paraformer) | …我在**滨海新区**有房…拿到**抚养权**… ✓ |
| whisper base | …我在**冰海心区**有房…我想要**扶养权**…上学**方面**… ✗ |
| whisper small | …我在**冰海新区**有房…我想要**抚养全**… ✗ |
| whisper large-v3-turbo | …滨海新区…上学**方面**… ✗ |

## Methodology

- **Data:** the standard 184-clip Mandarin benchmark set (`benchmark/testset.json`),
  ~44–60 s each, with human references.
- **Metric (canonical):** **micro-average CER** (`Σ edits / Σ ref chars`) after
  **`normalize_zh`**: `re.sub(r'[^\w一-鿿]', '', text).upper()` (strip punctuation/
  whitespace, keep word chars + CJK, upper-case; SenseVoice `<|...|>` tags stripped).
  This is the canonical FunASR口径 — the same one behind the published fp32 numbers.
  (A macro-average / simplified-normalize variant gives different, non-canonical
  numbers; it is not used here.)
- **FunASR fp32 reference:** PyTorch, micro + normalize_zh, 184 set — SenseVoice 7.81,
  Paraformer 10.18, Fun-ASR-Nano 8.06 (matches funasr.com / READMEs / model cards).
- **FunASR Q8 runtime:** this llama.cpp runtime (Q8 LLM / f16 encoder) + `fsmn-vad`
  front end (`max_single_segment_time=30000`), full 184. SenseVoice uses `use_itn=True`
  to match the reference.
- **whisper.cpp:** ggml `base` / `small` / `large-v3-turbo`, `-l zh`, internal 30 s
  windowing, full 184.
- **Speed (RTF):** `Σ compute_time / Σ audio_duration`, model-load excluded, **8 threads
  for all systems**.

## Caveats (fair use)

- This is a **Chinese** benchmark — FunASR's home turf. Whisper is a *general
  multilingual* model (translation, 99 languages, timestamps); for English / other
  languages it is the stronger general choice. The takeaway is specifically:
  **for Chinese ASR on CPU, FunASR is the accuracy + speed leader.**
- SenseVoiceSmall also outputs language ID / emotion / audio-event; Paraformer is
  Mandarin-specialised; Fun-ASR-Nano is the most accurate (LLM decoder). Pick per use case.

## Reproduce

See [`benchmarks/`](benchmarks/) — `compute_cer.py` (micro-CER + normalize_zh + RTF)
and the per-system run commands. Produce hypotheses with each tool, then compute CER
against the references and RTF against clip durations.
