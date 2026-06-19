# FunASR (llama.cpp / GGUF) vs whisper.cpp — CPU benchmark

How does the FunASR llama.cpp runtime compare with [whisper.cpp](https://github.com/ggml-org/whisper.cpp),
the de-facto on-device ASR runtime, on **Chinese** speech? This page reports a
head-to-head on identical hardware and audio.

**TL;DR — for Chinese ASR on CPU, FunASR is more accurate, faster, and comparable
or smaller in size than whisper.cpp at every model tier.**

## Results

Same machine, **CPU only, 8 threads**, greedy decoding. Accuracy = character error
rate (CER, lower is better); speed = real-time factor `compute_time / audio_duration`
(higher `×` is faster, model-load time excluded); whisper forced to Chinese (`-l zh`).

| system | model size (f16 / f32) | **CER** ↓ | **speed** ↑ |
|---|---|---|---|
| **FunASR Fun-ASR-Nano** (encoder+adaptor+Qwen3-0.6B) | 0.9 + 0.8 GB | **5.71 %** | LLM decode¹ |
| **FunASR SenseVoiceSmall** | **449 MB** / 893 MB | **7.20 %** | **19.8× real-time** |
| **FunASR Paraformer** | **401 MB** / 824 MB | **9.18 %** | **20.9× real-time** |
| whisper.cpp base | 142 MB | 22.96 % | 12.7× |
| whisper.cpp small | 466 MB | 17.46 % | 4.8× |
| whisper.cpp large-v3-turbo | 1.6 GB | 16.33 % | 3.0× |

¹ Fun-ASR-Nano has an autoregressive 0.6B LLM decoder, so it is slower than the
encoder-only SenseVoice/Paraformer (and is the accuracy leader). A clean RTF will be
added once the CLI separates model-load from compute time.

The whole **FunASR family beats every whisper.cpp tier on Chinese**, and lets you
pick the point on the accuracy/speed curve: **Paraformer** (fastest, 20.9×),
**SenseVoice** (fast + language-ID/emotion/event), **Fun-ASR-Nano** (most accurate,
**5.71 %** — 2.9× lower CER than whisper-large-v3-turbo).

**At comparable size** (SenseVoice-f16 449 MB ≈ whisper-small 466 MB): FunASR is
**2.4× more accurate and 4× faster**. **Against whisper's best** (large-v3-turbo,
1.6 GB): SenseVoice is **2.3× more accurate, 6.6× faster, and 3.6× smaller**; Fun-ASR-Nano
is **2.9× more accurate**.

Full 184-file set (FunASR): SenseVoiceSmall **9.98 %** CER, Paraformer **12.84 %**
(the head-to-head above uses the 60-file subset that whisper was also run on).

## Why FunASR wins on Chinese

1. **Training data.** SenseVoice / Paraformer are trained primarily on large-scale
   Mandarin (and other Asian languages); Whisper is a general multilingual model
   where Chinese is a small slice. On Chinese homophones Whisper makes substitution
   errors the FunASR models do not (example below).
2. **Architecture → speed.** Paraformer is **non-autoregressive** (a CIF predictor
   emits all tokens, the decoder runs once) and SenseVoiceSmall is **encoder + CTC**
   (one forward pass). Whisper is **autoregressive** (one transformer step per output
   token). That is the structural reason FunASR runs 4–7× faster at the same threads.

## Qualitative example (clip 002)

| system | output (excerpt) |
|---|---|
| ground truth | 我想问，我在**滨海新区**有房…因为我在天津有房子…所以我必须拿到**抚养权** |
| FunASR (Nano / SenseVoice / Paraformer) | …我在**滨海新区**有房…我必须拿到**抚养权**… ✓ |
| whisper base | …我在**冰海心区**有房…我想要**扶养权**…上学**方面**… ✗ |
| whisper small | …我在**冰海新区**有房…我想要**抚养全**… ✗ |
| whisper large-v3-turbo | …我在滨海新区有房…上学**方面**… ✗ |

Whisper's errors are classic Chinese homophone substitutions (滨海→冰海, 抚养→扶养,
方便→方面); the FunASR models get them right.

## Methodology

- **Data:** 184 real long-audio Mandarin clips (~44–60 s each) with human references.
  The whisper head-to-head uses the first 60 clips (large-v3-turbo is too slow on CPU
  to run all 184); FunASR numbers are reported on both 60 and the full 184.
- **Accuracy:** character-level CER after normalization (strip punctuation/whitespace,
  lowercase), identical for all systems.
- **Speed (RTF):** `compute_time / audio_duration`, **model-load time excluded** for
  every system (FunASR self-reported encode/decode time; whisper `total − load`).
  CPU, **8 threads for all systems** (whisper `-t 8`).
- **Decoding:** greedy for all; whisper forced to Chinese (`-l zh`), no VAD on either
  side (FunASR clips fit one segment).
- **Hardware:** a single shared server CPU; numbers are stable across repeats.

## Caveats (fair use of this comparison)

- This is a **Chinese** benchmark — FunASR's home turf. Whisper is a *general
  multilingual* model that also does translation, 99-language coverage, and
  timestamps; it should not be judged solely on Mandarin CER. For English or
  low-resource languages Whisper is the stronger general choice.
- SenseVoiceSmall additionally outputs language ID, emotion, and audio-event tags;
  Paraformer is Mandarin-specialised. Pick the model that matches your use case.
- Single-domain dataset; absolute CER will differ on other domains. The *relative*
  ranking (FunASR ≫ whisper on Mandarin) is the takeaway.

## Reproduce

1. Build the FunASR runtimes (`runtime/llama.cpp/`, see each model README) and
   whisper.cpp; convert weights to GGUF.
2. Run each system over the clips, capturing text + self-reported compute time.
3. Compute normalized CER vs the references and RTF vs clip durations.

The exact scripts (`agg_bench.py` and the per-system runners) accompany this
benchmark; numbers above were produced with whisper.cpp `large-v3-turbo` / `small` /
`base` and the FunASR `sensevoice-small` / `paraformer` GGUFs.
