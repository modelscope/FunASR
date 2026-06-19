# Fun-ASR-Nano on llama.cpp / GGUF

Run **Fun-ASR-Nano** entirely on the [llama.cpp](https://github.com/ggml-org/llama.cpp)
/ ggml stack — **CPU, edge, a single binary, no Python at runtime**. This is to
Fun-ASR what [whisper.cpp](https://github.com/ggml-org/whisper.cpp) is to Whisper.

## Why this exists

Fun-ASR-Nano normally runs on PyTorch / vLLM (GPU). That is great for a server
serving many requests, but it cannot run where there is no GPU and no Python.
This runtime ports the model to **ggml + GGUF**, so Fun-ASR-Nano can run:

- on a laptop / phone / Raspberry Pi / edge box, offline, CPU-only;
- embedded directly into a C/C++ application (one static binary);
- with quantized weights (Q8 / Q4), shrinking the model to ~1.3 GB total.

| | vLLM (existing) | this runtime (llama.cpp) |
|---|---|---|
| target | GPU server, high QPS | CPU / edge / embedded |
| deps | Python + CUDA + PyTorch | none (C/C++ single binary) |
| weights | HF fp16/bf16 | GGUF, quantized |
| best for | online service, batch | offline, on-device |

## Architecture

Fun-ASR-Nano = **SenseVoice SAN-M encoder (70 layers) + adaptor + Qwen3-0.6B LLM**.
The whole pipeline runs in C++:

```
 audio.wav (16k mono)
      │  kaldi 80-mel fbank + LFR            (C++)
      ▼
   features [T, 560]
      │  SAN-M encoder + adaptor             (ggml)   ── funasr-encoder.gguf
      ▼
   audio embeds [T', 1024]
      │  keep first fake_token_len frames (low-frame-rate)
      ▼
 [ prefix tokens | audio embeds | suffix tokens ]
      │  Qwen3-0.6B, embeds injected via llama_decode (llava/mtmd style)  ── qwen3-0.6b.gguf
      ▼
   transcription
```

The audio embeddings are fed into the LLM through `llama_decode`'s embedding-input
path — exactly how llava/mtmd inject vision embeddings.

## Quickstart

**1. Build** (drop the examples into a llama.cpp checkout):
```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cp -r /path/to/runtime/llama.cpp/funasr-cli examples/
echo 'add_subdirectory(funasr-cli)' >> examples/CMakeLists.txt
cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build -j --target llama-funasr-cli
```

**2. Convert weights to GGUF** (one-time; needs the checkpoint, e.g.
`FunAudioLLM/Fun-ASR-Nano-2512`):
```bash
# LLM half — Qwen3-0.6B is natively supported by llama.cpp
python llama.cpp/convert_hf_to_gguf.py <model>/Qwen3-0.6B-vllm \
    --outfile qwen3-0.6b-f32.gguf --outtype f32
build/bin/llama-quantize qwen3-0.6b-f32.gguf qwen3-0.6b-q8_0.gguf Q8_0   # smaller, recommended

# audio half — SenseVoice encoder + adaptor
python runtime/llama.cpp/export_encoder_gguf.py \
    --model_pt <model>/model.pt --out funasr-encoder.gguf              # f32, 935 MB
python runtime/llama.cpp/export_encoder_gguf.py \
    --model_pt <model>/model.pt --out funasr-encoder-f16.gguf --wtype f16   # 469 MB
```

**3. Transcribe:**
```bash
build/bin/llama-funasr-cli \
    --enc funasr-encoder.gguf -m qwen3-0.6b-q8_0.gguf \
    -a audio.wav --chunk 15
```
Expected output (one of the benchmark clips):
```
我想问我在滨海新区有房我一直没有照顾孩子但是我想要抚养权...你觉得这是正常的想法吗
[done] 7.40s ; chunk=15s
```

## Models & sizes

| file | dtype | size |
|---|---|---|
| funasr-encoder.gguf | f32 | 935 MB |
| funasr-encoder-f16.gguf | f16 (matmul weights) | 469 MB |
| qwen3-0.6b-f32.gguf | f32 | 3.0 GB |
| qwen3-0.6b-q8_0.gguf | Q8_0 | 805 MB |
| qwen3-0.6b-q4km.gguf | Q4_K_M | 484 MB |

Fully-quantized config (f16 encoder + Q8 LLM) ≈ **1.3 GB**, edge-friendly.

## Accuracy & validation

Validated against the PyTorch reference on the 184-file benchmark:

- **Encoder + adaptor (ggml) vs PyTorch:** cosine **1.000000**, max_abs_diff **5e-3** (f32).
- **kaldi fbank (C++) vs torchaudio:** cosine **1.000000**.
- **End-to-end CER, identical conditions (f32 LLM, 15 s chunking):**
  C++ macro 17.41% / micro 11.68%  vs  PyTorch macro 17.42% / micro 11.70%
  → aggregate CER matches to **0.02%**; the port is faithful.
- Best practical config (Q8 LLM + 15 s chunking): **micro-CER 9.51%** (production
  VAD-segmented reference is ~8.2%; the gap is fixed-window vs VAD, not the port).

## Tips & gotchas

- **Use `--chunk 15`** for long audio. Decoding a whole 60 s clip in one segment is
  out-of-distribution and makes greedy decoding loop; 15 s windows fix it
  (micro-CER 29% → 9.5%).
- **Low-frame-rate truncation** is required: only the first `fake_token_len`
  adaptor frames are real audio tokens. The CLI does this automatically; feeding
  all frames makes the LLM repeat.
- **Use bf16/fp32, avoid fp16 for the audio path** — the adaptor output has large
  magnitude (std ≈ 28, |max| ≈ 1187); fp16 can overflow. The GGUFs here are f32/f16
  weights with f32 activations, which is safe.
- **WAV input** currently assumes 16 kHz mono PCM16. Resample first if needed.
- Q8 quantization slightly *helps* greedy stability (quant noise regularizes away
  from repetition loops), so Q8 is a good default.

## Implementation notes

- FSMN depthwise memory is an exact f32 shift-accumulate (avoids the F16-only,
  upstream-flagged `ggml_conv_1d_dw`).
- LayerNorm eps = 1e-5; sinusoidal position encoding depth = input feature dim (560),
  positions start at 1; encoder input pre-scaled by sqrt(512).
- Prompt is fed as tokens via `llama_tokenize(parse_special=true)` (prefix = 18
  tokens, matching the HF tokenizer), so no Python embedding table is needed.

## Files
```
funasr-cli/        integrated binary: WAV → transcription
funasr-encoder/    encoder+adaptor only (ggml) — validation/debugging
funasr-embd/       LLM decode from precomputed embeds — validation/debugging
export_encoder_gguf.py   export the audio encoder + adaptor to GGUF
```

## Roadmap
- True FSMN-VAD segmentation (replace fixed windows; closes the last ~1.3% CER).
- Arbitrary WAV formats / resampling; encoder Q8 quantization; single packaged GGUF.
