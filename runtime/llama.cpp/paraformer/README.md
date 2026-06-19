# Paraformer on llama.cpp / GGUF

Run **Paraformer** (the non-autoregressive ASR model) on the
[llama.cpp](https://github.com/ggml-org/llama.cpp) / ggml stack — **CPU, edge,
a single binary, no Python at runtime**. Like whisper.cpp, but for Paraformer.

## Why this exists

Paraformer normally runs on PyTorch / ONNX. This runtime ports it to **ggml +
GGUF** so it runs CPU-only, offline, embedded in a C/C++ app, with quantized
weights — laptops, phones, edge boxes, no GPU and no Python. (For high-QPS GPU
serving, the PyTorch path is still the way.)

## Architecture

Paraformer is **non-autoregressive**: it predicts all output tokens in one pass.

```
 audio.wav (16k mono)
      │  kaldi 80-mel fbank + LFR + CMVN                        (C++)
      ▼
   features [T, 560]
      │  SANM encoder (50 layers: LN + fused QKV + FSMN + FFN)  (ggml)
      ▼
   encoder_out [T, 512]
      │  CIF predictor: conv1d → sigmoid → α; integrate-and-fire (host)
      ▼
   acoustic embeds [N_tok, 512]   (N_tok = number of output tokens)
      │  SANM decoder (16 layers: FFN → FSMN self-attn → cross-attn to encoder)  (ggml)
      ▼
   logits [N_tok, vocab] → argmax → token ids → text
```

CIF (Continuous Integrate-and-Fire) walks the encoder output accumulating a
predicted "weight" α per frame; each time the running sum crosses 1.0 it emits one
acoustic token. This both decides the token count and produces the acoustic
embeddings the decoder consumes. The SANM encoder/FSMN/attention primitives are
shared with the Fun-ASR-Nano and SenseVoice runtimes.

## Quickstart

**1. Build:**
```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cp -r /path/to/runtime/llama.cpp/funasr-paraformer examples/
echo 'add_subdirectory(funasr-paraformer)' >> examples/CMakeLists.txt
cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build -j --target llama-funasr-paraformer
```

**2. Convert weights** (needs the checkpoint, e.g. `funasr/paraformer-zh`):
```bash
python runtime/llama.cpp/export_paraformer_gguf.py \
    --model_pt <model>/model.pt --mvn <model>/am.mvn \
    --out paraformer.gguf                          # f32, ~863 MB
python runtime/llama.cpp/export_paraformer_gguf.py --wtype f16 \
    --model_pt <model>/model.pt --mvn <model>/am.mvn \
    --out paraformer-f16.gguf                       # half size
```

**3. Transcribe:**
```bash
build/bin/llama-funasr-paraformer -m paraformer.gguf -a audio.wav > ids.txt
python runtime/llama.cpp/detok.py <model>/tokens.json ids.txt
```
Expected output:
```
我想问我在滨海新区有房我一直没有照顾孩子...你觉得这是正常的想法吗
[paraformer] T=742 N_tok=105 enc 1.24s dec 0.48s
```

## Accuracy & validation

- Decoded text is **character-for-character identical** to the FunASR `AutoModel`
  output on a benchmark clip; the CIF token count matches exactly (105/105).
- Stage-by-stage vs PyTorch: encoder cosine 0.997, acoustic embeds cosine 0.993
  (the small residual is the reference frontend's random `dither=1.0`; the C++
  front end is deterministic, dither=0).
- Encode ≈ 1.2 s + decode ≈ 0.5 s on CPU for a 44 s clip.

## Tips & gotchas

- **CMVN IS applied** (unlike SenseVoice): `(fbank + shift) * scale`, per-dim (560),
  from `am.mvn`. Parsing note: `am.mvn` has three `[...]` blocks — `[Splice idx]`,
  `[AddShift=shift]`, `[Rescale=scale]`; use the two 560-length vectors. Getting
  this wrong makes the CIF predictor emit ~4× too few tokens.
- **CIF/predictor runs on host** (it's a sequential integrate-and-fire loop);
  the encoder and decoder run in ggml.
- The decoder self-attention is **FSMN-only** (no QK attention); cross-attention
  attends to the encoder output. The decoder FFN has an internal LayerNorm and the
  second linear has no bias.
- WAV input assumes 16 kHz mono PCM16.

## Files
```
funasr-paraformer/         ggml runtime: WAV → token ids
export_paraformer_gguf.py  export encoder + predictor + decoder + CMVN to GGUF
detok.py                   token-id → text (tokens.json)
```

## Roadmap
- Built-in detok; timestamps (CIF peaks give alignment); arbitrary WAV / resampling;
  encoder/decoder quantization.
