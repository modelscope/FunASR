# FunASR on llama.cpp / GGUF

Run FunASR models on the [llama.cpp](https://github.com/ggml-org/llama.cpp) / ggml
stack — **CPU, edge, a single binary, no Python at runtime, quantized weights**.
This is to FunASR what [whisper.cpp](https://github.com/ggml-org/whisper.cpp) is to
Whisper: it lets the models run where there is no GPU and no Python (laptops,
phones, edge boxes, embedded C/C++ apps), complementing the PyTorch / ONNX / vLLM
paths used for GPU serving.

## Models

| model | architecture | runtime | status |
|---|---|---|---|
| [Fun-ASR-Nano](fun-asr-nano/) | SenseVoice SAN-M encoder + adaptor + Qwen3-0.6B LLM | `llama-funasr-cli` | validated vs PyTorch |
| [SenseVoiceSmall](sensevoice/)  | SAN-M encoder + CTC | `llama-funasr-sensevoice` | CTC ids identical to PyTorch |
| [Paraformer](paraformer/)       | SAN-M encoder + CIF predictor + SAN-M decoder (non-autoregressive) | `llama-funasr-paraformer` | text identical to PyTorch |

All three share the same ggml SAN-M encoder / FSMN / attention primitives and the
same kaldi-compatible fbank front end (80-mel, LFR 7/6), so the C++ is consistent
across models.

## How it works

Each model's neural path is implemented as a ggml graph; the audio front end (kaldi
fbank) is plain C++. Weights are converted to GGUF (f32 or f16) with the per-model
`export_*_gguf.py` script. For Fun-ASR-Nano the LLM half is a standard Qwen3 GGUF
and the audio embeddings are injected into it via `llama_decode`'s embedding input
(the llava/mtmd mechanism). See each model's README for the architecture diagram,
build/convert/run quickstart, validation numbers, and gotchas.

## Download pre-built GGUF (fastest — no Python ML env)
```bash
./download-funasr-model.sh sensevoice          # or: paraformer | nano | fsmn-vad
```
Pre-converted GGUF on Hugging Face: [SenseVoiceSmall-GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF) · [Paraformer-GGUF](https://huggingface.co/FunAudioLLM/Paraformer-GGUF) · [Fun-ASR-Nano-GGUF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF) · [fsmn-vad-GGUF](https://huggingface.co/FunAudioLLM/fsmn-vad-GGUF). Or convert yourself with `convert-funasr-to-gguf.py`.

## Build (standalone, CI-friendly)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release      # fetches pinned llama.cpp; static, self-contained
cmake --build build -j                          # -> build/bin/llama-funasr-* (all tools)
```
## Build (shared)
```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cp -r /path/to/runtime/llama.cpp/<model>/<example-dir> examples/
echo 'add_subdirectory(<example-dir>)' >> examples/CMakeLists.txt
cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build -j --target <target>
```
The shared **FSMN-VAD** front end builds the same way (`funasr-vad/` + `funasr-common/`,
target `llama-funasr-vad`); export weights with `export_vad_gguf.py`. Pass
`--vad fsmn-vad.gguf` to any of the three tools for built-in long-audio segmentation.

## Validation

Each model was validated against the FunASR PyTorch reference (encoder cosine ≈ 1.0;
SenseVoice CTC token ids identical; Paraformer text identical; Fun-ASR-Nano aggregate
CER matches PyTorch within 0.02% under identical conditions). See per-model READMEs.

## Status / notes
- Any audio in (wav/mp3/flac, any rate/channels) via the bundled miniaudio loader.
- **Built-in FSMN-VAD (`--vad fsmn-vad.gguf`)** segments long audio inside the binary
  (native ggml, no Python front end); all three tools support it. Bare-binary full-184
  micro-CER: SenseVoice **8.01** / Paraformer **9.85** / Fun-ASR-Nano **8.30** (see
  [BENCHMARKS.md](BENCHMARKS.md)). `--chunk` fixed-window remains a simpler fallback.
- This adds a new `runtime/llama.cpp/` directory only; no existing code is modified.

## Further reading

See [DESIGN.md](DESIGN.md) for the full system design — architecture, the shared SAN-M encoder, GGUF weight format, numerical-fidelity and validation methodology, design trade-offs, and gotchas.
