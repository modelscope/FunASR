# SenseVoiceSmall on llama.cpp / GGUF

Run **SenseVoiceSmall** on the [llama.cpp](https://github.com/ggml-org/llama.cpp)
/ ggml stack — **CPU, edge, a single binary, no Python at runtime**. Like
[whisper.cpp](https://github.com/ggml-org/whisper.cpp), but for SenseVoice.

## Why this exists

SenseVoiceSmall normally runs on PyTorch / ONNX / libtorch. This runtime ports it
to **ggml + GGUF** so it can run CPU-only, offline, embedded in a C/C++ app, with
quantized weights. Use it on laptops / phones / edge boxes where there is no GPU
and no Python. (For high-QPS GPU serving, the PyTorch/vLLM path is still the way.)

## Architecture

SenseVoiceSmall = **SAN-M encoder (70 layers) + CTC head** — no LLM, no autoregression.
The whole pipeline runs in C++:

```
 audio.wav (16k mono)
      │  kaldi 80-mel fbank + LFR                         (C++)
      ▼
   features [T, 560]
      │  prepend 4 query tokens [lang, event, emotion, itn]
      ▼
   [4 + T, 560]
      │  SAN-M encoder                                    (ggml)  ── sensevoice-small.gguf
      ▼
   encoder out [4+T, 512]
      │  CTC head (Linear 512→25055) → greedy CTC (argmax, dedup, drop blank)
      ▼
   token ids
      │  SentencePiece detok                             (detok.py)
      ▼
   <|zh|><|NEUTRAL|><|Speech|><|woitn|> transcription...
```

The SAN-M encoder is the same architecture as Fun-ASR-Nano's, so the ggml forward
is shared between the two runtimes.

## Quickstart

**1. Build:**
```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cp -r /path/to/runtime/llama.cpp/funasr-sensevoice examples/
echo 'add_subdirectory(funasr-sensevoice)' >> examples/CMakeLists.txt
cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build -j --target llama-funasr-sensevoice
```

**2. Convert weights** (needs the checkpoint, e.g. `FunAudioLLM/SenseVoiceSmall`):
```bash
python runtime/llama.cpp/export_sensevoice_gguf.py \
    --model_pt <model>/model.pt --mvn <model>/am.mvn \
    --out sensevoice-small.gguf                          # f32, ~936 MB
python runtime/llama.cpp/export_sensevoice_gguf.py --wtype f16 \
    --model_pt <model>/model.pt --mvn <model>/am.mvn \
    --out sensevoice-small-f16.gguf                       # half size
```

**3. Transcribe:**
```bash
build/bin/llama-funasr-sensevoice -m sensevoice-small.gguf -a audio.wav > ids.txt
python runtime/llama.cpp/detok.py <model>/chn_jpn_yue_eng_ko_spectok.bpe.model ids.txt
```
Expected output:
```
<|zh|><|NEUTRAL|><|Speech|><|woitn|>我想问我在滨海新区有房我一直没有照顾孩子...你觉得这是正常的想法吗
[sensevoice] N=746 (q4+T742) encode 1.32s
```
The leading `<|...|>` tags are the predicted language / emotion / event / ITN.

## Accuracy & validation

- **CTC token ids (C++) vs PyTorch:** **identical** (108/108 on a benchmark clip).
- **Detokenized text:** matches the FunASR `AutoModel` output **exactly**.
- Encoder validated against PyTorch (shared with Fun-ASR-Nano runtime): cosine 1.0.
- Encode time ≈ **1.3 s** on CPU for a 44 s clip.

## Tips & gotchas

- **No CMVN at inference.** SenseVoice `inference()` feeds the **raw** log-mel fbank
  to the encoder; it does **not** apply `am.mvn`. Applying CMVN makes the model
  predict `<|nospeech|>`. (The export script reads `am.mvn` for completeness but the
  runtime does not use it.)
- **Query tokens (4)** are prepended from `embed.weight`, default indices
  `[language=auto(0), event=1, emotion=2, textnorm=woitn(15)]`. Change them for a
  fixed language or to enable ITN (`withitn=14`).
- **WAV input** assumes 16 kHz mono PCM16.
- LayerNorm eps = 1e-5; FSMN = exact f32 shift-accumulate; fbank matches torchaudio.

## Files
```
funasr-sensevoice/        ggml runtime: WAV → CTC token ids
export_sensevoice_gguf.py export encoder + CTC head + query embeddings to GGUF
detok.py                  SentencePiece id → text (bpe model ships with the checkpoint)
```

## Roadmap
- Built-in SentencePiece detok (drop the Python step); arbitrary WAV formats;
  encoder Q8 quantization; timestamps.
