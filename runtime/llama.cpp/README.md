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

The helper requires the Hugging Face CLI (`pip install -U huggingface_hub`). By
default it downloads one practical quantized variant plus FSMN-VAD, rather than
every GGUF in the repository:

```bash
./download-funasr-model.sh sensevoice          # q8 (default) + FSMN-VAD
./download-funasr-model.sh paraformer          # q8 (default) + FSMN-VAD
./download-funasr-model.sh nano                 # encoder-f16 + q8_0 (default) + FSMN-VAD
./download-funasr-model.sh fsmn-vad             # FSMN-VAD only
```

Use the optional third argument to choose another variant. SenseVoice and
Paraformer support `q8`, `f16`, `f32`, or `all`; Nano supports `q8_0`, `q4km`,
`q5km`, or `all`. The optional second argument is the output directory:

```bash
./download-funasr-model.sh sensevoice funasr-gguf f16
./download-funasr-model.sh nano funasr-gguf q4km
./download-funasr-model.sh paraformer funasr-gguf all  # explicitly download every GGUF
```

Pre-converted GGUF on Hugging Face: [SenseVoiceSmall-GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF) · [Paraformer-GGUF](https://huggingface.co/FunAudioLLM/Paraformer-GGUF) · [Fun-ASR-Nano-GGUF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF) · [fsmn-vad-GGUF](https://huggingface.co/FunAudioLLM/fsmn-vad-GGUF). Or convert yourself with `convert-funasr-to-gguf.py`.

## Build (standalone, CI-friendly)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release      # fetches pinned llama.cpp; static, self-contained
cmake --build build -j                          # -> build/bin/llama-funasr-* (all tools)
```

### Optional Windows CUDA backend for SenseVoiceSmall

The CPU release ZIPs are portable packages. Tagged releases also publish
`funasr-llamacpp-windows-x64-cuda.zip` for SenseVoiceSmall graph execution on
NVIDIA GPUs that match CUDA architecture 86. Download the CUDA ZIP from
[runtime-llamacpp-v0.1.9](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.1.9),
then select the backend at runtime:

```bash
# From the extracted windows-x64-cuda package:
./llama-funasr-sensevoice \
  -m sensevoice-small-q8.gguf --vad fsmn-vad.gguf -a sample.wav --backend cuda
```

Build from source to target other GPU architectures:

```bash
cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda -j --target llama-funasr-sensevoice
./build-cuda/bin/llama-funasr-sensevoice \
  -m sensevoice-small-f16.gguf -a sample.wav --backend cuda
```

Use the matching `CMAKE_CUDA_ARCHITECTURES` value for your GPU. RTX 50 /
Blackwell cards report compute capability 12.0 (`sm_120`), so the current
`windows-x64-cuda` prebuilt package for architecture 86 will not cover those
cards.

`--backend cpu` remains the default and is what the portable cross-platform
prebuilt binaries use. The CUDA package requires an NVIDIA driver compatible
with the CUDA Toolkit version configured by the release workflow. A binary built
without `-DGGML_CUDA=ON` exits with a clear message if `--backend cuda` is
requested.

### Optional Linux Vulkan backend for SenseVoiceSmall

Tagged releases also publish `funasr-llamacpp-linux-x64-vulkan.tar.gz` for
SenseVoiceSmall graph execution through ggml's Vulkan backend. This is useful on
Linux systems with AMD, Intel, NVIDIA, or integrated GPUs that expose a working
Vulkan driver/ICD. Download the `linux-x64-vulkan` asset, install your vendor GPU
driver, then select the backend at runtime:

```bash
# From the extracted linux-x64-vulkan package:
./llama-funasr-sensevoice \
  -m sensevoice-small-q8.gguf --vad fsmn-vad.gguf -a sample.wav --backend vulkan
```

Build from source when you need a local Vulkan SDK, distro-specific driver
stack, or to validate a device before release packaging:

```bash
sudo apt-get install libvulkan-dev glslc spirv-headers vulkan-tools
vulkaninfo --summary
cmake -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build-vulkan -j --target llama-funasr-sensevoice
./build-vulkan/bin/llama-funasr-sensevoice \
  -m sensevoice-small-f16.gguf -a sample.wav --backend vulkan
```

`--backend cpu` remains the default and is what the portable cross-platform
prebuilt binaries use. A binary built without `-DGGML_VULKAN=ON` exits with a
clear message if `--backend vulkan` is requested. Vulkan performance and device
availability depend on the installed GPU driver/ICD rather than on CUDA compute
capability.

### Optional Windows Vulkan backend for SenseVoiceSmall

Tagged releases publish `funasr-llamacpp-windows-x64-vulkan.zip` for the same
SenseVoiceSmall Vulkan graph execution on Windows. The prebuilt package does not
require the Vulkan SDK. It does require a current AMD, Intel, or NVIDIA graphics
driver that provides a working Vulkan loader and device:

```powershell
# From the extracted windows-x64-vulkan package:
vulkaninfo --summary  # Optional driver check when vulkaninfo is installed.
.\llama-funasr-sensevoice.exe `
  -m sensevoice-small-q8.gguf --vad fsmn-vad.gguf -a sample.wav --backend vulkan
```

If the command reports that no Vulkan device is available, update the vendor GPU
driver first. The package intentionally relies on the system `vulkan-1.dll`
installed by that driver instead of shipping an SDK copy.

To build on Windows, install the
[LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) with `glslc`,
open a Developer PowerShell where `VULKAN_SDK` is set, and install the
`SPIRV-Headers` CMake package expected by the pinned llama.cpp revision:

```powershell
glslc --version
git clone https://github.com/KhronosGroup/SPIRV-Headers.git
git -C SPIRV-Headers checkout 09913f088a1197aba4aefd300a876b2ebbaa3391
cmake -S SPIRV-Headers -B SPIRV-Headers-build `
  -DSPIRV_HEADERS_ENABLE_INSTALL=ON -DSPIRV_HEADERS_ENABLE_TESTS=OFF `
  -DCMAKE_INSTALL_PREFIX="$PWD/SPIRV-Headers-install"
cmake --install SPIRV-Headers-build --config Release
$env:CMAKE_PREFIX_PATH = "$PWD/SPIRV-Headers-install"

cmake -B build-vulkan -A x64 -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build-vulkan --config Release --target llama-funasr-sensevoice
.\build-vulkan\bin\Release\llama-funasr-sensevoice.exe `
  -m sensevoice-small-f16.gguf -a sample.wav --backend vulkan
```

`--backend cpu` remains the default. The Windows Vulkan package currently
accelerates SenseVoiceSmall only, matching the Linux Vulkan package.

## Build (shared)
```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cp -r /path/to/runtime/llama.cpp/funasr-common examples/   # shared audio loader (miniaudio); each example CMake adds ../funasr-common
cp -r /path/to/runtime/llama.cpp/<model>/<example-dir> examples/
echo 'add_subdirectory(<example-dir>)' >> examples/CMakeLists.txt
cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build -j --target <target>
```
The shared **FSMN-VAD** front end builds the same way (`funasr-vad/` + `funasr-common/`,
target `llama-funasr-vad`); export weights with `export_vad_gguf.py`. Pass
`--vad fsmn-vad.gguf` to any of the three tools for built-in long-audio segmentation.

## Lightweight HTTP server

The GGUF binaries are command-line tools first. For local apps that expect an
HTTP transcription endpoint, `server/funasr_gguf_server.py` wraps an existing
binary and exposes an OpenAI-compatible `POST /v1/audio/transcriptions` route.
It uses only the Python standard library and still runs inference in the C++
binary:

```bash
python server/funasr_gguf_server.py \
  --host 127.0.0.1 --port 8000 \
  --binary ./build/bin/llama-funasr-sensevoice \
  --model ./gguf/sensevoice-small-q8.gguf \
  --vad ./gguf/fsmn-vad.gguf
```

Then send audio with the same shape used by OpenAI-compatible clients:

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=funasr-gguf
```

Response:

```json
{"text": "transcribed text"}
```

CUDA- and Vulkan-enabled SenseVoice builds can be selected with `--backend cuda`
or `--backend vulkan`. Extra binary flags can be forwarded with repeated
`--extra-arg`, for example `--extra-arg --keep-tags`. This wrapper starts one subprocess per request,
so it is best for local tools, demos, and integration
tests. For sustained production traffic, use the Python `funasr-server`
OpenAI-compatible service or build a dedicated native server around the C++
runtime.

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
