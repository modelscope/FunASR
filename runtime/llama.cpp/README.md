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

## SRT subtitle output

Fun-ASR-Nano, SenseVoiceSmall, and Paraformer accept `--srt` and write standard
SRT entries to stdout. Use FSMN-VAD segmentation for speech-aligned timestamps;
Fun-ASR-Nano can also timestamp fixed windows selected with `--chunk`. Without
segmentation, SenseVoiceSmall and Paraformer emit one entry spanning the input.

```bash
./build/bin/llama-funasr-cli --enc encoder.gguf -m llm.gguf \
  --vad fsmn-vad.gguf -a audio.wav --srt > audio.srt
./build/bin/llama-funasr-sensevoice -m sensevoice-small.gguf \
  --vad fsmn-vad.gguf -a audio.wav --srt > audio.srt
./build/bin/llama-funasr-paraformer -m paraformer.gguf \
  --vad fsmn-vad.gguf -a audio.wav --srt > audio.srt
```

Progress and timing diagnostics remain on stderr, so redirecting stdout produces
a clean subtitle file. Normal text output is unchanged when `--srt` is omitted.

## Speaker diarization

The standalone llama.cpp / GGUF binaries do **not** currently implement CAM++
speaker embeddings or speaker clustering. `--vad` segments speech for long-audio
transcription, but it does not assign speaker labels.

For speaker-aware transcripts, use the Python `AutoModel` pipeline with
`spk_model="cam++"` (see the [FunASR quick start](../../README.md#quick-start)) or
the [Fun-ASR-Nano vLLM service](../../examples/industrial_data_pretraining/fun_asr_nano/serve_vllm.py),
which accepts `spk=true` and runs a separate speaker model. Keep using the
llama.cpp binaries when the requirement is self-contained ASR plus optional
FSMN-VAD without Python.

### Optional Windows CUDA backend for SenseVoiceSmall

The CPU release ZIPs are portable packages. Tagged releases publish separate
SenseVoiceSmall CUDA packages: `funasr-llamacpp-windows-x64-cuda.zip` targets
CUDA architecture 86, while `funasr-llamacpp-windows-x64-cuda-blackwell.zip`
targets CUDA architecture 120 (`sm_120`) for RTX 50 / Blackwell GPUs. Select the
matching archive, then enable the backend at runtime:

```bash
# From the extracted windows-x64-cuda package:
./llama-funasr-sensevoice \
  -m sensevoice-small-q8.gguf --vad fsmn-vad.gguf -a sample.wav --backend cuda
```

The Blackwell package uses the same command from its extracted directory. Build
from source to target other GPU architectures or to reproduce the architecture
120 build locally:

```bash
cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda -j --target llama-funasr-sensevoice
./build-cuda/bin/llama-funasr-sensevoice \
  -m sensevoice-small-f16.gguf -a sample.wav --backend cuda
```

Use the matching `CMAKE_CUDA_ARCHITECTURES` value for your GPU. A successful
release workflow build verifies architecture 120 code generation and ZIP
integrity; it does not prove execution on physical Blackwell hardware. Keep
hardware-specific reports open until the matching archive is retested on the
reported GPU.

`--backend cpu` remains the default and is what the portable cross-platform
prebuilt binaries use. The CUDA ZIPs bundle `cublas64_13.dll` and
`cublasLt64_13.dll`, include the NVIDIA license, and link the MSVC runtime
statically, so running the package does not require a separate CUDA Toolkit or
Visual C++ redistributable installation. It still requires an NVIDIA driver
compatible with the CUDA Toolkit version configured by the release workflow. A
binary built without `-DGGML_CUDA=ON` exits with a clear message if
`--backend cuda` is requested.

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

For access violations such as Windows exit code `-1073741819` (`0xC0000005`),
capture stderr and report the last completed boundary. Runtime v0.2.3 flushes
each boundary immediately, so the next missing line identifies the failing stage:

| Last completed boundary | Next stage to investigate |
| --- | --- |
| no `initializing vulkan backend ...` | device enumeration or selection |
| `initializing ...` | `ggml_backend_dev_init()` or the driver below it |
| `initialized ...; resolving buffer type` | default buffer-type resolution |
| `vulkan backend ready ...` | model metadata loading |
| `[sensevoice] model ready ...` | audio loading or feature extraction |
| `[sensevoice] audio ready ...` | VAD, when `--vad` is enabled |
| `[sensevoice] VAD ready ...` | graph construction |
| `[sensevoice] graph built` | graph allocation |
| `[sensevoice] graph allocated` | backend compute submission |
| `[sensevoice] compute starting` | Vulkan compute or the GPU driver |
| `[sensevoice] compute complete: status=0` | output transfer or CTC decoding |

Include the GPU model, driver version, complete command, and all stderr lines in
the issue. These boundaries diagnose the failure location; they do not by
themselves claim that a driver- or hardware-specific access violation is fixed.

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

#### AMD Windows troubleshooting

Recent AMD drivers can report `VK_ERROR_DEVICE_LOST` or terminate the process
when a graph submission exceeds the driver's timeout. The pinned ggml revision
reduces submission sizes on smaller AMD GPUs and fixes the batching threshold.
To force smaller submissions or collect the last submitted tensors for a bug
report, run from PowerShell with:

```powershell
$env:GGML_VK_MAX_NODES_PER_SUBMIT = "16"
$env:GGML_VK_SERIALIZE_SUBMISSIONS = "1"
.\llama-funasr-sensevoice.exe `
  -m sensevoice-small-f16.gguf --vad fsmn-vad.gguf -a sample.wav --backend vulkan
```

Try `8` or `1` if `16` still triggers a driver timeout. Include the complete
stderr output, GPU model, and driver version when reporting a failure. Remove
the variables after diagnosis because serial submissions can reduce throughput.
Use `--backend cpu` as the reliable fallback on affected driver/device pairs.
The current SenseVoiceSmall graph does not create a flash-attention operation,
so a `--no-flash-attn` switch would not change this execution path.

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
