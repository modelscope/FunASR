([简体中文](./README_zh.md)|English|[日本語](./README_ja.md)|[한국어](./README_ko.md))

<p align="center">
<a href="https://github.com/modelscope/FunASR"><img src="https://svg-banners.vercel.app/api?type=origin&text1=FunASR🤠&text2=💖%20A%20Fundamental%20End-to-End%20Speech%20Recognition%20Toolkit&width=800&height=210" alt="FunASR"></a>
</p>

<p align="center">
  <strong>Industrial speech recognition toolkit for offline, streaming, and edge deployment.</strong><br>
  <em>ASR · VAD · punctuation · speaker pipelines · emotion and audio-event models · OpenAI-compatible serving</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/funasr/"><img src="https://img.shields.io/pypi/v/funasr" alt="PyPI"></a>
  <a href="https://github.com/modelscope/FunASR"><img src="https://img.shields.io/github/stars/modelscope/FunASR?style=social" alt="Stars"></a>
  <a href="https://pypi.org/project/funasr/"><img src="https://img.shields.io/pypi/dm/funasr" alt="Downloads"></a>
  <a href="https://modelscope.github.io/FunASR/"><img src="https://img.shields.io/badge/docs-online-blue" alt="Docs"></a>
  <a href="https://mcptoplist.com/server/io.github.modelscope%2Ffunasr-mcp"><img src="https://mcptoplist.com/badge/io.github.modelscope%2Ffunasr-mcp.svg" alt="MCP Toplist"></a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/10479" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10479" alt="modelscope%2FFunASR | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> · <a href="./docs/model_selection.md">Model selection</a> · <a href="#model-zoo">Models</a> · <a href="./docs/deployment_matrix.md">Deployment matrix</a> · <a href="https://www.funasr.com/">Deployment hub</a> · <a href="https://www.funasr.com/en/docs/">Docs</a> · <a href="#benchmark">Benchmark</a> · <a href="./CONTRIBUTING.md">Contribute</a>
</p>

---

## Quick Start

### Native Transformers

For Fun-ASR-Nano transcription with the Hugging Face API, start with the [Transformers 5.17.0 CPU quickstart](./docs/transformers_native.md). No FunASR toolkit or remote Python code is needed.

[Space](https://huggingface.co/spaces/FunAudioLLM/Fun-ASR-Nano) · [Notebook](https://colab.research.google.com/github/QwenAudio/Fun-ASR/blob/main/examples/colab/fun_asr_nano_transformers.ipynb) · [Python / batch examples](https://github.com/QwenAudio/Fun-ASR/tree/main/examples/transformers)

### FunASR toolkit and pipelines

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modelscope/FunASR/blob/main/examples/colab/funasr_quickstart.ipynb)

No local setup? Open the [Colab quickstart](./examples/colab/) to transcribe a public sample or upload your own audio in a browser.

Found FunASR useful? [Star the project](https://github.com/modelscope/FunASR) so more builders can find it.

```bash
# CPU-only installs can use the default PyPI wheels.
pip install torch torchaudio
pip install funasr
```

For GPU quickstarts, install the PyTorch and torchaudio wheels that match your
NVIDIA driver from [pytorch.org](https://pytorch.org/get-started/locally/)
before installing FunASR. After installation, confirm the GPU is visible:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

Only use `device="cuda"` when this prints `True`; otherwise use `device="cpu"`
or reinstall PyTorch with the correct CUDA wheel.

**FunASR toolkit GPU example: Fun-ASR-Nano** (Chinese, English, Japanese, and Chinese dialect groups and regional accents; the separate native Transformers CPU path is linked above):

```python
from funasr import AutoModel

model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512", device="cuda")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")
print(result[0]["text"])
```

For the separate 31-language checkpoint, use
[Fun-ASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512).
Language coverage is checkpoint-specific, so Nano and MLT-Nano should be treated as distinct model choices.

For a CPU-first example with five-language ASR plus emotion and audio-event
tags, use **SenseVoiceSmall**. The pipeline below combines it with FSMN-VAD and
CAM++ for speaker-aware VAD segments; these are not native speaker outputs of
the SenseVoiceSmall checkpoint.
See the [SenseVoice paper](https://arxiv.org/abs/2407.04051),
[Hugging Face checkpoint](https://huggingface.co/FunAudioLLM/SenseVoiceSmall),
and [GGUF edge checkpoint](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF).

```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad", spk_model="cam++", device="cpu")
result = model.generate(
    input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
    batch_size_s=300,
)

# The AutoModel pipeline returns VAD segments with speaker ids and timestamps:
for seg in result[0]["sentence_info"]:
    print(f"[{seg['start']/1000:.1f}s] Speaker {seg['spk']}: {rich_transcription_postprocess(seg['sentence'])}")
```

This prints each returned segment's start time in seconds, anonymous speaker
index, and text with SenseVoice tags removed. Text and segment boundaries depend
on the audio and checkpoint; no fixed transcript is asserted here.

CAM++ extracts `spk_embedding` vectors. `AutoModel` clusters those embeddings
and assigns speaker indices to VAD segments. Indices are local to a recording,
not known-person identities. See the [SDK contract](./docs/python_api.md) for
the component and result boundaries. Change to `device="cuda"` only after
verifying a compatible GPU environment as described above.

### Scale & deploy the flagship

At scale, accelerate Fun-ASR-Nano with vLLM (batch processing):

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

model = AutoModelVLLM(model="FunAudioLLM/Fun-ASR-Nano-2512", tensor_parallel_size=1)
results = model.generate(["audio1.wav", "audio2.wav"], language="auto")
```

> **Deploy as API server:** [Local SenseVoice CPU recipe](#deploy) · [Nano GPU serving and pinned vLLM setup](./docs/vllm_guide.md)
>
> **Use with AI agents:** [MCP Server](examples/mcp_server/) for Claude/Cursor · [OpenAI API](examples/openai_api/) for LangChain/Dify/AutoGen
>
> **Use with voice agents:** [OpenClaw realtime plugin](integrations/openclaw/) for self-hosted Talk and Voice Call transcription

### Why FunASR?

FunASR is a toolkit: choose the task, checkpoint, and runtime separately.
Support in one model or adapter does not imply support in every serving backend.

| Task | Checkpoint or pipeline | Runtime entrypoint | Important limitation |
|---|---|---|---|
| File transcription with emotion/event tags | SenseVoiceSmall | Python `AutoModel`, CPU or GPU | Five-language checkpoint; tags do not identify speakers. |
| LLM-based file transcription | Fun-ASR-Nano | `AutoModel`; split-engine `AutoModelVLLM` for the documented GPU path | Base Nano covers zh/en/ja and Chinese dialects/accents; timestamp support depends on checkpoint and path. |
| Broader multilingual transcription | Fun-ASR-MLT-Nano | Python `AutoModel` | Separate 31-language checkpoint; do not transfer its coverage to base Nano. |
| Chunked live transcription | Paraformer-zh-streaming | Streaming SDK or runtime WebSocket service | Use the streaming checkpoint and per-session cache, not an offline checkpoint. |
| Speaker-aware file transcription | SenseVoiceSmall + FSMN-VAD + CAM++ | `AutoModel` with VAD and embedding clustering | Anonymous indices within a recording, not enrolled-speaker identification. |
| Joint text, timestamps, and speakers | MOSS-Transcribe-Diarize, third-party OpenMOSS | FunASR adapter or upstream backend in the MOSS guide | Offline, recording-local anonymous labels; no external VAD/speaker pipeline for its unified path. |
| Native CPU/edge transcription | Fun-ASR-Nano or SenseVoiceSmall GGUF | llama.cpp runtime | Requires matching converted weights; GGUF is not a Python `AutoModel` checkpoint. |

See the [Model Zoo](./model_zoo/readme.md) and [deployment matrix](./docs/deployment_matrix.md)
for checkpoint, interface, and licensing boundaries. Benchmark on your own audio
and hardware before choosing a runtime.

Trying FunASR for the first time? Use the [Colab quickstart](./examples/colab/) before setting up a local environment. Choosing a first model? Start with the [model selection guide](./docs/model_selection.md). Planning a switch from Whisper or a cloud ASR provider? Use the [migration guide](./docs/migration_from_whisper.md) and [benchmark example](./examples/migration/) to test representative audio, map features, and roll out safely.

---

## Installation

```bash
pip install funasr
```

<details><summary>From source / Requirements</summary>

```bash
git clone https://github.com/modelscope/FunASR.git && cd FunASR
pip install -e ./
```
Requirements: Python ≥ 3.8. Install PyTorch + torchaudio first ([pytorch.org](https://pytorch.org/get-started/locally/)), then `pip install funasr`.

</details>

---

## Model Zoo

This list includes third-party models. **OpenMOSS** publishes MOSS-Transcribe-Diarize;
FunASR provides an adapter, not ownership of its weights. Its unified path is
offline, with anonymous labels scoped to each recording, not realtime or
known-person identification. Model licenses are separate from the toolkit's MIT license.

| Model | Task | Languages | Params | Links |
|-------|------|-----------|--------|-------|
| **Fun-ASR-Nano** | ASR | zh/en/ja + Chinese dialects and accents | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [HF / Transformers](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf) · [HF / FunASR](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) [GGUF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF) |
| **Fun-ASR-MLT-Nano** | ASR | 31 languages | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) |
| **SenseVoiceSmall** | ASR + emotion + events | zh/en/ja/ko/yue | 234M | [⭐](https://www.modelscope.cn/models/iic/SenseVoiceSmall) [🤗](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) [GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF) [paper](https://arxiv.org/abs/2407.04051) |
| **MOSS-Transcribe-Diarize** | Third-party OpenMOSS: offline ASR + timestamps + anonymous speakers | See official card | See official card | [🤗](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) [guide](./docs/moss_transcribe_diarize.md) |
| **Paraformer-zh** | ASR + timestamps | zh/en | 220M | [⭐](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) [🤗](https://huggingface.co/funasr/paraformer-zh) |
| Paraformer-zh-streaming | Streaming ASR | zh/en | 220M | [⭐](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary) [🤗](https://huggingface.co/funasr/paraformer-zh-streaming) |
| Qwen3-ASR | ASR, 52 languages | multilingual | 1.7B | [usage](examples/industrial_data_pretraining/qwen3_asr) |
| GLM-ASR-Nano | ASR, 17 languages | multilingual | 1.5B | [usage](examples/industrial_data_pretraining/glm_asr) |
| Whisper-large-v3 | ASR + translation | multilingual | 1550M | [usage](examples/industrial_data_pretraining/whisper) |
| Whisper-large-v3-turbo | ASR + translation | multilingual | 809M | [usage](examples/industrial_data_pretraining/whisper) |
| ct-punc | Punctuation | zh/en | 290M | [⭐](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/summary) [🤗](https://huggingface.co/funasr/ct-punc) |
| fsmn-vad | VAD | zh/en | 0.4M | [⭐](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) [🤗](https://huggingface.co/funasr/fsmn-vad) |
| cam++ | Speaker embeddings (pipeline component) | — | 7.2M | [⭐](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/summary) [🤗](https://huggingface.co/funasr/campplus) |
| emotion2vec+large | Emotion recognition | — | 300M | [⭐](https://modelscope.cn/models/iic/emotion2vec_plus_large/summary) [🤗](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |

---

## Usage

> [Python tutorial](./docs/tutorial/README.md) · [SDK parameters and outputs](./docs/python_api.md) · [Training](./docs/training.md) · [Model registration](./docs/model_registration.md)

```python
from funasr import AutoModel

# Chinese production (VAD + ASR + punctuation + speaker)
model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", spk_model="cam++", device="cuda")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav", hotword="关键词 20")

# Optional Silero VAD (install first: python -m pip install "funasr[silero]")
model = AutoModel(
    model="paraformer-zh", vad_model="silero-vad", device="cuda",
    vad_kwargs={"silero_threshold": 0.5, "silero_min_silence_duration_ms": 100},
)
result = model.generate(input="audio.wav")

# Streaming real-time (feed audio chunk by chunk)
import soundfile as sf
model = AutoModel(model="paraformer-zh-streaming", device="cuda")
audio, sr = sf.read("speech.wav", dtype="float32")   # 16 kHz mono
chunk_size = [0, 10, 5]                               # 600 ms chunks
chunk_stride = chunk_size[1] * 960
cache = {}
n_chunks = (len(audio) - 1) // chunk_stride + 1
for i in range(n_chunks):
    chunk = audio[i * chunk_stride : (i + 1) * chunk_stride]
    res = model.generate(input=chunk, cache=cache, is_final=(i == n_chunks - 1),
                         chunk_size=chunk_size, encoder_chunk_look_back=4, decoder_chunk_look_back=1)
    if res[0]["text"]:
        print(res[0]["text"], end="", flush=True)

# Emotion recognition
model = AutoModel(model="emotion2vec_plus_large", device="cuda")
result = model.generate(input="audio.wav", granularity="utterance")
```


### CLI (Agent-Friendly)

```bash
# Transcribe audio (simplest)
funasr audio.wav

# JSON output (for AI agents)
funasr audio.wav --output-format json

# SRT subtitles
funasr audio.wav --output-format srt --output-dir ./subs

# Speaker diarization + timestamps
funasr audio.wav --spk --timestamps -f json

# Choose model and language
funasr audio.wav --model paraformer --language zh

# Batch transcribe
funasr *.wav --output-format srt --output-dir ./output
```

Available models: `sensevoice` (default), `paraformer`, `paraformer-en`, `fun-asr-nano`

---

## Deploy

Start a local SenseVoice CPU service from a fresh directory in a POSIX shell with
Python 3.11. This installs the PyPI release into a separate environment, not this
source checkout. Keep the unauthenticated service on loopback; use the
[security guide](./examples/openai_api/SECURITY.md) before exposing it to other clients.

```bash
python3.11 -m venv .venv-funasr-http
. .venv-funasr-http/bin/activate
python -m pip install torch torchaudio
python -m pip install funasr fastapi uvicorn python-multipart
python -m pip check
funasr-server --host 127.0.0.1 --port 8000 --model sensevoice --device cpu
```

Wait for model download and server startup. In a second terminal, use the same
directory and curl 7.76+ to download a public Chinese audio sample and transcribe it.
The request uses the preloaded model; no fixed transcript or speaker labels are promised.

```bash
curl --fail --location https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav && \
curl --fail-with-body http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

For offline joint ASR and anonymous speaker labels (`moss-transcribe-diarize`), prepare the separate environment
in the [MOSS service, Docker, Kubernetes, vLLM, SGLang, LocalAI, and FunClip guide →](./docs/moss_transcribe_diarize.md).
It is an alternative service, not another command in the CPU environment. Stop the
CPU service before reusing port 8000. For Nano GPU serving, follow the
[pinned split-engine guide](./docs/vllm_guide.md) and inspect the actual backend logs;
selecting a model does not by itself prove that vLLM was loaded.

```bash
# Docker streaming service
docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.12
```

### CPU / Edge — llama.cpp / GGUF (no GPU, no Python)

Run **SenseVoice / Paraformer / Fun-ASR-Nano** as a **single self-contained binary** on CPU and edge devices — this is to FunASR what [whisper.cpp](https://github.com/ggml-org/whisper.cpp) is to Whisper, but with **~3× lower CER than whisper.cpp on Chinese**. Built-in FSMN-VAD, no Python at runtime.

```bash
# Linux / macOS: run from the extracted release directory
bash download-funasr-model.sh sensevoice ./gguf        # or: paraformer | nano
./llama-funasr-sensevoice -m ./gguf/sensevoice-small-q8.gguf --vad ./gguf/fsmn-vad.gguf -a audio.wav
# → 欢迎大家来体验达摩院推出的语音识别模型
```

```powershell
# Windows PowerShell: run from the extracted archive root (with the `hf` CLI installed)
hf download FunAudioLLM/SenseVoiceSmall-GGUF sensevoice-small-q8.gguf --local-dir .\gguf
hf download FunAudioLLM/fsmn-vad-GGUF fsmn-vad.gguf --local-dir .\gguf
.\llama-funasr-sensevoice.exe -m .\gguf\sensevoice-small-q8.gguf --vad .\gguf\fsmn-vad.gguf -a audio.wav
# Use the windows-x64-vulkan package with a current AMD, Intel, or NVIDIA Vulkan driver:
.\llama-funasr-sensevoice.exe -m .\gguf\sensevoice-small-q8.gguf --vad .\gguf\fsmn-vad.gguf -a audio.wav --backend vulkan
# Use the windows-x64-cuda package on RTX 30-class GPUs:
.\llama-funasr-sensevoice.exe -m .\gguf\sensevoice-small-q8.gguf --vad .\gguf\fsmn-vad.gguf -a audio.wav --backend cuda
```

Use `funasr-llamacpp-linux-x64-vulkan.tar.gz` on Linux GPU systems with a
working Vulkan driver/ICD:

```bash
./llama-funasr-sensevoice -m ./gguf/sensevoice-small-q8.gguf --vad ./gguf/fsmn-vad.gguf -a audio.wav --backend vulkan
```

The Windows Vulkan ZIP uses the system Vulkan loader supplied by the GPU driver;
installing the Vulkan SDK is only necessary when building from source. Both
Vulkan packages currently accelerate SenseVoiceSmall.

Tagged releases provide two Windows CUDA packages. The standard
`windows-x64-cuda` ZIP targets CUDA architecture 86, while
`windows-x64-cuda-blackwell` targets architecture 120 (`sm_120`) for RTX 50 /
Blackwell GPUs. Both ZIPs bundle the required cuBLAS DLLs and use the static MSVC
runtime, so users need a compatible NVIDIA driver but not a separate CUDA Toolkit
installation. CI verifies the architecture and package boundary; it does not prove
inference on physical Blackwell hardware.

**Prebuilt binaries:** [Releases](https://github.com/modelscope/FunASR/releases) · [v0.2.6](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.6) · [Linux Vulkan tarball](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-linux-x64-vulkan.tar.gz) · [Windows Vulkan zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-vulkan.zip) · [Windows CUDA zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-cuda.zip) · [Windows Blackwell CUDA zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-cuda-blackwell.zip) · **Download & quickstart:** [funasr.com/deploy/llama-cpp](https://www.funasr.com/en/deploy/llama-cpp.html) · **GGUF models:** [Hugging Face](https://huggingface.co/FunAudioLLM) · **Docs & benchmarks:** [runtime/llama.cpp/](./runtime/llama.cpp/)

[OpenAI API example →](./examples/openai_api/) · [Gradio demo →](./examples/openai_api/GRADIO.md) · [Client recipes →](./examples/openai_api/CLIENTS.md) · [JavaScript/TypeScript recipes →](./examples/openai_api/JAVASCRIPT.md) · [Kubernetes template →](./examples/openai_api/kubernetes/) · [Workflow recipes →](./examples/openai_api/WORKFLOWS.md) · [Postman collection →](./examples/openai_api/POSTMAN.md) · [OpenAPI spec →](./examples/openai_api/OPENAPI.md) · [Security guide →](./examples/openai_api/SECURITY.md) · [Deployment matrix →](./docs/deployment_matrix.md) · [Deployment docs →](./runtime/readme.md) · [Agent integration →](https://modelscope.github.io/FunASR/agent.html)

---

## Benchmark

The [historical benchmark report](https://modelscope.github.io/FunASR/benchmark.html)
and [split-engine measurements](./docs/vllm_guide.md#benchmark) retain their
original results. They are separate records, not universal speed rankings or
production capacity guarantees.

Use the [RTFx and reproducibility notes](./docs/benchmark/rtf_reproducibility.md)
to compare checkpoint/revision, audio set, hardware, batching, warmup, timing
scope, and CER/WER. Offline throughput is not streaming latency. The
[migration benchmark example](./examples/migration/) helps measure your own
recordings with the same evaluation scope.

---

## What's new

- **MOSS-Transcribe-Diarize** brings long-form ASR, timestamps, and anonymous speaker labels to FunASR services, Docker, Kubernetes, vLLM/SGLang workflows, and FunClip. [Deploy MOSS ->](./docs/moss_transcribe_diarize.md)
- **FunASR 1.4.15** adds tested NumPy 2 compatibility and fixes streaming KWS/VAD boundaries and checkpoint ranking. Install with `python -m pip install -U "funasr==1.4.15"`. [Release and verification scope ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.15)
- **Native Transformers:** Released **5.17.0** supports Fun-ASR-Nano with the official `-hf` checkpoint, CPU examples and a notebook. [Get started ->](./docs/transformers_native.md)

> See [GitHub Releases](https://github.com/modelscope/FunASR/releases) for the complete changelog and downloadable assets.

---

## Community

Start with [troubleshooting](./docs/troubleshooting.md) before reporting a problem.
Include your exact model, runtime, environment and a minimal reproduction.

|  |  |
|---|---|
| 📖 [Documentation](https://modelscope.github.io/FunASR/) | 🐛 [Issues](https://github.com/modelscope/FunASR/issues) |
| 💬 [Discussions](https://github.com/modelscope/FunASR/discussions) | 🤗 [HuggingFace](https://huggingface.co/funasr) |
| 🤝 [Contributing](./CONTRIBUTING.md) | 🌐 [funasr.com](https://www.funasr.com) |
| 🗺️ [Repository roles & roadmap](./docs/repository_roles.md) | 📈 [Growth plan](./docs/community_growth_20k.md) |
| 🧩 [Community projects](./docs/community_projects.md) | 💡 [Use-case showcase](./docs/use_case_showcase.md) |

## Star History

<a href="https://star-history.com/#modelscope/FunASR&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date" width="600" />
 </picture>
</a>

## License

- FunASR toolkit source code in this repository: [MIT License](./LICENSE).
- Pretrained model weights are licensed separately. Check the license shown on each model card; when a model card links to the [FunASR Model Open Source License Agreement](./MODEL_LICENSE), those terms apply.

## Citations

```bibtex
@inproceedings{gao2023funasr,
  author={Zhifu Gao and others},
  title={FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
  booktitle={INTERSPEECH},
  year={2023}
}
```
