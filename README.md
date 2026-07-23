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
</p>

<p align="center">
<a href="https://trendshift.io/repositories/10479" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10479" alt="modelscope%2FFunASR | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> · <a href="./examples/colab/">Colab</a> · <a href="#benchmark">Benchmark</a> · <a href="./docs/model_selection.md">Model selection</a> · <a href="./docs/migration_from_whisper.md">Migration guide</a> · <a href="./docs/use_case_showcase.md">Use cases</a> · <a href="./docs/community_projects.md">Community integrations</a> · <a href="./docs/deployment_matrix.md">Deployment matrix</a> · <a href="./docs/troubleshooting.md">Troubleshooting</a> · <a href="#model-zoo">Models</a> · <a href="https://modelscope.github.io/FunASR/agent.html">Agent Integration</a> · <a href="https://modelscope.github.io/FunASR/">Docs</a> · <a href="./CONTRIBUTING.md">Contribute</a>
</p>

---

## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modelscope/FunASR/blob/main/examples/colab/funasr_quickstart.ipynb)

No local setup? Open the [Colab quickstart](./examples/colab/) to transcribe a public sample or upload your own audio in a browser.

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

**Flagship model — Fun-ASR-Nano** (LLM-ASR for Chinese, English, and Japanese, plus Chinese dialect groups and regional accents; needs a GPU):

```python
from funasr import AutoModel

model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512", device="cuda")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")
print(result[0]["text"])
# 欢迎大家来体验达摩院推出的语音识别模型。
```

For the separate 31-language checkpoint, use
[Fun-ASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512).
Language coverage is checkpoint-specific, so Nano and MLT-Nano should be treated as distinct model choices.

On CPU (or for five-language ASR plus emotion and audio-event tags), use
**SenseVoiceSmall**. The pipeline below composes SenseVoiceSmall with FSMN-VAD
and CAM++; diarization is provided by the separate CAM++ model, not by the
SenseVoiceSmall checkpoint:
See the [SenseVoice paper](https://arxiv.org/abs/2407.04051),
[Hugging Face checkpoint](https://huggingface.co/FunAudioLLM/SenseVoiceSmall),
and [GGUF edge checkpoint](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF).

```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad", spk_model="cam++", device="cuda")  # use device="cpu" if you don't have a GPU
result = model.generate(
    input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
    batch_size_s=300,
)

# The AutoModel pipeline returns VAD segments with speaker ids and timestamps:
for seg in result[0]["sentence_info"]:
    print(f"[{seg['start']/1000:.1f}s] Speaker {seg['spk']}: {rich_transcription_postprocess(seg['sentence'])}")
```

**Output** — structured text with speaker labels, timestamps, and punctuation:
```
[0.6s] Speaker 0: 欢迎大家来体验达摩院推出的语音识别模型
```

One `AutoModel` pipeline call coordinates the configured ASR, VAD, and speaker
models and returns the combined result.

### Scale & deploy the flagship

At scale, accelerate Fun-ASR-Nano with vLLM (batch processing):

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

model = AutoModelVLLM(model="FunAudioLLM/Fun-ASR-Nano-2512", tensor_parallel_size=1)
results = model.generate(["audio1.wav", "audio2.wav"], language="auto")
```

> **Deploy as API server:** `funasr-server --device cuda` → OpenAI-compatible endpoint at localhost:8000
>
> **Use with AI agents:** [MCP Server](examples/mcp_server/) for Claude/Cursor · [OpenAI API](examples/openai_api/) for LangChain/Dify/AutoGen

### Why FunASR?

Whisper is a single model; **FunASR is a toolkit** — you pick the right model
per job: **Fun-ASR-Nano** (Chinese, English, Japanese, and Chinese dialects;
GPU), **Fun-ASR-MLT-Nano** (31 languages), **SenseVoiceSmall** (five-language
ASR plus emotion and audio events), and **Paraformer** (low-latency streaming).
The table shows toolkit-level capabilities and names the model or pipeline that
provides each one:

| | FunASR (toolkit) | Whisper | Cloud APIs |
|---|---|---|---|
| Top speed | **340x realtime** (Fun-ASR-Nano + vLLM) | 13x realtime | ~1x realtime |
| Speaker ID | ✅ via VAD + CAM++ pipeline | ❌ Needs pyannote | ✅ Extra cost |
| Emotion | ✅ via SenseVoice | ❌ | ❌ |
| Languages | Checkpoint-specific (for example Qwen3-ASR 52, MLT-Nano 31, Nano zh/en/ja) | 57 | Varies |
| Streaming | ✅ WebSocket (Paraformer) | ❌ | ✅ |
| CPU viable | ✅ 17x realtime (SenseVoice) | ❌ Too slow | N/A |
| Self-hosted | ✅ Yes (toolkit: MIT; model licenses vary) | ✅ MIT license | ❌ Cloud only |
| Cost | Free | Free | $0.006/min+ |

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

| Model | Task | Languages | Params | Links |
|-------|------|-----------|--------|-------|
| **Fun-ASR-Nano** | ASR | zh/en/ja + Chinese dialects and accents | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) [GGUF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF) |
| **Fun-ASR-MLT-Nano** | ASR | 31 languages | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) |
| **SenseVoiceSmall** | ASR + emotion + events | zh/en/ja/ko/yue | 234M | [⭐](https://www.modelscope.cn/models/iic/SenseVoiceSmall) [🤗](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) [GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF) [paper](https://arxiv.org/abs/2407.04051) |
| **Paraformer-zh** | ASR + timestamps | zh/en | 220M | [⭐](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) [🤗](https://huggingface.co/funasr/paraformer-zh) |
| Paraformer-zh-streaming | Streaming ASR | zh/en | 220M | [⭐](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary) [🤗](https://huggingface.co/funasr/paraformer-zh-streaming) |
| Qwen3-ASR | ASR, 52 languages | multilingual | 1.7B | [usage](examples/industrial_data_pretraining/qwen3_asr) |
| GLM-ASR-Nano | ASR, 17 languages | multilingual | 1.5B | [usage](examples/industrial_data_pretraining/glm_asr) |
| Whisper-large-v3 | ASR + translation | multilingual | 1550M | [usage](examples/industrial_data_pretraining/whisper) |
| Whisper-large-v3-turbo | ASR + translation | multilingual | 809M | [usage](examples/industrial_data_pretraining/whisper) |
| ct-punc | Punctuation | zh/en | 290M | [⭐](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/summary) [🤗](https://huggingface.co/funasr/ct-punc) |
| fsmn-vad | VAD | zh/en | 0.4M | [⭐](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) [🤗](https://huggingface.co/funasr/fsmn-vad) |
| cam++ | Speaker diarization | — | 7.2M | [⭐](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/summary) [🤗](https://huggingface.co/funasr/campplus) |
| emotion2vec+large | Emotion recognition | — | 300M | [⭐](https://modelscope.cn/models/iic/emotion2vec_plus_large/summary) [🤗](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |

---

## Usage

> Full examples with parameter docs: [Tutorial →](https://modelscope.github.io/FunASR/tutorial.html)

```python
from funasr import AutoModel

# Chinese production (VAD + ASR + punctuation + speaker)
model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", spk_model="cam++", device="cuda")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav", hotword="关键词 20")


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

```bash
# OpenAI-compatible API (recommended)
pip install torch torchaudio
pip install funasr vllm fastapi uvicorn python-multipart
funasr-server --device cuda
# → POST /v1/audio/transcriptions at localhost:8000
```

Verify it with a public sample:

```bash
curl -L https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

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
# Use the windows-x64-cuda package on RTX 30-class GPUs:
.\llama-funasr-sensevoice.exe -m .\gguf\sensevoice-small-q8.gguf --vad .\gguf\fsmn-vad.gguf -a audio.wav --backend cuda
```

Use `funasr-llamacpp-linux-x64-vulkan.tar.gz` on Linux GPU systems with a
working Vulkan driver/ICD:

```bash
./llama-funasr-sensevoice -m ./gguf/sensevoice-small-q8.gguf --vad ./gguf/fsmn-vad.gguf -a audio.wav --backend vulkan
```

The current Windows CUDA package targets CUDA architecture 86. RTX 50 / Blackwell
GPUs report compute capability 12.0 (`sm_120`) and should use the CPU package or
build from source with `-DCMAKE_CUDA_ARCHITECTURES=120` until a dedicated CUDA
asset is published.

**Prebuilt binaries:** [Releases](https://github.com/modelscope/FunASR/releases) · [v0.1.8](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.1.8) · [Linux Vulkan tarball](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.1.8/funasr-llamacpp-linux-x64-vulkan.tar.gz) · [Windows CUDA zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.1.8/funasr-llamacpp-windows-x64-cuda.zip) · **Download & quickstart:** [funasr.com/llama-cpp](https://www.funasr.com/llama-cpp.html) · **GGUF models:** [Hugging Face](https://huggingface.co/FunAudioLLM) · **Docs & benchmarks:** [runtime/llama.cpp/](./runtime/llama.cpp/)

[OpenAI API example →](./examples/openai_api/) · [Gradio demo →](./examples/openai_api/GRADIO.md) · [Client recipes →](./examples/openai_api/CLIENTS.md) · [JavaScript/TypeScript recipes →](./examples/openai_api/JAVASCRIPT.md) · [Kubernetes template →](./examples/openai_api/kubernetes/) · [Workflow recipes →](./examples/openai_api/WORKFLOWS.md) · [Postman collection →](./examples/openai_api/POSTMAN.md) · [OpenAPI spec →](./examples/openai_api/OPENAPI.md) · [Security guide →](./examples/openai_api/SECURITY.md) · [Deployment matrix →](./docs/deployment_matrix.md) · [Deployment docs →](./runtime/readme.md) · [Agent integration →](https://modelscope.github.io/FunASR/agent.html)

---

## Benchmark

> 184 long-form audio files (192 min). [Full report →](https://modelscope.github.io/FunASR/benchmark.html) · [RTFx and reproducibility notes →](./docs/benchmark/rtf_reproducibility.md)

| Model | Chinese CER ↓ | GPU Speed | CPU Speed | vs Whisper-large-v3 |
|-------|------|-----------|-----------|-------------------|
| **Fun-ASR-Nano** (vLLM) | **8.20%** | **340x** realtime | — | 🚀 **26x faster** |
| **SenseVoice-Small** | **7.81%** | **170x** realtime | **17x** realtime | 🚀 **13x faster** |
| **Paraformer-Large** | 10.18% | **120x** realtime | **15x** realtime | 🚀 **9x faster** |
| Whisper-large-v3-turbo | 21.71% | 46x realtime | ❌ | 3.4x faster |
| Whisper-large-v3 | 20.02% | 13x realtime | ❌ | baseline |

> **Key takeaway:** FunASR models run on CPU faster than Whisper runs on GPU.

---

## What's new

- 2026/07/23: **v1.3.26 on PyPI** — `funasr-server --model fun-asr-nano --hub ms` now honors the requested ModelScope hub for the default Fun-ASR-Nano model in both the vLLM path and the AutoModel fallback, avoiding unintended Hugging Face downloads when users choose ModelScope. Install with `python -m pip install -U "funasr==1.3.26"`. [Release ->](https://github.com/modelscope/FunASR/releases/tag/v1.3.26)
- 2026/07/23: **v1.3.25 on PyPI** — realtime WebSocket users can now use deterministic final-text hotword corrections with `POSTPROCESS_HOTWORDS:wrong=>right` or `--postprocess-hotword-file`, keeping fixed-name cleanup separate from model-level `HOTWORDS:` decoding bias. The source-tree realtime entrypoint also works without preinstalling the package. Install with `python -m pip install -U "funasr==1.3.25"`. [Release ->](https://github.com/modelscope/FunASR/releases/tag/v1.3.25)
- 2026/07/23: **v1.3.24 on PyPI** — OpenAI-compatible server deployments now support custom model paths and hub selection, the llama.cpp/GGUF runtime docs include the HTTP transcription wrapper and Linux Vulkan package, and public docs links were refreshed for cleaner onboarding. Install with `python -m pip install -U "funasr==1.3.24"`. [Release ->](https://github.com/modelscope/FunASR/releases/tag/v1.3.24)
- 2026/07/22: **v1.3.23 on PyPI** — packaging and onboarding refresh for this week's community integrations: the PyPI long description now highlights the current OpenAI-compatible server path, llama.cpp/GGUF runtime notes, Windows CUDA architecture guidance, and browser quickstart links shipped in the repository docs. Runtime code is unchanged from v1.3.22. Install with `python -m pip install -U "funasr==1.3.23"`. [Release ->](https://github.com/modelscope/FunASR/releases/tag/v1.3.23)
- 2026/07/22: **llama.cpp runtime v0.1.8** — adds `funasr-llamacpp-linux-x64-vulkan.tar.gz` for SenseVoiceSmall on Linux Vulkan GPUs. Run `llama-funasr-sensevoice ... --backend vulkan`; CPU, AVX2, macOS arm64, Windows CPU/AVX2, and Windows CUDA packages remain available. [Release ->](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.1.8)
- 2026/07/19: **v1.3.22 on PyPI** — `funasr-server` now fills OpenAI-compatible `verbose_json.segments` for text-only SenseVoice/Paraformer fallback responses, so subtitle clients no longer see an empty `segments` array when `text` is populated. Install with `python -m pip install -U "funasr==1.3.22"`. [Release ->](https://github.com/modelscope/FunASR/releases/tag/v1.3.22)
- 2026/07/19: **v1.3.21 on PyPI** — fixes first-import onboarding in fresh environments where users install `funasr` before choosing a platform-specific PyTorch build. `import funasr` and `funasr.__version__` now work without torch; accessing `AutoModel` still requires PyTorch and raises a clear install hint. Install with `python -m pip install -U "funasr==1.3.21"`. [Release ->](https://github.com/modelscope/FunASR/releases/tag/v1.3.21)
- 2026/07/19: **v1.3.20 on PyPI** — PyPI metadata and install guidance now point at the current FunASR docs, community integrations, and quoted `python -m pip install -U "funasr>=1.3.19"` commands for Fun-ASR-Nano deployment paths. This is a documentation/packaging sync; runtime code remains unchanged from v1.3.19. Install with `python -m pip install -U "funasr==1.3.20"`. [Release ->](https://github.com/modelscope/FunASR/releases/tag/v1.3.20)
- 2026/07/19: **v1.3.19 on PyPI** — realtime WebSocket long-session troubleshooting docs are now shipped with the package. Run the server with `--enable-spk --log-session-stats-interval 30` and attach the emitted `Session stats:` lines when reporting disconnects or memory growth. Install with `python -m pip install -U "funasr==1.3.19"`. [Long-session guide ->](docs/vllm_guide.md#long-session-diagnostics) · [Release ->](https://github.com/modelscope/FunASR/releases/tag/v1.3.19)
- 2026/07/19: **v1.3.18 on PyPI** — CLI SRT/TSV subtitle output now requests sentence timestamps and loads punctuation when needed, so `funasr audio.wav --output-format srt --output-dir ./subs` writes segmented subtitle cues instead of one full-text block. Install with `python -m pip install -U "funasr==1.3.18"`. [Release ->](https://github.com/modelscope/FunASR/releases/tag/v1.3.18)
- 2026/07/18: **v1.3.16 on PyPI** — client-driven realtime endpoints for Fun-ASR-Nano. Start one WebSocket session, stream PCM, and send `COMMIT` for each utterance without loading server-side VAD; short utterances finalize and timestamps remain monotonic across commits. Install with `pip install --upgrade funasr`, then run `funasr-realtime-server --endpoint-mode client`. [Guide →](examples/industrial_data_pretraining/fun_asr_nano/docs/realtime_demo.md)
- 2026/07/18: **llama.cpp runtime v0.1.7** — prebuilt Windows CUDA package for SenseVoiceSmall (`funasr-llamacpp-windows-x64-cuda.zip`) plus Linux / macOS / Windows CPU packages. Download the GGUF model, then run `llama-funasr-sensevoice ... --backend cuda` on supported NVIDIA GPUs. [Release →](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.1.7)
- 2026/06/20: **llama.cpp / GGUF runtime** — run SenseVoice / Paraformer / Fun-ASR-Nano on CPU & edge as a single self-contained binary (a whisper.cpp-style alternative), built-in FSMN-VAD, no Python at runtime. Prebuilt binaries for Linux / macOS / Windows + **q8 quantized models (~half the size, same accuracy)**. [runtime/llama.cpp/](./runtime/llama.cpp/) · [Releases](https://github.com/modelscope/FunASR/releases)
- 2026/06/21: **v1.3.12** on PyPI — rolling fixes (qwen3-asr language codes, glm_asr, vLLM repetition_penalty). `pip install --upgrade funasr`
- 2026/05/24: **vLLM Inference Engine** — 2-3x faster LLM decoding for Fun-ASR-Nano. Streaming WebSocket service with VAD + Speaker Diarization. [Guide →](docs/vllm_guide.md) · [Realtime WS tuning →](docs/vllm_guide.md#67-production-concurrency-and-multi-process-deployment) · [API stability checklist →](docs/vllm_guide.md#production-api-stability-checklist)
- 2026/05/24: **Dynamic VAD** — adaptive silence threshold (default on). Short sentences stay intact, long segments get auto-split. [Details →](docs/vllm_guide.md#附录dynamicstreamingvad)
- 2026/05/24: **v1.3.3** — `funasr-server` CLI, OpenAI-compatible API, MCP Server for AI agents. `pip install --upgrade funasr`
- 2026/05/20: Added Qwen3-ASR (0.6B/1.7B) — 52 languages, auto detection. [usage](examples/industrial_data_pretraining/qwen3_asr)
- 2026/05/20: Added GLM-ASR-Nano (1.5B) — 17 languages, dialect support. [usage](examples/industrial_data_pretraining/glm_asr)
- 2026/05/19: Fun-ASR-Nano and SenseVoice can be combined with VAD and CAM++ for speaker diarization.
- 2025/12/15: [Fun-ASR-Nano-2512](https://github.com/FunAudioLLM/Fun-ASR) — Chinese, English, Japanese, and Chinese dialect support; trained on tens of millions of hours.

<details><summary>Older</summary>

- 2024/10/10: Whisper-large-v3-turbo support added.
- 2024/07/04: [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — ASR + emotion + audio events.
- 2024/01/30: FunASR 1.0 released.

</details>

---

## Community

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
