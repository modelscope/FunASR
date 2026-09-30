# Troubleshooting

This short FAQ covers the install and deployment failures that most often block a first FunASR trial. For model choice, see the [model selection guide](./model_selection.md); for serving options, see the [deployment matrix](./deployment_matrix.md).

## Top support questions from recent issues

Recent issue triage shows these common first-run blockers:

- **Which install or hub path should I use?** See [Install or import fails](#install-or-import-fails) and [Model download is slow or fails](#model-download-is-slow-or-fails). This covers reports like #3321, #3045, #3042, #2973, and #2976.
- **Which runtime package should I run on CPU, CUDA, Vulkan, or GGUF?** See [llama.cpp or GGUF runtime does not start](#llamacpp-or-gguf-runtime-does-not-start). This covers reports like #3298, #3297, #3296, #3289, and #3243.
- **Why is realtime, VAD, vLLM, or server output delayed, empty, or different from local Python?** See [`funasr-server` starts but OpenAI-compatible requests fail](#funasr-server-starts-but-openai-compatible-requests-fail) and [WebSocket realtime output is empty or delayed](#websocket-realtime-output-is-empty-or-delayed). This covers reports like #3101, #3109, #3038, #3031, #2968, and #2965.

## Install or import fails

- Follow the [Python SDK installation guide](./installation/installation.md). Install matching `torch` and `torchaudio` builds for your OS, interpreter and accelerator before choosing a released FunASR package or a source checkout. A generic upgrade command does not select the right CUDA build for every environment.
- Keep PyTorch-family packages compatible. If you use vLLM, follow [the vLLM guide](./vllm_guide.md) and avoid mixing unrelated CUDA wheels in the same environment.
- Check the interpreter used by your IDE as well as your terminal. Use `python -m pip --version` and `python -m pip check` in that environment. Include the versions, paths and exact traceback when reporting an import failure; redact personal directory names if needed.

## Is AutoModel using the GPU?

For the PyTorch `AutoModel` path, there is no separate "FunASR GPU edition". A system CUDA installation or working llama.cpp GPU backend does not establish CUDA support in the Python environment running FunASR. Low whole-GPU utilization alone cannot identify the model's device.

Add this after your existing `model = AutoModel(...)` call, using the same interpreter and run configuration. It inspects already-loaded objects and does not load another model:

```python
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from itertools import chain
import torch

print("Python:", sys.executable)
try:
    print("FunASR:", version("funasr"))
except PackageNotFoundError:
    print("FunASR: no installed package metadata; check source checkout/PYTHONPATH")
print("PyTorch:", torch.__version__, torch.__file__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
print("ASR resolved device:", model.kwargs.get("device"))

for name, attr in [("ASR", "model"), ("VAD", "vad_model"),
                   ("PUNC", "punc_model"), ("SPK", "spk_model")]:
    module = getattr(model, attr, None)
    if module is None:
        print(name, "not enabled")
    elif isinstance(module, torch.nn.Module):
        devices = sorted({str(t.device) for t in chain(module.parameters(), module.buffers())})
        print(name, "parameter/buffer devices:", devices or ["no parameters/buffers"])
    else:
        print(name, "non-PyTorch module:", type(module).__name__)
```

- CUDA build `None`: this PyTorch build has no CUDA support. Changing `cuda` to `cuda:0` cannot add it.
- A CUDA build version with availability `False`: check driver compatibility, device visibility and the actual interpreter before changing FunASR parameters.
- Parameter/buffer devices such as `cuda:0`: those tensors are on the GPU. This does not mean every preprocessing operation runs there or that utilization stays high. Check ASR, VAD, punctuation and speaker models separately; an empty tensor list is not proof of CPU placement.
- Resolved device `cpu`: in the [current implementation](../funasr/auto/auto_model.py), unavailable CUDA or `ngpu=0` causes CPU fallback. Check the installed version and actual settings. A requested device is not proof of effective placement.

For a support request, paste this text output and the selected model/configuration rather than only Task Manager screenshots. Whole-card memory includes other processes. Redact personal paths; audio is not needed for this initial environment check. Use the installation guide's official PyTorch links if a different build is needed; do not uninstall ONNX Runtime to diagnose this PyTorch path.

## Model download is slow or fails

- In mainland China, try ModelScope first. Use the `iic/...` model names shown in the README and model zoo, or pass the ModelScope hub option when a command exposes it.
- Outside mainland China, Hugging Face mirrors are often faster. For GGUF or edge runtime models, use the public FunAudioLLM repositories on Hugging Face.
- If a download is interrupted, clear only the partial cache for that model and retry. Include the hub, model id, network environment, and error log in a **Deployment Help** issue.

## `funasr-server` starts but OpenAI-compatible requests fail

- Confirm the server extras are installed, including FastAPI, Uvicorn, and multipart upload support.
- Smoke test the health of the OpenAI-compatible transcription route with a small local WAV file before wiring it into an agent or SDK:

```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/transcriptions" \
  -F "file=@example.wav" \
  -F "model=FunAudioLLM/SenseVoiceSmall"
```

- If the curl request works but a browser reports a CORS or network error, restart the server with the browser page's exact origin (scheme, host, and port):

```bash
funasr-server --device cpu --model sensevoice \
  --cors-origin http://localhost:3000
```

- Repeat `--cors-origin` for each trusted browser origin, for example when both `localhost` and `127.0.0.1` are used. Browser CORS access is disabled by default; avoid a wildcard on machines reachable by other users.
- If `/v1/audio/transcriptions` returns 4xx or 5xx, attach the startup command, full server log, request command, model id, hub, and audio duration.

## WebSocket realtime output is empty or delayed

- Check that the client sends the audio format expected by the WebSocket demo, especially sample rate, channel count, chunk size, and PCM encoding.
- Use a short known-good WAV first. Long silence, unsupported codecs, or mismatched sample rates can look like a serving failure.
- When filing **Deployment Help**, include the WebSocket URL, client command or browser console output, model id, sample rate, and the server-side session statistics.

## llama.cpp or GGUF runtime does not start

- Download the current `runtime-llamacpp-v0.1.9` release package from the README or [funasr.com/llama-cpp](https://www.funasr.com/llama-cpp.html).
- Match the package to the machine: CPU builds work broadly, Vulkan builds need a working Vulkan runtime, and CUDA builds need compatible NVIDIA drivers.
- Use the current GGUF model repositories on Hugging Face, such as `FunAudioLLM/Fun-ASR-Nano-GGUF` or `FunAudioLLM/SenseVoiceSmall-GGUF`.
- For GPU issues, include `nvidia-smi`, operating system, driver version, runtime package name, model file name, and the complete llama.cpp command in a **Deployment Help** issue.

## What to include in a Deployment Help issue

Please include:

- operating system, Python version, install command, and virtual environment tool;
- `torch`, `torchaudio`, CUDA, driver, and GPU details;
- FunASR version, model id, hub (`ModelScope` or `Hugging Face`), and deployment mode;
- exact command, minimal audio sample details, full error log, and whether the same sample works in the local Python pipeline.
