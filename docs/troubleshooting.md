# Troubleshooting

This short FAQ covers the install and deployment failures that most often block a first FunASR trial. For model choice, see the [model selection guide](./model_selection.md); for serving options, see the [deployment matrix](./deployment_matrix.md).

## Install or import fails

- Install `torch` and `torchaudio` first, then install FunASR:

```bash
python -m pip install -U torch torchaudio
python -m pip install -U "funasr==1.3.26"
```

- Keep `torch`, `torchaudio`, and `torchvision` on compatible versions from the same install channel. If you use vLLM, follow [the vLLM guide](./vllm_guide.md) and avoid mixing unrelated CUDA wheels in the same environment.
- If import still fails, create a fresh virtual environment and include the complete Python version, operating system, CUDA driver, `pip list | grep -E "torch|torchaudio|funasr"`, and the exact traceback in a **Deployment Help** issue.

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

- If `/v1/audio/transcriptions` returns 4xx or 5xx, attach the startup command, full server log, request command, model id, hub, and audio duration.

## WebSocket realtime output is empty or delayed

- Check that the client sends the audio format expected by the WebSocket demo, especially sample rate, channel count, chunk size, and PCM encoding.
- Use a short known-good WAV first. Long silence, unsupported codecs, or mismatched sample rates can look like a serving failure.
- When filing **Deployment Help**, include the WebSocket URL, client command or browser console output, model id, sample rate, and the server-side session statistics.

## llama.cpp or GGUF runtime does not start

- Download the current `runtime-llamacpp-v0.1.8` release package from the README or [funasr.com/llama-cpp](https://www.funasr.com/llama-cpp.html).
- Match the package to the machine: CPU builds work broadly, Vulkan builds need a working Vulkan runtime, and CUDA builds need compatible NVIDIA drivers.
- Use the current GGUF model repositories on Hugging Face, such as `FunAudioLLM/Fun-ASR-Nano-GGUF` or `FunAudioLLM/SenseVoiceSmall-GGUF`.
- For GPU issues, include `nvidia-smi`, operating system, driver version, runtime package name, model file name, and the complete llama.cpp command in a **Deployment Help** issue.

## What to include in a Deployment Help issue

Please include:

- operating system, Python version, install command, and virtual environment tool;
- `torch`, `torchaudio`, CUDA, driver, and GPU details;
- FunASR version, model id, hub (`ModelScope` or `Hugging Face`), and deployment mode;
- exact command, minimal audio sample details, full error log, and whether the same sample works in the local Python pipeline.
