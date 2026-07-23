# 排障 FAQ

这份简短 FAQ 汇总首次安装和部署 FunASR 时最常见的阻塞问题。模型选择请看[模型选择指南](./model_selection_zh.md)，服务化选型请看[部署选型](./deployment_matrix_zh.md)。

## 最近 issue 里的 Top 支持问题

最近 issue 巡检显示，最容易阻塞首次试用的三类问题是：

- **应该用哪个安装命令、模型 id 或 hub？** 看[安装或 import 失败](#安装或-import-失败)和[模型下载慢或失败](#模型下载慢或失败)。对应 #3321、#3045、#3042、#2973、#2976 等问题。
- **CPU、CUDA、Vulkan、GGUF 应该下载哪个 runtime 包？** 看[llama.cpp 或 GGUF runtime 无法启动](#llamacpp-或-gguf-runtime-无法启动)。对应 #3298、#3297、#3296、#3289、#3243 等问题。
- **实时服务、VAD、vLLM 或 server 输出为什么延迟、为空，或和本地 Python 不一致？** 看[`funasr-server` 启动后 OpenAI 兼容接口请求失败](#funasr-server-启动后-openai-兼容接口请求失败)和[WebSocket 实时输出为空或延迟很大](#websocket-实时输出为空或延迟很大)。对应 #3101、#3109、#3038、#3031、#2968、#2965 等问题。

## 安装或 import 失败

- 先安装 `torch` 和 `torchaudio`，再安装 FunASR：

```bash
python -m pip install -U torch torchaudio
python -m pip install -U "funasr==1.3.26"
```

- 保持 `torch`、`torchaudio`、`torchvision` 来自同一安装渠道且版本兼容。如果使用 vLLM，请按 [vLLM 指南](./vllm_guide_zh.md)配置，避免在同一环境里混装不匹配的 CUDA wheel。
- 如果仍然 import 失败，建议新建干净虚拟环境复现，并在 **Deployment Help** issue 里附上 Python 版本、操作系统、CUDA driver、`pip list | grep -E "torch|torchaudio|funasr"` 和完整 traceback。

## 模型下载慢或失败

- 中国大陆网络优先尝试 ModelScope。README 和 model_zoo 里的 `iic/...` 模型名是当前入口；命令支持 hub 参数时可以选择 ModelScope。
- 海外网络通常 Hugging Face 更快。GGUF 和边缘 runtime 模型请使用 Hugging Face 上的 FunAudioLLM 公开仓库。
- 下载中断后，只清理该模型的半截缓存再重试。提交 **Deployment Help** issue 时请附 hub、model id、网络环境和错误日志。

## `funasr-server` 启动后 OpenAI 兼容接口请求失败

- 确认服务依赖已安装，包括 FastAPI、Uvicorn 和 multipart upload 支持。
- 接入 agent 或 SDK 前，先用一个很短的本地 WAV 文件 smoke test `/v1/audio/transcriptions`：

```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/transcriptions" \
  -F "file=@example.wav" \
  -F "model=FunAudioLLM/SenseVoiceSmall"
```

- 如果 `/v1/audio/transcriptions` 返回 4xx 或 5xx，请附启动命令、完整 server log、请求命令、model id、hub 和音频时长。

## WebSocket 实时输出为空或延迟很大

- 检查客户端发送的音频格式是否符合 WebSocket demo 要求，尤其是采样率、通道数、chunk size 和 PCM 编码。
- 先用一段已知可用的短 WAV 文件排查。长静音、不支持的编码、采样率不匹配，都可能看起来像服务端失败。
- 提交 **Deployment Help** 时，请附 WebSocket URL、客户端命令或浏览器 console、model id、采样率和服务端 session statistics。

## llama.cpp 或 GGUF runtime 无法启动

- 从 README 或 [funasr.com/llama-cpp](https://www.funasr.com/llama-cpp.html) 下载当前 `runtime-llamacpp-v0.1.9` release 包。
- 按机器环境选择包：CPU 包兼容性最好，Vulkan 包需要可用 Vulkan runtime，CUDA 包需要兼容的 NVIDIA driver。
- GGUF 模型请使用 Hugging Face 上当前的公开仓库，例如 `FunAudioLLM/Fun-ASR-Nano-GGUF` 或 `FunAudioLLM/SenseVoiceSmall-GGUF`。
- GPU 问题请在 **Deployment Help** issue 里附 `nvidia-smi`、操作系统、driver 版本、runtime 包名、模型文件名和完整 llama.cpp 命令。

## Deployment Help issue 需要提供什么

请尽量提供：

- 操作系统、Python 版本、安装命令和虚拟环境工具；
- `torch`、`torchaudio`、CUDA、driver 和 GPU 信息；
- FunASR 版本、model id、hub（`ModelScope` 或 `Hugging Face`）和部署方式；
- 精确命令、最小音频样例信息、完整错误日志，以及同一音频在本地 Python pipeline 是否可用。
