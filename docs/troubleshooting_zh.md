# 排障 FAQ

这份简短 FAQ 汇总首次安装和部署 FunASR 时最常见的阻塞问题。模型选择请看[模型选择指南](./model_selection_zh.md)，服务化选型请看[部署选型](./deployment_matrix_zh.md)。

## 最近 issue 里的 Top 支持问题

最近 issue 巡检显示，首次试用常见的阻塞问题包括：

- **应该用哪个安装命令、模型 id 或 hub？** 看[安装或 import 失败](#安装或-import-失败)和[模型下载慢或失败](#模型下载慢或失败)。对应 #3321、#3045、#3042、#2973、#2976 等问题。
- **CPU、CUDA、Vulkan、GGUF 应该下载哪个 runtime 包？** 看[llama.cpp 或 GGUF runtime 无法启动](#llamacpp-或-gguf-runtime-无法启动)。对应 #3298、#3297、#3296、#3289、#3243 等问题。
- **实时服务、VAD、vLLM 或 server 输出为什么延迟、为空，或和本地 Python 不一致？** 看[`funasr-server` 启动后 OpenAI 兼容接口请求失败](#funasr-server-启动后-openai-兼容接口请求失败)和[WebSocket 实时输出为空或延迟很大](#websocket-实时输出为空或延迟很大)。对应 #3101、#3109、#3038、#3031、#2968、#2965 等问题。
- **电话车牌、方言、重复确认语音怎么微调和评测？** 看[电话车牌语音的微调与评测](./vehicle_plate_finetuning_zh.md)。热词是软提示，不是确定性车牌纠错。

## 安装或 import 失败

- 按[Python SDK 安装指南](./installation/installation_zh.md)操作：先安装与操作系统、解释器和加速设备匹配的 `torch`、`torchaudio`，再选择已发布的 FunASR 包或源码安装。通用升级命令不能替所有环境选择正确的 CUDA 构建。
- 保持 PyTorch 系列包的版本兼容。如果使用 vLLM，请按 [vLLM 指南](./vllm_guide_zh.md)配置，避免混装不匹配的 CUDA wheel。
- 同时确认 IDE 和终端实际使用的解释器。在对应环境运行 `python -m pip --version` 和 `python -m pip check`。报告 import 问题时附上版本、路径和完整 traceback；路径中的个人信息可以打码。

## AutoModel 是否真的在使用 GPU

PyTorch `AutoModel` 路径不需要另装“FunASR GPU 版”。系统装有 CUDA、或者 llama.cpp 的 GPU 后端可用，都不能证明运行 FunASR 的 Python 环境支持 CUDA。仅凭整张显卡的利用率低，也无法判断模型设备。

把下面代码放在已有的 `model = AutoModel(...)` 之后，使用原来的解释器和运行配置执行。它只检查已加载对象，不会再次加载模型：

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

- CUDA build 为 `None`：当前 PyTorch 构建不含 CUDA。把 `cuda` 改为 `cuda:0` 不能补上 CUDA 支持。
- CUDA build 有版本、availability 为 `False`：先检查驱动兼容性、设备可见性及实际解释器，再判断是否为 FunASR 参数问题。
- 参数/缓冲区设备为 `cuda:0`：这些张量确实在 GPU 上，但不代表每个预处理步骤都使用 GPU，或利用率必须一直很高。ASR、VAD、标点和说话人模型分别查看；张量列表为空也不能证明它在 CPU 上。
- resolved device 为 `cpu`：[当前实现](../funasr/auto/auto_model.py)在 CUDA 不可用或 `ngpu=0` 时会回退 CPU。请结合已安装版本和实际配置排查；请求的设备不等于最终设备。

求助时请贴上述文本输出和所选模型/配置，不要只发任务管理器截图。整卡显存包含其他进程；路径中的个人信息可以打码，初步环境检查不需要录音。确需更换 PyTorch 构建时，使用安装指南中的官方链接；不要通过卸载 ONNX Runtime 来诊断这条 PyTorch 路径。

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

- 如果 curl 成功，但浏览器报 CORS 或 network error，请按浏览器页面的精确 origin（scheme、host、port）重启服务：

```bash
funasr-server --device cpu --model sensevoice \
  --cors-origin http://localhost:3000
```

- 每个可信浏览器 origin 都要重复传入一次 `--cors-origin`，例如同时使用 `localhost` 和 `127.0.0.1` 时。浏览器 CORS 默认关闭；机器可被其他用户访问时不要使用通配符。
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
