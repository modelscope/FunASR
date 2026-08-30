([English](./README.md)|简体中文|[日本語](./README_ja.md)|[한국어](./README_ko.md))

<p align="center">
<a href="https://github.com/modelscope/FunASR"><img src="https://svg-banners.vercel.app/api?type=origin&text1=FunASR🤠&text2=💖%20A%20Fundamental%20End-to-End%20Speech%20Recognition%20Toolkit&width=800&height=210" alt="FunASR"></a>
</p>

<p align="center">
  <strong>面向离线、流式与边缘部署的工业级语音识别工具箱。</strong><br>
  <em>ASR · VAD · 标点 · 说话人 pipeline · 情感与音频事件模型 · OpenAI 兼容服务</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/funasr/"><img src="https://img.shields.io/pypi/v/funasr" alt="PyPI"></a>
  <a href="https://github.com/modelscope/FunASR"><img src="https://img.shields.io/github/stars/modelscope/FunASR?style=social" alt="Stars"></a>
  <a href="https://pypi.org/project/funasr/"><img src="https://img.shields.io/pypi/dm/funasr" alt="Downloads"></a>
  <a href="https://modelscope.github.io/FunASR/zh/"><img src="https://img.shields.io/badge/文档-在线-blue" alt="Docs"></a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/10479" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10479" alt="modelscope%2FFunASR | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> · <a href="./examples/colab/README_zh.md">Colab</a> · <a href="#性能评测">性能评测</a> · <a href="./docs/model_selection_zh.md">模型选择</a> · <a href="./docs/migration_from_whisper_zh.md">迁移指南</a> · <a href="./docs/use_case_showcase_zh.md">场景速览</a> · <a href="./docs/community_projects_zh.md">社区集成</a> · <a href="./docs/deployment_matrix_zh.md">部署选型</a> · <a href="https://www.funasr.com/">部署中心</a> · <a href="./docs/troubleshooting_zh.md">排障 FAQ</a> · <a href="#模型列表">模型列表</a> · <a href="https://modelscope.github.io/FunASR/agent.html">Agent 集成</a> · <a href="./integrations/openclaw/">OpenClaw</a> · <a href="https://modelscope.github.io/FunASR/zh/">文档</a> · <a href="./CONTRIBUTING.md">贡献</a>
</p>

---

## 快速开始

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modelscope/FunASR/blob/main/examples/colab/funasr_quickstart.ipynb)

不想先配置本地环境？可以打开 [Colab 快速体验](./examples/colab/README_zh.md) 在浏览器里转写公开样例或上传自己的音频。

```bash
pip install torch torchaudio
pip install funasr
```

如果要运行 GPU quickstart，请先按 [pytorch.org](https://pytorch.org/get-started/locally/)
选择与你的 NVIDIA driver 匹配的 PyTorch / torchaudio CUDA wheel，再安装 FunASR。
安装后先确认 GPU 可见：

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

只有这里输出 `True` 时才使用 `device="cuda"`；否则请先使用
`device="cpu"`，或重新安装匹配 CUDA 的 PyTorch wheel。

```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad", spk_model="cam++", device="cuda")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")

# AutoModel pipeline 返回带说话人 id 和时间戳的 VAD 分段：
for seg in result[0]["sentence_info"]:
    print(f"[{seg['start']/1000:.1f}s] 说话人{seg['spk']}: {rich_transcription_postprocess(seg['sentence'])}")
```

**输出** — 带说话人标签、时间戳和标点的结构化文本：
```
[0.6s] 说话人0: 欢迎大家来体验达摩院推出的语音识别模型
```

这是一次 `AutoModel` pipeline 调用，实际组合了 SenseVoiceSmall、FSMN-VAD
和 CAM++ 三个独立模型；说话人分离由 CAM++ 提供，并非 SenseVoiceSmall
checkpoint 的内置输出。
SenseVoice 论文见 [arXiv:2407.04051](https://arxiv.org/abs/2407.04051)，
模型见 [Hugging Face checkpoint](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)，
边缘部署可用 [GGUF checkpoint](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF)。

### LLM 语音识别：Fun-ASR-Nano

Fun-ASR-Nano 是基于 SenseVoice 编码器 + Qwen3-0.6B 解码器的 LLM-ASR，
支持中文、英语、日语，以及 7 种中文方言和 26 种地域口音：

```python
from funasr import AutoModel

model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512", vad_model="fsmn-vad", device="cuda")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")
```

需要 31 语种时，请改用独立的
[Fun-ASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512)
checkpoint。Nano 与 MLT-Nano 的语言范围不同，使用时请按 checkpoint 选择。

使用 vLLM 做高吞吐批处理：

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

model = AutoModelVLLM(model="FunAudioLLM/Fun-ASR-Nano-2512", tensor_parallel_size=1)
results = model.generate(["audio1.wav", "audio2.wav"], language="auto")
```

> **部署为 API 服务：** `funasr-server --device cuda` → 本地 OpenAI 兼容接口 localhost:8000
>
> **接入 AI Agent：** [MCP 服务](examples/mcp_server/) 支持 Claude/Cursor · [OpenAI API](examples/openai_api/README_zh.md) 支持 LangChain/Dify/AutoGen
>
> **接入语音 Agent：** [OpenClaw 实时转写插件](integrations/openclaw/) 支持私有部署的 Talk 与 Voice Call 转写

### 为什么选 FunASR？

Whisper 是单个模型，**FunASR 是一个工具箱**——按场景挑模型：
**Fun-ASR-Nano**（中/英/日及中文方言，需 GPU）、
**Fun-ASR-MLT-Nano**（31 语种）、**SenseVoiceSmall**（五语种 ASR，
并返回情感与音频事件标签）、**Paraformer**（低延迟流式）。下表展示的是
工具箱级能力，并标明由哪个模型或 pipeline 提供：

| | FunASR（工具箱） | Whisper | 云端 API |
|---|---|---|---|
| 最高速度 | **340 倍实时**（Fun-ASR-Nano + vLLM） | 13 倍实时 | ~1 倍实时 |
| 说话人识别 | ✅ 由 VAD + CAM++ pipeline 提供 | ❌ 需要 pyannote | ✅ 额外付费 |
| 情感识别 | ✅ 由 SenseVoice 提供 | ❌ | ❌ |
| 语言数 | 取决于 checkpoint（例如 Qwen3-ASR 52、MLT-Nano 31、Nano 中/英/日） | 57 | 因服务而异 |
| 流式识别 | ✅ WebSocket（Paraformer） | ❌ | ✅ |
| CPU 可用 | ✅ 17 倍实时（SenseVoice） | ❌ 太慢 | 不适用 |
| 私有部署 | ✅ 支持（工具箱 MIT；模型协议各异） | ✅ MIT 开源 | ❌ 仅云端 |
| 费用 | 免费 | 免费 | ¥0.04/分钟起 |

第一次试用 FunASR？可以先跑 [Colab 快速体验](./examples/colab/README_zh.md)，再配置本地环境。还不确定先用哪个模型？先看 [模型选择指南](./docs/model_selection_zh.md)。计划从 Whisper 或云端 ASR 切换？请按 [迁移指南](./docs/migration_from_whisper_zh.md) 和 [评测示例](./examples/migration/) 用代表性音频评测、映射功能并安全上线。

---

<a name="性能评测"></a>

## 性能评测

> 184 条长音频（共 192 分钟）。[完整报告 →](https://modelscope.github.io/FunASR/zh/benchmark.html)

| 模型 | 中文 CER ↓ | GPU 速度 | CPU 速度 | 对比 Whisper-large-v3 |
|------|------|----------|----------|---------------------|
| **Fun-ASR-Nano**（vLLM） | **8.20%** | **340 倍**实时 | — | 🚀 **快 26 倍** |
| **SenseVoice-Small** | **7.81%** | **170 倍**实时 | **17 倍**实时 | 🚀 **快 13 倍** |
| **Paraformer-Large** | 10.18% | **120 倍**实时 | **15 倍**实时 | 🚀 **快 9 倍** |
| Whisper-large-v3-turbo | 21.71% | 46 倍实时 | ❌ | 快 3.4 倍 |
| Whisper-large-v3 | 20.02% | 13 倍实时 | ❌ | 基准 |

> **一句话：** FunASR 在 CPU 上的速度，比 Whisper 在 GPU 上还快。

---

## 最新动态

- 2026/08/31：**v1.4.11 已发布到 PyPI** — 修复中韩等多语言可读字幕因时间戳词元携带 SentencePiece `▁` 边界标记而丢失标点的问题。此前一个不可见标记就可能让整段标点与时间戳对齐失败，回退为无标点 VAD 片段，即使顶层识别文本已有标点。报告者的 10 分 33 秒样本现在将 272 个预测标点完整保留到 275 个时间戳句子和 137 条可读字幕中，最长 7.94 秒、最长 42 字；正常对照样本保持对齐。升级命令：`python -m pip install -U "funasr==1.4.11"`；[#3539](https://github.com/modelscope/FunASR/issues/3539) 将继续保持开放，等待报告者确认。[修复 ->](https://github.com/modelscope/FunASR/pull/3587) · [发布页 ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.11)
- 2026/08/31：**v1.4.10 已发布到 PyPI** — 修复超长字幕拆分时破坏中文词语边界的问题。回退算法不再机械填满字符上限，而是在 8 秒和 42 字符的硬限制内平衡时长与长度，并优先选择 Jieba 词边界、标点、文字系统切换和真实时间戳间隙。在报告者同一份 10 分 33 秒中韩混合样本上，84 个源片段被拆为 133 条字幕，全部满足限制，同时保留“钟书成为”“扭了一下”等完整短语。升级命令：`python -m pip install -U "funasr==1.4.10"`；[#3539](https://github.com/modelscope/FunASR/issues/3539) 将继续保持开放，等待报告者确认。[修复 ->](https://github.com/modelscope/FunASR/pull/3583) · [发布页 ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.10)
- 2026/08/30：**v1.4.9 已发布到 PyPI** — 将 v1.4.8 发布后合入的字幕修复交付为可直接安装的版本。可读 SRT 渲染现在会把对齐后的词级时间戳带入源句，并且只在模型的真实时间边界上拆分原本已经过长的句子。在报告者提供的 10 分 33 秒中韩混合样本上，精确合并代码把 84 段、其中 26 段超过 8 秒的字幕变为 131 段；最长不超过 8 秒或 42 个字符，无重叠，也不伪造时间戳。升级命令：`python -m pip install -U "funasr==1.4.9"`；[#3539](https://github.com/modelscope/FunASR/issues/3539) 会保持开放，等待报告者复测。[修复 ->](https://github.com/modelscope/FunASR/pull/3574) · [发布页 ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.9)
- 2026/08/30：**v1.4.8 已发布到 PyPI** — 补齐 v1.4.7 之后合入的第三方 MOSS-Transcribe-Diarize vLLM 能力。`backend="vllm"` 现在可接收官方 `response_format=diarized_json` 说话人分段，并统一为 FunASR `sentence_info`；长会议可通过 `max_completion_tokens` 传递生成上限，文档对超过默认 5120-token 边界的录音给出 8192-token 配置。MOSS 仍然端到端完成转写、时间戳和说话人识别，不要外挂 `vad_model` 或 `spk_model`。升级命令：`python -m pip install -U "funasr==1.4.8"`。GitHub 发布页同时提供已验证的 llama.cpp v0.2.6 十平台运行包。[MOSS 部署指南 ->](./docs/moss_transcribe_diarize_zh.md) · [发布页 ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.8)
- 2026/08/30：**llama.cpp runtime v0.2.6** — 新增面向 RTX 50 / Blackwell 的 Windows CUDA architecture 120（`sm_120`）专用包，同时保留 architecture 86 包。两个 CUDA ZIP 都包含所需的 NVIDIA cuBLAS DLL 与许可证，静态链接 MSVC runtime，并通过 PE 导入审计；十个 Linux、macOS 与 Windows 压缩包由同一个准确发布提交构建，并按公开 SHA-256 逐一复核。构建与打包通过不代表 Blackwell 实机推理已经验证，因此硬件报告会保持开放，等待用户使用匹配资产复测。[实现 →](https://github.com/modelscope/FunASR/pull/3570) · [发布页 →](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.6)
- 2026/08/30：**v1.4.7 已发布到 PyPI** — OpenMOSS 的第三方模型 MOSS-Transcribe-Diarize 已接入 FunASR `AutoModel`。可选择本地 Transformers（`backend="hf"`）或已有 vLLM 服务（`backend="vllm"`）；两条路径都会将模型标签输出统一为 `text`、`raw_text`、毫秒级 `timestamp` 和带说话人标签的 `sentence_info`。MOSS 在一次推理中完成转写、时间戳和说话人分离，因此不要再外挂 `vad_model` 或 `spk_model`。本版本还改善了同一说话人极短间隔处的 SRT 字幕连续性，并增加可选的实时解码性能统计，便于定位长会话延迟。升级命令：`python -m pip install -U "funasr==1.4.7"`。GitHub 发布页同时提供已验证的 llama.cpp v0.2.5 九平台运行包。[MOSS 部署指南 →](./docs/moss_transcribe_diarize_zh.md) · [发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.4.7)
- 2026/08/29：**llama.cpp runtime v0.2.5** — 在计算图执行前，将 host 权重上传到所选 Vulkan backend buffer。Q8 与 F16 权重已通过本地 Linux Vulkan llvmpipe 验证，九个 Linux、macOS 与 Windows 压缩包均由准确提交 `f371370d4c5e4c61d13d4eb9c55cda2f4dd95e4f` 构建，并按公开 SHA-256 逐一复核。本版本不宣称 AMD Windows 硬件崩溃已经修复；[#3479](https://github.com/modelscope/FunASR/issues/3479) 保持开放，等待报告者实机复测。[修复 →](https://github.com/modelscope/FunASR/pull/3555) · [发布页 →](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.5)
- 2026/08/29：**llama.cpp runtime v0.2.4** — 修复 F16 GGUF 模型偶发空转写。运行时现在按 GGML 的 F16/F32 类型解码查询 embedding，不再把 F16 存储误读为 F32。精确的 v0.2.3 AVX2 发布包在 298 次完整运行中复现 22 次空结果；修复后 100/100 次输出稳定唯一，并与 Q8 模型逐字节一致。发布工作流将在同一准确提交上构建九个 Linux、macOS 与 Windows 压缩包。[修复 →](https://github.com/modelscope/FunASR/pull/3550) · [发布页 →](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.4)
- 2026/08/29：**v1.4.6 已发布到 PyPI** — 实时 WebSocket 服务继续每 20 秒发送 ping，但当排队解码延迟 pong 处理时，默认不再误关仍健康的连接；运维人员仍可在测量生产队列与解码延迟后显式配置正数超时。本版本同时避免空英文时间戳片段触发 `IndexError`，并将相邻字幕词组合为更易读的字幕段。升级命令：`python -m pip install -U "funasr==1.4.6"`。GitHub 发布页同时提供已验证的 llama.cpp v0.2.3 九平台运行包。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.4.6)
- 2026/08/29：**llama.cpp runtime v0.2.3** — 后端初始化完成后新增可立即刷新的阶段边界，覆盖模型加载、音频/VAD、计算图构建与分配、推理计算，可继续缩小发生在 `vulkan backend ready` 之后的 Windows AMD Vulkan `0xC0000005` 崩溃范围；本版本不宣称已经修复该硬件相关崩溃。九个 Linux、macOS 与 Windows 压缩包由同一准确提交构建并发布。[排障边界 →](./runtime/llama.cpp/README.md#optional-windows-vulkan-backend-for-sensevoicesmall) · [发布页 →](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.3)
- 2026/08/28：**v1.4.5 已发布到 PyPI** — `torchaudio` 不再是推理的硬依赖。特征提取会优先使用 `torchaudio.compliance.kaldi`，不可用时可切换到可选的 `kaldi-native-fbank` 后端；仍然必须使用 `torchaudio` 的操作会给出可执行的依赖提示。该 fallback 已在 Ascend 910B 上端到端验证：70.47 秒音频用时 1.15 秒（RTF 0.016）。常规升级：`python -m pip install -U "funasr==1.4.5"`；没有 `torchaudio` 的环境请安装：`python -m pip install -U "funasr[knf]==1.4.5"`。GitHub 发布页同时提供已验证的 llama.cpp v0.2.1 九平台运行包。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.4.5)
- 2026/08/27：**llama.cpp runtime v0.2.1** — Vulkan 设备选择现在接受匹配的集成 GPU；同时存在匹配独显时优先独显，否则回退到核显。九个 Linux、macOS 与 Windows 压缩包已重新构建并通过公开 SHA-256 复核。Radeon 780M 仍需报告者实机确认；单独的 RX 9070 XT `0xC0000005` 初始化崩溃不在本次修复声明内。[下载矩阵与快速开始 →](https://www.funasr.com/deploy/llama-cpp.html) · [发布页 →](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.1)
- 2026/08/26：**v1.4.4 已发布到 PyPI** — 实时 WebSocket 解码会对兼容会话进行批处理，不再让所有连接排队经过同一个进程级锁。在 H100 回归负载下，12 路 STOP p95 从 19.8 秒降至 0.4 秒，16 路聚合吞吐从 8.6x 提升到 13.2x，且客户端零错误。本热修复同时兼容不提供 `torch.amp` 的 PyTorch 版本，并让运行时绑定抛出真实异常。升级命令：`python -m pip install -U "funasr==1.4.4"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.4.4)
- 2026/08/21：**v1.4.3 已发布到 PyPI** — `AutoModel(vad_model="silero-vad")` 现可选用 Silero VAD 适配器，返回毫秒级片段，并支持阈值、8/16 kHz 输入、ONNX 模式和最长片段限制。基础升级命令：`python -m pip install -U "funasr==1.4.3"`；启用该适配器：`python -m pip install -U "funasr[silero]==1.4.3"`。已知说话人数的说话人分离在大规模 embedding 输入下改用固定 K 聚类，避免内存开销较高的稠密谱聚类。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.4.3)
- 2026/08/14：**v1.4.2 已发布到 PyPI** — 标点模型的 token 边界落在带时间戳的 ASR 词内部时，句子对齐现在仍能保留正确的字幕分段。分布式训练会在每个梯度累积窗口的最后一个 microbatch 同步 DDP/FSDP 梯度，并从解析后的配置正确初始化 DeepSpeed/FSDP 模式。对应 GitHub 源码 tag 同时包含 llama.cpp SRT 输出和 v0.2.0 AMD Vulkan submission 更新。安装命令：`python -m pip install -U "funasr==1.4.2"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.4.2)
- 2026/08/11：**llama.cpp runtime v0.2.0** — 统一固定上游 llama.cpp 到 `803b7fca`，通过同一套测试工作流发布 9 个带 SHA-256 校验值的 Linux、macOS 与 Windows 压缩包。Fun-ASR-Nano、SenseVoice 和 Paraformer CLI 现在可直接输出 SRT 字幕；Vulkan 启动会给出可操作的 AMD 诊断信息和 CPU fallback。AMD Windows Vulkan 崩溃修复仍等待 issue 报告者在原硬件上确认。[下载矩阵与快速开始 →](https://www.funasr.com/deploy/llama-cpp.html) · [发布页 →](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.0)
- 2026/08/04：**v1.4.1 已发布到 PyPI** — Hugging Face 的 `paraformer-en` 别名现在会解析到官方英文 checkpoint，不再静默下载中文模型。本补丁还包含 Fun-ASR-Nano LoRA 微调与更安全的 checkpoint 处理；对应 GitHub 源码 tag 同时提供 JSONL 时间戳输出、SenseVoice TensorRT 部署和 OpenClaw 实时转写集成。安装命令：`python -m pip install -U "funasr==1.4.1"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.4.1)
- 2026/08/04：**OpenClaw 实时转写集成** — 新增 [`openclaw-funasr`](integrations/openclaw/) 源码包，把私有部署的 FunASR `online`、`offline` 与 `2pass` WebSocket 识别接入 OpenClaw Talk 和 Voice Call。8 kHz G.711 mu-law 转换、60 ms 分帧、partial/final 文本、重连上限、安装包与运行时注册均已基于 OpenClaw `2026.7.2` 验证；npm 与 ClawHub 发布将在所需的[上游 SDK 改动](https://github.com/openclaw/openclaw/pull/118977)合入后进行。
- 2026/07/31：**v1.4.0 已发布到 PyPI** — `AutoModel` 现在会在下载模型前拒绝常见的 `vda_model` 误拼写并明确提示使用 `vad_model`，避免依赖 VAD 的分段、说话人处理和 `sentence_info` 被静默关闭。GitHub 源码发布同时更新 legacy WebSocket 文件运行时：客户端会等待明确的输入结束确认，服务端先刷新待处理的 offline、online 与 2pass 音频，并把收尾失败返回给客户端。Python 包安装命令：`python -m pip install -U "funasr==1.4.0"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.4.0)
- 2026/07/27：**v1.3.30 已发布到 PyPI** — WAV、MP3、FLAC、OGG、MP4/M4A 和 WebM 等容器格式的音频字节现在会通过对应编解码器解码，不再被误当作原始 PCM。OpenAI 兼容响应会保留说话人标签，标点不匹配时仍保留 VAD 分句时间，受信任的浏览器客户端可按需启用 CORS，vLLM 的 VAD 分段上限为 30 秒。GitHub 发布页还同时提供覆盖九种桌面和服务器目标的当前 llama.cpp 预编译运行包。安装命令：`python -m pip install -U "funasr==1.3.30"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.30)
- 2026/07/24：**v1.3.29 热修复已发布到 PyPI** — SenseVoice 长音频在没有词级时间戳和标点模型时，现在会通过 `sentence_info` 返回每个 VAD 语音片段。字幕客户端可直接获得识别文本及真实的毫秒级起止时间，不再退化为零时长或覆盖整段媒体的单条字幕。安装命令：`python -m pip install -U "funasr==1.3.29"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.29)
- 2026/07/24：**v1.3.28 热修复已发布到 PyPI** — 实时 WebSocket 在 VAD 锁句结果退化为短前缀、重复幻觉或解码异常时，会保留连续且完整覆盖当前语音段的干净 partial；短音频 STOP、VAD 收尾和说话人结束现在统一走可靠的完成路径。SenseVoice 字幕分句也会正确对齐富标签、标点与词/BPE 时间戳，不再把中文压成一个字幕块，也不会破坏英文原文。安装命令：`python -m pip install -U "funasr==1.3.28"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.28)
- 2026/07/24：**v1.3.27 已发布到 PyPI** — OpenAI 兼容服务现在会在 `verbose_json` 中返回 SenseVoice 检测到的语言，并在 vLLM 降级后复用已缓存的 Fun-ASR-Nano `AutoModel`。当 vLLM/VAD 初始化及其 fallback 均失败时，不会残留半初始化的 engine 状态，后续请求可以重试。安装命令：`python -m pip install -U "funasr==1.3.27"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.27)
- 2026/07/23：**llama.cpp runtime v0.1.9** — 新增独立的 Windows Vulkan 包 `funasr-llamacpp-windows-x64-vulkan.zip`，支持在安装当前 AMD、Intel 或 NVIDIA Vulkan 驱动的 Windows 机器上运行 SenseVoiceSmall；Linux Vulkan、Windows CUDA、CPU/AVX2、Linux arm64 和 macOS arm64 包继续提供。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.1.9)
- 2026/07/23：**v1.3.26 已发布到 PyPI** — `funasr-server --model fun-asr-nano --hub ms` 现在会在默认 Fun-ASR-Nano 的 vLLM 路径和 AutoModel fallback 路径中都尊重 ModelScope hub 选择，避免用户指定 ModelScope 时仍误走 Hugging Face 下载。安装命令：`python -m pip install -U "funasr==1.3.26"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.26)
- 2026/07/23：**v1.3.25 已发布到 PyPI** — 实时 WebSocket 服务新增 `POSTPROCESS_HOTWORDS:错词=>正确词` 与 `--postprocess-hotword-file`，可在 final 文本阶段做确定性热词纠正，避免把固定错词修正误用成模型层 `HOTWORDS:` 解码偏置；源码目录下的实时服务入口也可直接运行。安装命令：`python -m pip install -U "funasr==1.3.25"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.25)
- 2026/07/23：**v1.3.24 已发布到 PyPI** — OpenAI 兼容服务现在支持自定义模型路径和 hub 选择，llama.cpp/GGUF 文档补充 HTTP 转写 wrapper 与 Linux Vulkan 包，公开文档链接也已刷新，便于新用户顺利上手。安装命令：`python -m pip install -U "funasr==1.3.24"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.24)
- 2026/07/19：**v1.3.22 已发布到 PyPI** — `funasr-server` 现在会为 SenseVoice/Paraformer fallback 的纯文本结果补齐 OpenAI 兼容 `verbose_json.segments`，避免字幕类客户端在 `text` 已有内容时仍拿到空 `segments` 数组。安装命令：`python -m pip install -U "funasr==1.3.22"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.22)
- 2026/07/19：**v1.3.21 已发布到 PyPI** — 修复全新环境里先安装 `funasr`、尚未选择平台对应 PyTorch 版本时的首次导入阻塞。现在 `import funasr` 和 `funasr.__version__` 不再因为缺少 torch 失败；真正访问 `AutoModel` 时仍会要求安装 PyTorch，并给出明确安装提示。安装命令：`python -m pip install -U "funasr==1.3.21"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.21)
- 2026/07/19：**v1.3.20 已发布到 PyPI** — PyPI 项目页和安装引导已同步到当前 FunASR 文档、社区集成列表，以及 Fun-ASR-Nano 部署路径中带引号的 `python -m pip install -U "funasr>=1.3.19"` 命令。本版本是文档/打包元数据同步，运行时代码与 v1.3.19 保持一致。安装命令：`python -m pip install -U "funasr==1.3.20"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.20)
- 2026/07/19：**v1.3.19 已发布到 PyPI** — 实时 WebSocket 长会话排障文档已随包发布。启动服务时加上 `--enable-spk --log-session-stats-interval 30`，如果仍遇到断连或内存增长，请在 issue 中附上输出的 `Session stats:` 日志。安装命令：`python -m pip install -U "funasr==1.3.19"`。[长会话诊断 →](docs/vllm_guide_zh.md#长会话诊断) · [发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.19)
- 2026/07/19：**v1.3.18 已发布到 PyPI** — CLI 的 SRT/TSV 字幕输出现在会请求句级时间戳，并在需要时加载标点模型；`funasr audio.wav --output-format srt --output-dir ./subs` 会输出分句字幕，不再退化成一个全文字幕块。安装命令：`python -m pip install -U "funasr==1.3.18"`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/v1.3.18)
- 2026/07/18：**v1.3.16 已发布到 PyPI** — Fun-ASR-Nano 实时服务新增客户端分句模式。一个 WebSocket 会话可连续发送 PCM，并用 `COMMIT` 提交每个句子；无需加载服务端 VAD，短句可正常结束，多轮时间戳保持递增。执行 `pip install --upgrade funasr` 后，可用 `funasr-realtime-server --endpoint-mode client` 启动。[使用文档 →](examples/industrial_data_pretraining/fun_asr_nano/docs/realtime_demo.md)
- 2026/07/22：**llama.cpp runtime v0.1.8** — 新增 Linux Vulkan 预编译包 `funasr-llamacpp-linux-x64-vulkan.tar.gz`，可在支持 Vulkan driver/ICD 的 Linux GPU 上运行 `llama-funasr-sensevoice ... --backend vulkan`；CPU、AVX2、macOS arm64、Windows CPU/AVX2、Windows CUDA 包继续保留。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.1.8)
- 2026/07/18：**llama.cpp runtime v0.1.7** — 新增 SenseVoiceSmall 的 Windows CUDA 预编译包 `funasr-llamacpp-windows-x64-cuda.zip`，并保留 Linux / macOS / Windows CPU 包。下载 GGUF 模型后，可在支持的 NVIDIA GPU 上运行 `llama-funasr-sensevoice ... --backend cuda`。[发布页 →](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.1.7)
- 2026/05/24：**vLLM 推理引擎** — Fun-ASR-Nano 解码加速 2-3 倍。支持流式 WebSocket 服务（VAD + 说话人分离 + 热词）。[文档 →](docs/vllm_guide_zh.md) · [实时 WS 调优 →](docs/vllm_guide_zh.md#67-生产并发与多进程部署) · [API 稳定性清单 →](docs/vllm_guide_zh.md#生产-api-稳定性清单)
- 2026/05/24：**动态 VAD** — 自适应静音阈值（默认开启），短句不切碎、长句自动切分。[详情 →](docs/vllm_guide_zh.md#7-动态-vad)
- 2026/05/24：**v1.3.3** — `funasr-server` 命令行工具、OpenAI 兼容 API、MCP 服务。`pip install --upgrade funasr`
- 2026/05/20：新增 Qwen3-ASR (0.6B/1.7B)，52 种语言自动检测。[使用方法](examples/industrial_data_pretraining/qwen3_asr)
- 2026/05/20：新增 GLM-ASR-Nano (1.5B)，17 种语言，方言优化。[使用方法](examples/industrial_data_pretraining/glm_asr)
- 2026/05/19：Fun-ASR-Nano 和 SenseVoice 可与 VAD、CAM++ 组合为说话人分离 pipeline。
- 2025/12/15：[Fun-ASR-Nano-2512](https://github.com/QwenAudio/Fun-ASR) 上线，支持中/英/日及中文方言。

<details><summary>更早</summary>

- 2024/10/10：支持 Whisper-large-v3-turbo。
- 2024/07/04：[SenseVoice](https://github.com/QwenAudio/SenseVoice) 发布。
- 2024/01/30：FunASR 1.0 发布。

</details>

---

## 安装

```bash
pip install funasr
```

<details><summary>从源码安装</summary>

```bash
git clone https://github.com/modelscope/FunASR.git && cd FunASR
pip install -e ./
```
环境要求：Python ≥ 3.8、PyTorch ≥ 1.13、torchaudio

</details>

---

<a name="模型列表"></a>

## 模型列表

| 模型 | 任务 | 语言 | 参数量 | 链接 |
|------|------|------|--------|------|
| **Fun-ASR-Nano** | 识别 | 中/英/日 + 中文方言 | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) [GGUF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF) |
| **Fun-ASR-MLT-Nano** | 识别 | 31 种语言 | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) |
| **SenseVoiceSmall** | 识别 + 情感 + 事件 | 中/英/日/韩/粤 | 234M | [⭐](https://www.modelscope.cn/models/iic/SenseVoiceSmall) [🤗](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) [GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF) [论文](https://arxiv.org/abs/2407.04051) |
| **Paraformer-zh** | 识别 + 时间戳 | 中/英 | 220M | [⭐](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) [🤗](https://huggingface.co/funasr/paraformer-zh) |
| Paraformer-zh-streaming | 流式识别 | 中/英 | 220M | [⭐](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary) [🤗](https://huggingface.co/funasr/paraformer-zh-streaming) |
| Qwen3-ASR | 识别，52 种语言 | 多语言 | 1.7B | [使用](examples/industrial_data_pretraining/qwen3_asr) |
| GLM-ASR-Nano | 识别，17 种语言 | 多语言 | 1.5B | [使用](examples/industrial_data_pretraining/glm_asr) |
| Whisper-large-v3 | 识别 + 翻译 | 多语言 | 1550M | [使用](examples/industrial_data_pretraining/whisper) |
| Whisper-large-v3-turbo | 识别 + 翻译 | 多语言 | 809M | [使用](examples/industrial_data_pretraining/whisper) |
| ct-punc | 标点恢复 | 中/英 | 290M | [⭐](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/summary) [🤗](https://huggingface.co/funasr/ct-punc) |
| fsmn-vad | 语音检测 | 中/英 | 0.4M | [⭐](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) [🤗](https://huggingface.co/funasr/fsmn-vad) |
| cam++ | 说话人分离 | — | 7.2M | [⭐](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/summary) [🤗](https://huggingface.co/funasr/campplus) |
| emotion2vec+large | 情感识别 | — | 300M | [⭐](https://modelscope.cn/models/iic/emotion2vec_plus_large/summary) [🤗](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |

---

## 使用示例

> 完整参数文档：[教程 →](https://modelscope.github.io/FunASR/zh/tutorial.html)

```python
from funasr import AutoModel

# 中文生产级（VAD + 识别 + 标点 + 说话人）
model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", spk_model="cam++", device="cuda")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav", hotword="关键词 20")

# 使用 Silero VAD（先安装：python -m pip install "funasr[silero]"）
model = AutoModel(
    model="paraformer-zh", vad_model="silero-vad", device="cpu",
    vad_kwargs={"silero_threshold": 0.5, "silero_min_silence_duration_ms": 100},
)
result = model.generate(input="audio.wav")

# 中/英/日 + 中文方言
model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512", hub="hf", trust_remote_code=True,
                  vad_model="fsmn-vad", vad_kwargs={"max_single_segment_time": 30000}, device="cuda")
result = model.generate(input="audio.wav", batch_size=1)

# 流式实时识别(逐块喂音频)
import soundfile as sf
model = AutoModel(model="paraformer-zh-streaming", device="cuda")
audio, sr = sf.read("speech.wav", dtype="float32")   # 16 kHz 单声道
chunk_size = [0, 10, 5]                               # 每块 600ms
chunk_stride = chunk_size[1] * 960
cache = {}
n_chunks = (len(audio) - 1) // chunk_stride + 1
for i in range(n_chunks):
    chunk = audio[i * chunk_stride : (i + 1) * chunk_stride]
    res = model.generate(input=chunk, cache=cache, is_final=(i == n_chunks - 1),
                         chunk_size=chunk_size, encoder_chunk_look_back=4, decoder_chunk_look_back=1)
    if res[0]["text"]:
        print(res[0]["text"], end="", flush=True)

# 情感识别
model = AutoModel(model="emotion2vec_plus_large", device="cuda")
result = model.generate(input="audio.wav", granularity="utterance")
```

### 命令行工具（Agent 友好）

```bash
# 转写音频（最简用法）
funasr audio.wav

# JSON 输出（适合 AI Agent 调用）
funasr audio.wav --output-format json

# 生成 SRT 字幕
funasr audio.wav --output-format srt --output-dir ./subs

# 说话人分离 + 时间戳
funasr audio.wav --spk --timestamps -f json

# 指定模型和语言
funasr audio.wav --model paraformer --language zh

# 批量转写
funasr *.wav --output-format srt --output-dir ./output
```

可用模型：`sensevoice`（默认）、`paraformer`、`paraformer-en`、`fun-asr-nano`


---

## 部署

```bash
# OpenAI 兼容 API（推荐）
pip install funasr fastapi uvicorn python-multipart
funasr-server --model sensevoice --device cuda
# → POST /v1/audio/transcriptions，地址 localhost:8000
```

使用公开样例音频验证服务：

```bash
curl -L https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

```bash
# Docker 流式服务
docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.12
```

### CPU / 边缘部署 - llama.cpp / GGUF（无需 GPU、无需 Python）

在 CPU 和边缘设备上用单个自包含二进制运行 **SenseVoice / Paraformer / Fun-ASR-Nano**，无需 Python 运行环境，并内置 FSMN-VAD。

```bash
# Linux / macOS：在解压后的发布目录中执行
bash download-funasr-model.sh sensevoice ./gguf        # 也可使用 paraformer 或 nano
./llama-funasr-sensevoice -m ./gguf/sensevoice-small-q8.gguf --vad ./gguf/fsmn-vad.gguf -a audio.wav
# -> 欢迎大家来体验达摩院推出的语音识别模型
```

```powershell
# Windows PowerShell：在解压根目录执行（需已安装 `hf` CLI）
hf download FunAudioLLM/SenseVoiceSmall-GGUF sensevoice-small-q8.gguf --local-dir .\gguf
hf download FunAudioLLM/fsmn-vad-GGUF fsmn-vad.gguf --local-dir .\gguf
.\llama-funasr-sensevoice.exe -m .\gguf\sensevoice-small-q8.gguf --vad .\gguf\fsmn-vad.gguf -a audio.wav
# 使用 windows-x64-vulkan 包，并安装 AMD、Intel 或 NVIDIA 的当前 Vulkan 显卡驱动：
.\llama-funasr-sensevoice.exe -m .\gguf\sensevoice-small-q8.gguf --vad .\gguf\fsmn-vad.gguf -a audio.wav --backend vulkan
# RTX 30 系列等架构 86 GPU 可使用 windows-x64-cuda 包：
.\llama-funasr-sensevoice.exe -m .\gguf\sensevoice-small-q8.gguf --vad .\gguf\fsmn-vad.gguf -a audio.wav --backend cuda
```

Linux GPU 用户可下载 `funasr-llamacpp-linux-x64-vulkan.tar.gz`，在已安装可用
Vulkan driver/ICD 的机器上运行：

```bash
./llama-funasr-sensevoice -m ./gguf/sensevoice-small-q8.gguf --vad ./gguf/fsmn-vad.gguf -a audio.wav --backend vulkan
```

Windows Vulkan ZIP 使用显卡驱动提供的系统 Vulkan loader，不需要另外安装 Vulkan SDK；当前与 Linux Vulkan 包一样，仅加速 SenseVoiceSmall。

带 tag 的发布提供两个 Windows CUDA 包：标准 `windows-x64-cuda` ZIP 面向 CUDA
architecture 86，`windows-x64-cuda-blackwell` 面向 RTX 50 / Blackwell 的 architecture
120（`sm_120`）。两个 ZIP 都包含所需的 cuBLAS DLL，并使用静态 MSVC runtime；用户只需
安装兼容的 NVIDIA 驱动，无需另装 CUDA Toolkit。CI 验证架构与打包边界，但不代表已经在
Blackwell 实机上完成推理验证。

**预编译二进制：** [Releases](https://github.com/modelscope/FunASR/releases) · [v0.2.6](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.6) · [Linux Vulkan tarball](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-linux-x64-vulkan.tar.gz) · [Windows Vulkan zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-vulkan.zip) · [Windows CUDA zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-cuda.zip) · [Windows Blackwell CUDA zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-cuda-blackwell.zip) · **下载与快速开始：** [funasr.com/deploy/llama-cpp](https://www.funasr.com/deploy/llama-cpp.html) · **GGUF 模型：** [Hugging Face](https://huggingface.co/FunAudioLLM) · **文档与评测：** [runtime/llama.cpp/](./runtime/llama.cpp/)

[OpenAI API 示例 →](./examples/openai_api/README_zh.md) · [Gradio Demo →](./examples/openai_api/GRADIO_zh.md) · [客户端配方 →](./examples/openai_api/CLIENTS.md) · [JavaScript/TypeScript 配方 →](./examples/openai_api/JAVASCRIPT_zh.md) · [Kubernetes 模板 →](./examples/openai_api/kubernetes/README_zh.md) · [工作流配方 →](./examples/openai_api/WORKFLOWS_zh.md) · [Postman 集合 →](./examples/openai_api/POSTMAN_zh.md) · [OpenAPI 规范 →](./examples/openai_api/OPENAPI_zh.md) · [安全指南 →](./examples/openai_api/SECURITY_zh.md) · [部署选型 →](./docs/deployment_matrix_zh.md) · [部署文档 →](./runtime/readme_cn.md) · [Agent 集成 →](https://modelscope.github.io/FunASR/agent.html)

---

## 社区

|  |  |
|---|---|
| 📖 [文档](https://modelscope.github.io/FunASR/zh/) | 🐛 [问题反馈](https://github.com/modelscope/FunASR/issues) |
| 💬 [讨论](https://github.com/modelscope/FunASR/discussions) | 🤗 [HuggingFace](https://huggingface.co/funasr) |
| 🤝 [贡献指南](./CONTRIBUTING.md) | 📈 [20k 增长计划](./docs/community_growth_20k.md) |
| 🗺️ [仓库职责与路线图](./docs/repository_roles_zh.md) | 🌐 [funasr.com](https://www.funasr.com) |
| 🧩 [社区集成](./docs/community_projects_zh.md) | 💡 [使用案例](./docs/use_case_showcase_zh.md) |

## Star 趋势

<a href="https://star-history.com/#modelscope/FunASR&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=modelscope/FunASR&type=Date" width="600" />
 </picture>
</a>

## 许可证

- 本仓库的 FunASR 工具箱源码：[MIT License](./LICENSE)。
- 预训练模型权重单独授权，请以各模型卡标注的协议为准；模型卡若链接本仓库的 [FunASR 模型开源协议](./MODEL_LICENSE)，则适用该协议。

## 引用

```bibtex
@inproceedings{gao2023funasr,
  author={Zhifu Gao and others},
  title={FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
  booktitle={INTERSPEECH},
  year={2023}
}
```
