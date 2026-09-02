# 仓库职责与路线图

本文档说明 FunASR 生态四个仓库的职责边界、用户入口和 issue 路由,并给出一份方向性路线图。

> **方向性路线图，不是版本承诺。**
> 本文档记录已交付能力与正在推进的工作，但不承诺未来版本号或日期。当前 Python
> 版本是 [`funasr==1.4.13`](https://github.com/modelscope/FunASR/releases/tag/v1.4.13)。
> 任何未来的 breaking release 仍需 maintainer 确认 milestone 与迁移方案。

---

## 为什么需要这份文档

四个仓库共享同一套模型和工具链,但职责边界此前没有写下来,带来两个实际问题:

1. **Issue 路由错位** —— 模型问题提到工具包,部署问题提到模型仓,来回转派。
2. **重复实现漂移** —— 同一个实时服务在多个仓库各有一份副本,修复只落在其中一处。[#3101](https://github.com/modelscope/FunASR/issues/3101) 就是这样产生的:长会话状态无界的缺陷需要在两个仓库分别修复([#3214](https://github.com/modelscope/FunASR/pull/3214) 与 [QwenAudio/Fun-ASR#135](https://github.com/QwenAudio/Fun-ASR/pull/135))。

---

## 四仓职责

| 仓库 | 核心职责 | 不在这里 |
|---|---|---|
| [modelscope/FunASR](https://github.com/modelscope/FunASR)（工具包 / 运行时） | 框架与推理管线 (pipelines)、训练与微调、VAD / 标点 / ITN / 说话人等组件、**部署服务(含实时 WebSocket 服务)**、`funasr` PyPI 包 | 模型权重与 model card;应用层 UI |
| [QwenAudio/Fun-ASR](https://github.com/QwenAudio/Fun-ASR)（模型仓） | Fun-ASR-Nano / MLT 模型家族与 LLM-ASR 身份:模型说明、权重发布、能力范围(语言 / 方言 / 口音 / 热词 / 时间戳 / 说话人)、模型评测、微调，以及模型级集成(Transformers、vLLM、GGUF) | 服务实现(链接 FunASR,不再自带权威副本) |
| [QwenAudio/SenseVoice](https://github.com/QwenAudio/SenseVoice)（模型仓） | SenseVoice 语音理解基础模型:ASR / 语种识别(LID) / 情感识别(SER) / 音频事件检测(AED),及其模型侧用法 | 通用推理框架;部署服务 |
| [modelscope/FunClip](https://github.com/modelscope/FunClip)（应用层） | 基于 FunASR 的视频转写、字幕生成与 LLM 辅助剪辑;本地 Gradio UI | 底层 ASR 能力与模型问题(上游到 FunASR / 模型仓) |

---

## 用户入口

| 我想… | 去哪里 |
|---|---|
| 用 Python 做语音识别 / 训练 / 微调 | [modelscope/FunASR](https://github.com/modelscope/FunASR) |
| 部署实时流式 ASR 服务，推荐 Fun-ASR-Nano + vLLM 做实时识别 | [modelscope/FunASR/fun_asr_nano](https://github.com/modelscope/FunASR/tree/main/examples/industrial_data_pretraining/fun_asr_nano) —— **推荐实现,见下节** |
| 用一个模型完成长音频多人转写、时间戳与匿名说话人标签 | [MOSS-Transcribe-Diarize 部署指南](./moss_transcribe_diarize_zh.md) —— OpenMOSS 模型通过本地 Transformers 或 vLLM 接入 FunASR，也可通过原生 SGLang Omni 独立服务，不需要额外的外部 VAD 或说话人模型。标签只区分本段录音中的说话人，不能识别已知人物。 |
| 了解 Fun-ASR-Nano / MLT 的能力范围、权重、评测,或使用 Transformers / vLLM / GGUF 集成 | [QwenAudio/Fun-ASR](https://github.com/QwenAudio/Fun-ASR) |
| 需要情感识别 / 音频事件检测 | [QwenAudio/SenseVoice](https://github.com/QwenAudio/SenseVoice) |
| 做视频字幕 / 剪辑 | [modelscope/FunClip](https://github.com/modelscope/FunClip) |

---

## Issue 路由

| 问题类型 | 提到 |
|---|---|
| 框架、推理管线 (pipelines)、训练、微调 | `modelscope/FunASR` |
| 部署服务:实时 WebSocket、离线服务、SDK | `modelscope/FunASR` |
| VAD / 标点 / ITN / 说话人 组件行为 | `modelscope/FunASR` |
| MOSS-Transcribe-Diarize 等第三方模型的 FunASR 适配或部署行为 | `modelscope/FunASR`;模型权重和架构问题仍由上游模型所有者负责 |
| Fun-ASR 系列模型的识别效果、语言支持、权重、评测,以及 Transformers / vLLM / GGUF 等模型级集成 | `QwenAudio/Fun-ASR` |
| SenseVoice 的识别 / 情感 / 事件检测效果 | `QwenAudio/SenseVoice` |
| 视频剪辑、字幕导出、Gradio UI | `modelscope/FunClip` |

**判断法则:换一个模型后问题是否还在?**

- **还在** → 是框架 / 服务问题 → `modelscope/FunASR`
- **只在某个模型上出现** → 是模型问题 → 对应模型仓

---

## 实时 WebSocket 服务:推荐实现

**`modelscope/FunASR` 中的[Fun-ASR-Nano + vLLM 实时 WebSocket 服务](https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py)是推荐实现。**

- 功能开发、缺陷修复、行为变更,**一律先在 `modelscope/FunASR` 落地**。
- 模型仓(`QwenAudio/Fun-ASR`)**只链接到权威实现**,不再把自带副本描述为权威实现。
- 相关 issue 一律提到 `modelscope/FunASR`。

**为什么:** 两份副本各自演进,修复就会只落在其中一处。[#3101](https://github.com/modelscope/FunASR/issues/3101) 已经暴露过这个代价——同一个长会话状态无界的缺陷,需要分别在 [#3214](https://github.com/modelscope/FunASR/pull/3214) 和 [QwenAudio/Fun-ASR#135](https://github.com/QwenAudio/Fun-ASR/pull/135) 修两次。收敛到唯一权威实现是 Next 的一项工作。

---

## 参与路线图

路线图是一组可验收的结果，不是只有维护者才能处理的愿望清单。请从实时的 [help wanted](https://github.com/modelscope/FunASR/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) 和 [ready for PR](https://github.com/modelscope/FunASR/issues?q=is%3Aissue+is%3Aopen+label%3A%22ready+for+PR%22) 查询中选择任务，避免复制很快过期的静态清单。范围较小的任务位于 [good first issue](https://github.com/modelscope/FunASR/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)。

| 标签 | 含义 |
|---|---|
| `good first issue` | 范围已有边界，维护者可以指出相关代码或文档。 |
| `help wanted` | 结果重要，但仍缺维护者没有的硬件、领域知识或实现精力。 |
| `ready for PR` | 预期行为和验收证据已经足够明确，可以开始实现；动手前先留言，避免重复劳动。 |
| `needs feedback` | 正等待报告者或硬件所有者验证结果。仅有 PR 合并或版本发布不能作为关闭 issue 的理由。 |

### 当前需要贡献者的工作

| 方向 | 当前问题 | 验收证据 | 特别有价值的贡献 |
|---|---|---|---|
| [L20 等 GPU 上的实时预览效率](https://github.com/modelscope/FunASR/issues/3528) | 对齐 partial 消息数量后，怎样选择刷新间隔和 partial window，才能在不静默跳过预览的前提下取得合适的延迟/吞吐平衡？ | 基于 exact commit 的客户端 JSONL 和 `--log-decode-profile` 服务端日志；固定 SPK、ping、音频、并发数、partial window 与 partial 消息数量 | 在 L20、L4、A10 或其他非 H100 GPU 上复现，并分析 queue、encoder 与 engine 时间 |
| [AMD Windows Vulkan 稳定性](https://github.com/modelscope/FunASR/issues/3479) | 当前 runtime 能否在报告者的 AMD GPU 上完成模型初始化和转写；若不能，最后成功的初始化边界在哪里？ | 精确压缩包名称和 SHA256、GPU/驱动/Windows 版本、完整初始化日志，以及报告者硬件复测 | AMD Windows 硬件所有者和 Vulkan/llama.cpp 贡献者 |
| [恢复公开 checkpoint 的完整能力](https://github.com/modelscope/FunASR/issues/3496) | 如何从有权限的模型所有者账号发布缺失 CTC tensors，并在上传后完成验证？ | 不可变模型 revision、文件哈希、公开 clean-cache 回下载和真实时间戳/说话人推理 | 有 Hugging Face 写权限的模型所有者和 checkpoint 验证贡献者 |
| [上游模型集成](https://github.com/huggingface/transformers/pull/46180) | 如何让 Fun-ASR-Nano 保持 Transformers 上游兼容，同时保留固定的 model card 和回归测试边界？ | exact-head 上游 CI、聚焦本地测试、model card review 与维护者 review | Transformers reviewer，以及能在合并前验证下游加载的用户 |

### 认领前

1. 读完 issue 的完整时间线，确认没有其他贡献者正在处理。
2. 留言说明可以负责的环境或模块，以及计划提供的证据。
3. 结论必须基于 exact commit、不可变模型 revision 或 release asset，并附可复现命令。
4. 区分 issue 与 PR 的关闭条件：实现可以合并，但报告者验证仍可保持开放。

贡献和 issue 证据可以使用中文或英文。路线图或仓库职责变更应在同一个 PR 中同步更新本文与 [`repository_roles.md`](./repository_roles.md)。

---

## 路线图（方向性）

> 每一项均链接到现有 issue / PR。没有 owner 或验收证据的条目不写完成日期。

### 已交付

- **实时服务长会话状态有界** —— [#3214](https://github.com/modelscope/FunASR/pull/3214) 与 [QwenAudio/Fun-ASR#135](https://github.com/QwenAudio/Fun-ASR/pull/135) 已合并，诊断能力已发布，报告者证据使 [#3101](https://github.com/modelscope/FunASR/issues/3101) 可以关闭。
- **稳定的应用接口** —— 工具包现已提供 OpenAI-compatible 转写服务、健康检查、浏览器与命令行 smoke test，并在[部署矩阵](./deployment_matrix_zh.md)中列出 Python / CLI / HTTP / WebSocket 入口。
- **工业与边缘部署路径** —— vLLM 服务和签名发布流程已有文档；经验证的十平台 [`runtime-llamacpp-v0.2.6`](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.6) 压缩包覆盖 Linux、macOS 与 Windows 的 CPU/GPU 变体，并提供面向 RTX 50 / Blackwell 的 Windows CUDA architecture 120 专用包。
- **联合转写与说话人识别** —— 第三方 [MOSS-Transcribe-Diarize](./moss_transcribe_diarize_zh.md) 模型已通过 `AutoModel` 接入本地 Transformers 与 vLLM 后端，也可通过原生 SGLang Omni 独立服务。SGLang Omni 不是 `AutoModel` backend。它在一次推理中生成时间戳与说话人标签，不需要额外的外部 VAD 或说话人模型；模型所有者仍是 OpenMOSS。
- **仓库职责与 issue 路由** —— [#3203](https://github.com/modelscope/FunASR/issues/3203) 继续跟踪本文档以及尚未回答完的模型权重和 vLLM 入口问题。在这些问题有证据且报告者有合理确认时间之前，issue 保持开放。

### 进行中

- **Fun-ASR-Nano 的 Transformers 原生集成** —— [huggingface/transformers#46180](https://github.com/huggingface/transformers/pull/46180) 正在审查；以该 PR 的 exact-head CI 与 review 状态为准。
- **恢复公开 checkpoint 的完整能力** —— [#3496](https://github.com/modelscope/FunASR/issues/3496) 跟踪 Hugging Face checkpoint 缺少时间戳与说话人路径所需 CTC tensors 的问题。
- **实时预览效率与 L20 验证** —— [#3528](https://github.com/modelscope/FunASR/issues/3528) 已确认 v1.3.9 看似更快，是因为事件循环阻塞时静默跳过了大部分 partial 预览。该 issue 继续开放，用于等工作量的 L20 profiling 和明确的刷新/window 策略；不能把它当成已经解决的吞吐回退。
- **Qwen3-ASR 离线 vLLM 工作流** —— [#3592](https://github.com/modelscope/FunASR/pull/3592) 增加经过验证的原生 `Qwen3ASRModel.LLM` 示例。[#3419](https://github.com/modelscope/FunASR/issues/3419) 继续开放，直到能用精确模型 revision、服务配置和评分脚本复现报告者的 8–9% CER。
- **AMD Windows Vulkan 验证** —— [#3479](https://github.com/modelscope/FunASR/issues/3479) 保持开放，等待报告者在 `runtime-llamacpp-v0.2.6` 上进行硬件复测；发布压缩包不等于硬件崩溃已经修复。

### 下一步

- **把重复实时服务收敛到唯一权威实现**(见上节)，在有兼容性证据后再删除或明确废弃镜像。
- **保持部署矩阵可执行**：每条推荐的 Python / CLI / HTTP / WebSocket / vLLM / llama.cpp 路径都应保留依赖边界、固定测试音频、启动 smoke test 与明确的 CPU/GPU 范围。
- **把容器镜像作为独立验证方向**：只有 CPU/GPU 启动、健康检查、转写与重建测试进入 CI 后，才确定权威镜像和版本标签。本文档不指定集群平台。

### 后续

- 仅在具体接口迁移确实需要 breaking changes 时评估 `2.x`。
- 版本号与发布计划通过 maintainer 确认的 milestone 决定，不在本文档中预测。

---

## 相关链接

- English version: [`repository_roles.md`](./repository_roles.md)
- 贡献指南:[`CONTRIBUTING.md`](../CONTRIBUTING.md)
