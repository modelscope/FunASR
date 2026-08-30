# 仓库职责与路线图

本文档说明 FunASR 生态四个仓库的职责边界、用户入口和 issue 路由,并给出一份方向性路线图。

> **方向性路线图，不是版本承诺。**
> 本文档记录已交付能力与正在推进的工作，但不承诺未来版本号或日期。当前 Python
> 版本是 [`funasr==1.4.9`](https://github.com/modelscope/FunASR/releases/tag/v1.4.9)。
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
| 用一个模型完成长音频多人转写、时间戳与说话人身份识别 | [MOSS-Transcribe-Diarize 部署指南](./moss_transcribe_diarize_zh.md) —— OpenMOSS 模型通过本地 Transformers 或 vLLM 接入 FunASR，也可通过原生 SGLang Omni 独立服务，不需要额外的外部 VAD 或说话人模型 |
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
- **实时并发性能回归** —— [#3528](https://github.com/modelscope/FunASR/issues/3528) 保持开放，等待可复现的压测证据与有边界的修复。
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
