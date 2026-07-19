# 仓库职责与路线图

本文档说明 FunASR 生态四个仓库的职责边界、用户入口和 issue 路由,并给出一份方向性路线图。

> **版本化发布路线图：待维护者确认。**
> 本文档不承诺版本号和发布日期。当前仓库没有已建立的 milestone;GitHub tag 至 `v1.3.13`,PyPI 已发布 `1.3.14`;`1.4 / 1.5 / 2.0` 的边界尚未经核心 maintainer 确认。

---

## 为什么需要这份文档

四个仓库共享同一套模型和工具链,但职责边界此前没有写下来,带来两个实际问题:

1. **Issue 路由错位** —— 模型问题提到工具包,部署问题提到模型仓,来回转派。
2. **重复实现漂移** —— 同一个实时服务在多个仓库各有一份副本,修复只落在其中一处。[#3101](https://github.com/modelscope/FunASR/issues/3101) 就是这样产生的:长会话状态无界的缺陷需要在两个仓库分别修复([#3214](https://github.com/modelscope/FunASR/pull/3214) 与 [FunAudioLLM/Fun-ASR#135](https://github.com/FunAudioLLM/Fun-ASR/pull/135))。

---

## 四仓职责

| 仓库 | 核心职责 | 不在这里 |
|---|---|---|
| [modelscope/FunASR](https://github.com/modelscope/FunASR)（工具包 / 运行时） | 框架与推理管线 (pipelines)、训练与微调、VAD / 标点 / ITN / 说话人等组件、**部署服务(含实时 WebSocket 服务)**、`funasr` PyPI 包 | 模型权重与 model card;应用层 UI |
| [FunAudioLLM/Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR)（模型仓） | Fun-ASR-Nano / MLT 模型家族与 LLM-ASR 身份:模型说明、权重发布、能力范围(语言 / 方言 / 口音 / 热词 / 时间戳 / 说话人)、模型评测、微调，以及模型级集成(Transformers、vLLM、GGUF) | 服务实现(链接 FunASR,不再自带权威副本) |
| [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)（模型仓） | SenseVoice 语音理解基础模型:ASR / 语种识别(LID) / 情感识别(SER) / 音频事件检测(AED),及其模型侧用法 | 通用推理框架;部署服务 |
| [modelscope/FunClip](https://github.com/modelscope/FunClip)（应用层） | 基于 FunASR 的视频转写、字幕生成与 LLM 辅助剪辑;本地 Gradio UI | 底层 ASR 能力与模型问题(上游到 FunASR / 模型仓) |

---

## 用户入口

| 我想… | 去哪里 |
|---|---|
| 用 Python 做语音识别 / 训练 / 微调 | [modelscope/FunASR](https://github.com/modelscope/FunASR) |
| 部署实时流式 ASR 服务，推荐 Fun-ASR-Nano + vLLM 做实时识别 | [modelscope/FunASR/fun_asr_nano](https://github.com/modelscope/FunASR/tree/main/examples/industrial_data_pretraining/fun_asr_nano) —— **推荐实现,见下节** |
| 了解 Fun-ASR-Nano / MLT 的能力范围、权重、评测,或使用 Transformers / vLLM / GGUF 集成 | [FunAudioLLM/Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR) |
| 需要情感识别 / 音频事件检测 | [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice) |
| 做视频字幕 / 剪辑 | [modelscope/FunClip](https://github.com/modelscope/FunClip) |

---

## Issue 路由

| 问题类型 | 提到 |
|---|---|
| 框架、推理管线 (pipelines)、训练、微调 | `modelscope/FunASR` |
| 部署服务:实时 WebSocket、离线服务、SDK | `modelscope/FunASR` |
| VAD / 标点 / ITN / 说话人 组件行为 | `modelscope/FunASR` |
| Fun-ASR 系列模型的识别效果、语言支持、权重、评测,以及 Transformers / vLLM / GGUF 等模型级集成 | `FunAudioLLM/Fun-ASR` |
| SenseVoice 的识别 / 情感 / 事件检测效果 | `FunAudioLLM/SenseVoice` |
| 视频剪辑、字幕导出、Gradio UI | `modelscope/FunClip` |

**判断法则:换一个模型后问题是否还在?**

- **还在** → 是框架 / 服务问题 → `modelscope/FunASR`
- **只在某个模型上出现** → 是模型问题 → 对应模型仓

---

## 实时 WebSocket 服务:推荐实现

**`modelscope/FunASR` 中的[Fun-ASR-Nano + vLLM 实时 WebSocket 服务](https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py)是推荐实现。**

- 功能开发、缺陷修复、行为变更,**一律先在 `modelscope/FunASR` 落地**。
- 模型仓(`FunAudioLLM/Fun-ASR`)**只链接到权威实现**,不再把自带副本描述为权威实现。
- 相关 issue 一律提到 `modelscope/FunASR`。

**为什么:** 两份副本各自演进,修复就会只落在其中一处。[#3101](https://github.com/modelscope/FunASR/issues/3101) 已经暴露过这个代价——同一个长会话状态无界的缺陷,需要分别在 [#3214](https://github.com/modelscope/FunASR/pull/3214) 和 [FunAudioLLM/Fun-ASR#135](https://github.com/FunAudioLLM/Fun-ASR/pull/135) 修两次。收敛到唯一权威实现是 Next 的一项工作。

---

## 路线图（方向性）

> 每一项均链接到现有 issue / PR。没有 owner 或验收证据的条目不写完成日期。

### 当前

- **实时服务长会话有界状态** —— 修复已由 [#3214](https://github.com/modelscope/FunASR/pull/3214) 合并，诊断能力已随 `funasr==1.3.19` 发布；模型仓镜像修复 [FunAudioLLM/Fun-ASR#135](https://github.com/FunAudioLLM/Fun-ASR/pull/135) 亦已合并。[#3101](https://github.com/modelscope/FunASR/issues/3101) 仍保持 open，等待 reporter 提供复测日志。
- **Fun-ASR-Nano 的 Transformers 原生集成** —— [huggingface/transformers#46180](https://github.com/huggingface/transformers/pull/46180),正在审查中;当前 CI 与审查状态请以链接的 PR 为准。
- **明确四仓职责与 issue 路由** —— [#3203](https://github.com/modelscope/FunASR/issues/3203);即本文档。

### 下一步

- **收敛重复的实时服务到唯一权威实现**(见上节),避免再次漂移。
- **建立经过 smoke test 的支持矩阵**:Python / CLI / WebSocket / 容器。目标是从顶层 README 能找到一个权威入口、依赖锁定、有固定测试音频与启动 smoke test、CPU/GPU 支持范围写清楚——而不是多个脚本各自声称是推荐入口。
- **稳定的 headless / API 契约**:不依赖 Gradio 或浏览器操作的 CLI / HTTP / gRPC / WebSocket 路径;请求与响应可机器解析;有健康检查、错误码和兼容性测试,适合服务与 agent 调用。
- **容器化(独立方向)**:需要后续单独确定官方镜像、版本标签、CPU/GPU 支持矩阵、健康检查与构建 CI。本文档不提供安装步骤,也不推荐任何特定集群方案;该工作将另开任务,由能够实际验证 CPU/GPU 镜像的人负责。

### 下一阶段

- 在上述接口与兼容性测试稳定之后,再评估需要 breaking changes 的 `2.x`。
- **版本号与发布计划由核心 maintainer 通过 milestone / release plan 确认**,不在本文档中预设。

---

## 相关链接

- English version: [`repository_roles.md`](./repository_roles.md)
- 贡献指南:[`CONTRIBUTING.md`](../CONTRIBUTING.md)
