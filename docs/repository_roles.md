# Repository Roles and Roadmap

This document explains the responsibility boundaries, user entry points, and issue routing across the four FunASR ecosystem repositories, along with a directional roadmap.

> **Directional roadmap, not a release promise.**
> This document records shipped capabilities and active work, but does not commit to
> future version numbers or dates. The current Python release is
> [`funasr==1.4.14`](https://github.com/modelscope/FunASR/releases/tag/v1.4.14).
> Any future breaking release still requires a maintainer-approved milestone and
> migration plan.

---

## Why this document exists

The four repositories share models and tooling but their responsibility boundaries were never written down, causing two practical problems:

1. **Misrouted issues** — model problems get filed against the toolkit, deployment questions land in model repos, and issues bounce between them.
2. **Duplicate implementation drift** — the same realtime service exists in multiple repositories, and fixes only land in one copy. [#3101](https://github.com/modelscope/FunASR/issues/3101) is a concrete example: an unbounded long-session state bug had to be fixed separately in [#3214](https://github.com/modelscope/FunASR/pull/3214) and [QwenAudio/Fun-ASR#135](https://github.com/QwenAudio/Fun-ASR/pull/135).

---

## Repository responsibilities

| Repository | Canonical responsibility | Not here |
|---|---|---|
| [modelscope/FunASR](https://github.com/modelscope/FunASR) (toolkit / runtime) | Framework and inference pipelines, training and fine-tuning, components (VAD / punctuation / ITN / speaker), **deployment services (including realtime WebSocket)**, `funasr` PyPI package | Model weights and model cards; application-layer UI |
| [QwenAudio/Fun-ASR](https://github.com/QwenAudio/Fun-ASR) (model repo) | Fun-ASR-Nano / MLT model family and LLM-ASR identity: model documentation, weight releases, capability scope (languages / dialects / accents / hotwords / timestamps / speaker), benchmarks, fine-tuning, and model-level integrations (Transformers, vLLM, GGUF) | Service implementation (links to FunASR; no longer maintains its own authoritative copy) |
| [QwenAudio/SenseVoice](https://github.com/QwenAudio/SenseVoice) (model repo) | SenseVoice speech understanding foundation model: ASR / language identification (LID) / speech emotion recognition (SER) / audio event detection (AED), and model-side usage | General inference framework; deployment services |
| [modelscope/FunClip](https://github.com/modelscope/FunClip) (application layer) | FunASR-based video transcription, subtitle generation, and LLM-assisted clipping; local Gradio UI | Underlying ASR capabilities and model issues (upstream to FunASR / model repos) |

---

## User entry points

| I want to… | Go to |
|---|---|
| Use Python for speech recognition / training / fine-tuning | [modelscope/FunASR](https://github.com/modelscope/FunASR) |
| Deploy a realtime streaming ASR service, recommend **Fun-ASR-Nano + vLLM** | [modelscope/FunASR/fun_asr_nano](https://github.com/modelscope/FunASR/tree/main/examples/industrial_data_pretraining/fun_asr_nano) — **canonical implementation, see below** |
| Transcribe long multi-speaker audio with timestamps and anonymous speaker labels in one model pass | [MOSS-Transcribe-Diarize deployment guide](./moss_transcribe_diarize.md) — an OpenMOSS model integrated with FunASR through local Transformers or vLLM, and independently available through native SGLang Omni; no separate external VAD or speaker model. Labels distinguish speakers within the recording and do not identify a known person. |
| Understand Fun-ASR-Nano / MLT capabilities, checkpoints, benchmarks, or use Transformers / vLLM / GGUF integrations | [QwenAudio/Fun-ASR](https://github.com/QwenAudio/Fun-ASR) |
| Use emotion recognition / audio event detection | [QwenAudio/SenseVoice](https://github.com/QwenAudio/SenseVoice) |
| Generate video subtitles / clip videos | [modelscope/FunClip](https://github.com/modelscope/FunClip) |

---

## Issue routing

| Problem type | File it in |
|---|---|
| Framework, inference pipeline, training, fine-tuning | `modelscope/FunASR` |
| Deployment services: realtime WebSocket, offline service, SDK | `modelscope/FunASR` |
| VAD / punctuation / ITN / speaker component behavior | `modelscope/FunASR` |
| FunASR adapter or deployment behavior for a third-party model such as MOSS-Transcribe-Diarize | `modelscope/FunASR`; model weights and architecture remain with the upstream model owner |
| Fun-ASR model family recognition quality, language support, weights, benchmarks, or model-level integrations (Transformers / vLLM / GGUF) | `QwenAudio/Fun-ASR` |
| SenseVoice recognition / emotion / event detection quality | `QwenAudio/SenseVoice` |
| Video clipping, subtitle export, Gradio UI | `modelscope/FunClip` |

**Quick test: does the problem persist if you swap in a different model?**

- **Yes** → it is a framework / service issue → `modelscope/FunASR`
- **Only with a specific model** → it is a model issue → the corresponding model repo

---

## Realtime WebSocket service: canonical source

**The realtime WebSocket service in [Fun-ASR-Nano + vLLM realtime WebSocket service](https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py) is the recommended implementation.**

- Feature development, bug fixes, and behavior changes **always land in `modelscope/FunASR` first**.
- Model repos (`QwenAudio/Fun-ASR`) **link to the canonical implementation only** and no longer describe their own copy as the authoritative version.
- Related issues should all be filed in `modelscope/FunASR`.

**Why:** two copies evolving independently means fixes land in only one. [#3101](https://github.com/modelscope/FunASR/issues/3101) demonstrated the cost — the same unbounded long-session state bug required separate fixes in [#3214](https://github.com/modelscope/FunASR/pull/3214) and [QwenAudio/Fun-ASR#135](https://github.com/QwenAudio/Fun-ASR/pull/135). Converging to a single canonical source is a Next roadmap item.

---

## Contribute to the roadmap

The roadmap is a queue of testable outcomes, not a list reserved for maintainers. Use the live [help wanted](https://github.com/modelscope/FunASR/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) and [ready for PR](https://github.com/modelscope/FunASR/issues?q=is%3Aissue+is%3Aopen+label%3A%22ready+for+PR%22) queries instead of copying a static task list. Smaller bounded tasks are listed under [good first issue](https://github.com/modelscope/FunASR/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

| Label | What it means |
|---|---|
| `good first issue` | The scope is bounded and maintainers can point to the relevant code or documentation. |
| `help wanted` | The outcome matters, but maintainer hardware, domain knowledge, or implementation capacity is missing. |
| `ready for PR` | The expected behavior and acceptance evidence are clear enough to implement. Comment before starting so work is not duplicated. |
| `needs feedback` | A reporter or hardware owner is validating an outcome. A merged PR or release alone is not a reason to close the issue. |

### Work that needs contributors now

| Area | Current question | Acceptance evidence | Especially useful contribution |
|---|---|---|---|
| [Realtime preview efficiency on L20-class GPUs](https://github.com/modelscope/FunASR/issues/3528) | After matching the number of partial messages, which refresh interval and partial window provide the best latency/throughput trade-off without silently skipping previews? | Client JSONL and `--log-decode-profile` server logs from the exact commit, with SPK, ping, audio, concurrency, partial window, and partial-message count held constant | Reproduction on L20, L4, A10, or other non-H100 GPUs; analysis of queue, encoder, and engine time |
| [AMD Windows Vulkan stability](https://github.com/modelscope/FunASR/issues/3479) | Does the current runtime reach model initialization and transcription on the reporter's AMD GPU, and where is the last successful initialization boundary if it does not? | Exact archive name and SHA256, GPU/driver/Windows versions, full initialization log, and a reporter hardware retest | AMD Windows hardware owners and Vulkan/llama.cpp contributors |
| [Complete public checkpoint functionality](https://github.com/modelscope/FunASR/issues/3496) | How should the missing CTC tensors be published from an authorized model-owner account and validated after upload? | Immutable model revision, file hashes, public clean-cache download, and real timestamp/diarization inference | Model owners with Hugging Face write access and checkpoint validation experience |
| [Upstream model integrations](https://github.com/huggingface/transformers/pull/46180) | Can Fun-ASR-Nano remain compatible with upstream Transformers while preserving pinned model-card and regression-test boundaries? | Exact-head upstream CI, focused local tests, model-card review, and maintainer review | Transformers reviewers and users who can test downstream loading before merge |

### Before claiming an item

1. Read the complete issue timeline and confirm that no contributor is already working on it.
2. Comment with the environment or part you can own and the evidence you plan to produce.
3. Base conclusions on an exact commit, immutable model revision or release asset, and include the command needed to reproduce them.
4. Keep issue and PR closure separate: implementation can merge while reporter validation remains open.

Contributions and issue evidence may be written in Chinese or English. A roadmap or repository-role change should update both this file and [`repository_roles_zh.md`](./repository_roles_zh.md) in the same PR.

---

## Roadmap (directional)

> Each item links to an existing issue or PR where available. Items without an owner or acceptance evidence do not have completion dates.

### Delivered

- **Bounded realtime long-session state** — fixes merged via [#3214](https://github.com/modelscope/FunASR/pull/3214) and [QwenAudio/Fun-ASR#135](https://github.com/QwenAudio/Fun-ASR/pull/135), diagnostics shipped, and reporter evidence allowed [#3101](https://github.com/modelscope/FunASR/issues/3101) to close.
- **Stable application-facing APIs** — the toolkit now ships an OpenAI-compatible transcription server, health checks, browser and command-line smoke tests, and documented Python / CLI / HTTP / WebSocket entry points in the [deployment matrix](./deployment_matrix.md).
- **Industrial and edge deployment paths** — vLLM serving and signed release workflows are documented; the verified ten-platform [`runtime-llamacpp-v0.2.6`](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.6) archives cover Linux, macOS, and Windows CPU/GPU variants, including a dedicated Windows CUDA architecture 120 package for RTX 50 / Blackwell GPUs.
- **Joint transcription and diarization** — the third-party [MOSS-Transcribe-Diarize](./moss_transcribe_diarize.md) model is available through `AutoModel` with local Transformers and vLLM backends, or as an independent native SGLang Omni service. SGLang Omni is not an `AutoModel` backend. The model produces timestamps and speaker labels in one pass without a separate external VAD or speaker model; OpenMOSS remains the model owner.
- **Repository roles and issue routing** — [#3203](https://github.com/modelscope/FunASR/issues/3203) tracks this document and the remaining model-weight and vLLM entry-point questions. It stays open until those questions have evidence and the reporter has time to confirm.

### In progress

- **Fun-ASR-Nano native Transformers integration** — [huggingface/transformers#46180](https://github.com/huggingface/transformers/pull/46180) is in review; use the PR's exact-head CI and review state as the source of truth.
- **Restore complete public checkpoint functionality** — [#3496](https://github.com/modelscope/FunASR/issues/3496) tracks missing CTC tensors needed by timestamp and diarization paths in the Hugging Face checkpoint.
- **Realtime preview efficiency and L20 validation** — [#3528](https://github.com/modelscope/FunASR/issues/3528) established that v1.3.9 appeared faster by silently skipping most partial previews while its event loop was blocked. The issue remains open for equal-work L20 profiling and a deliberate refresh/window policy; it is not treated as a resolved throughput regression.
- **Qwen3-ASR offline vLLM workflow** — [#3592](https://github.com/modelscope/FunASR/pull/3592) adds a tested native `Qwen3ASRModel.LLM` example. [#3419](https://github.com/modelscope/FunASR/issues/3419) remains open until the reporter's 8–9% CER result can be reproduced with an exact model revision, service configuration, and scoring script.
- **AMD Windows Vulkan validation** — [#3479](https://github.com/modelscope/FunASR/issues/3479) remains open for reporter hardware retesting against `runtime-llamacpp-v0.2.6`; publication of the archive is not evidence that the hardware crash is fixed.

### Next

- **Converge duplicate realtime services to the canonical source** (see above), then remove or clearly deprecate mirrors after compatibility evidence.
- **Keep the deployment matrix executable**: every recommended Python / CLI / HTTP / WebSocket / vLLM / llama.cpp path should retain pinned boundaries, a fixed test audio file, a startup smoke test, and an explicit CPU/GPU scope.
- **Container images as a separately verified track**: choose a canonical image and version tags only after CPU and GPU startup, health-check, transcription, and rebuild tests run in CI. This roadmap does not prescribe a cluster platform.

### Later

- Evaluate a breaking `2.x` line only when a concrete interface migration requires it.
- Confirm version numbers and release plans through maintainer-approved milestones rather than predicting them in this document.

---

## Related

- 中文版: [`repository_roles_zh.md`](./repository_roles_zh.md)
- Contributing guide: [`CONTRIBUTING.md`](../CONTRIBUTING.md)
