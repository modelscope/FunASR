# Repository Roles and Roadmap

This document explains the responsibility boundaries, user entry points, and issue routing across the four FunASR ecosystem repositories, along with a directional roadmap.

> **Versioned release roadmap: pending maintainer confirmation.**
> This document does not commit to future version numbers or release dates. The repository has no established milestones; as of the 2026-07-22 maintainer patrol, GitHub and PyPI both have the `v1.3.26` / `1.3.26` release line published. The boundaries for `1.4 / 1.5 / 2.0` have not been confirmed by core maintainers.

---

## Why this document exists

The four repositories share models and tooling but their responsibility boundaries were never written down, causing two practical problems:

1. **Misrouted issues** — model problems get filed against the toolkit, deployment questions land in model repos, and issues bounce between them.
2. **Duplicate implementation drift** — the same realtime service exists in multiple repositories, and fixes only land in one copy. [#3101](https://github.com/modelscope/FunASR/issues/3101) is a concrete example: an unbounded long-session state bug had to be fixed separately in [#3214](https://github.com/modelscope/FunASR/pull/3214) and [FunAudioLLM/Fun-ASR#135](https://github.com/FunAudioLLM/Fun-ASR/pull/135).

---

## Repository responsibilities

| Repository | Canonical responsibility | Not here |
|---|---|---|
| [modelscope/FunASR](https://github.com/modelscope/FunASR) (toolkit / runtime) | Framework and inference pipelines, training and fine-tuning, components (VAD / punctuation / ITN / speaker), **deployment services (including realtime WebSocket)**, `funasr` PyPI package | Model weights and model cards; application-layer UI |
| [FunAudioLLM/Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR) (model repo) | Fun-ASR-Nano / MLT model family and LLM-ASR identity: model documentation, weight releases, capability scope (languages / dialects / accents / hotwords / timestamps / speaker), benchmarks, fine-tuning, and model-level integrations (Transformers, vLLM, GGUF) | Service implementation (links to FunASR; no longer maintains its own authoritative copy) |
| [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice) (model repo) | SenseVoice speech understanding foundation model: ASR / language identification (LID) / speech emotion recognition (SER) / audio event detection (AED), and model-side usage | General inference framework; deployment services |
| [modelscope/FunClip](https://github.com/modelscope/FunClip) (application layer) | FunASR-based video transcription, subtitle generation, and LLM-assisted clipping; local Gradio UI | Underlying ASR capabilities and model issues (upstream to FunASR / model repos) |

---

## User entry points

| I want to… | Go to |
|---|---|
| Use Python for speech recognition / training / fine-tuning | [modelscope/FunASR](https://github.com/modelscope/FunASR) |
| Deploy a realtime streaming ASR service, recommend **Fun-ASR-Nano + vLLM** | [modelscope/FunASR/fun_asr_nano](https://github.com/modelscope/FunASR/tree/main/examples/industrial_data_pretraining/fun_asr_nano) — **canonical implementation, see below** |
| Understand Fun-ASR-Nano / MLT capabilities, checkpoints, benchmarks, or use Transformers / vLLM / GGUF integrations | [FunAudioLLM/Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR) |
| Use emotion recognition / audio event detection | [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice) |
| Generate video subtitles / clip videos | [modelscope/FunClip](https://github.com/modelscope/FunClip) |

---

## Issue routing

| Problem type | File it in |
|---|---|
| Framework, inference pipeline, training, fine-tuning | `modelscope/FunASR` |
| Deployment services: realtime WebSocket, offline service, SDK | `modelscope/FunASR` |
| VAD / punctuation / ITN / speaker component behavior | `modelscope/FunASR` |
| Fun-ASR model family recognition quality, language support, weights, benchmarks, or model-level integrations (Transformers / vLLM / GGUF) | `FunAudioLLM/Fun-ASR` |
| SenseVoice recognition / emotion / event detection quality | `FunAudioLLM/SenseVoice` |
| Video clipping, subtitle export, Gradio UI | `modelscope/FunClip` |

**Quick test: does the problem persist if you swap in a different model?**

- **Yes** → it is a framework / service issue → `modelscope/FunASR`
- **Only with a specific model** → it is a model issue → the corresponding model repo

---

## Realtime WebSocket service: canonical source

**The realtime WebSocket service in [Fun-ASR-Nano + vLLM realtime WebSocket service](https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py) is the recommended implementation.**

- Feature development, bug fixes, and behavior changes **always land in `modelscope/FunASR` first**.
- Model repos (`FunAudioLLM/Fun-ASR`) **link to the canonical implementation only** and no longer describe their own copy as the authoritative version.
- Related issues should all be filed in `modelscope/FunASR`.

**Why:** two copies evolving independently means fixes land in only one. [#3101](https://github.com/modelscope/FunASR/issues/3101) demonstrated the cost — the same unbounded long-session state bug required separate fixes in [#3214](https://github.com/modelscope/FunASR/pull/3214) and [FunAudioLLM/Fun-ASR#135](https://github.com/FunAudioLLM/Fun-ASR/pull/135). Converging to a single canonical source is a Next roadmap item.

---

## Roadmap (directional)

> Each item links to an existing issue or PR where available. Items without an owner or acceptance evidence do not have completion dates.

### Now

- **Bounded realtime long-session state** — fixes merged via [#3214](https://github.com/modelscope/FunASR/pull/3214), diagnostics shipped in `funasr==1.3.19`, and the model-repo mirror fix [FunAudioLLM/Fun-ASR#135](https://github.com/FunAudioLLM/Fun-ASR/pull/135) also merged. [#3101](https://github.com/modelscope/FunASR/issues/3101) remains open while waiting for reporter retest logs.
- **Fun-ASR-Nano native Transformers integration** — [huggingface/transformers#46180](https://github.com/huggingface/transformers/pull/46180); in review. See the linked PR for current CI and review status.
- **Clarify repository roles and issue routing** — [#3203](https://github.com/modelscope/FunASR/issues/3203); this document.

### Next

- **Converge duplicate realtime services to canonical source** (see above), preventing further drift.
- **Establish a smoke-tested support matrix**: Python / CLI / WebSocket / container. The goal is a single canonical entry point reachable from the top-level README, with pinned dependencies, a fixed test audio file and startup smoke test, and clear CPU/GPU support scope — rather than multiple scripts each claiming to be the recommended entry point.
- **Stable headless / API contract**: CLI / HTTP / gRPC / WebSocket paths that do not depend on Gradio or browser interaction; machine-parseable requests and responses; health checks, error codes, and compatibility tests suitable for services and agent integration.
- **Containerization (separate track)**: requires follow-up work to determine the canonical image, version tags, CPU/GPU support matrix, health checks, and build CI. This document does not provide installation steps or recommend any specific cluster solution; that work will be tracked separately by someone who can verify CPU/GPU images end to end.

### Later

- After the interfaces and compatibility tests above are stable, evaluate `2.x` for any breaking changes needed.
- **Version numbers and release plans are confirmed by core maintainers through milestones / release plans**, not predetermined in this document.

---

## Related

- 中文版: [`repository_roles_zh.md`](./repository_roles_zh.md)
- Contributing guide: [`CONTRIBUTING.md`](../CONTRIBUTING.md)
