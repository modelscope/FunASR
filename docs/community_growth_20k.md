# FunASR Ecosystem Growth Plan: +20,000 GitHub Stars

Baseline on 2026-06-27: the FunASR ecosystem repositories had 31,224 combined GitHub stars. The next target is earning 20,000 additional stars across the ecosystem by 2026-09-30, reaching 51,224+ combined stars through better onboarding, stronger proof, and repeatable launches.

This plan focuses on useful adoption work rather than vanity marketing: if more users can install, evaluate, deploy, and share FunASR successfully, stars follow naturally.

## Repository scope

| Repository | Role in the ecosystem | Growth surface |
|---|---|---|
| `modelscope/FunASR` | Core toolkit, Python package, runtime services, deployment docs | First-run success, production deployment, issue/PR operations |
| `FunAudioLLM/Fun-ASR` | Fun-ASR-Nano model family and LLM-ASR identity | Model cards, Transformers integration, benchmarks, vLLM/GGUF stories |
| `FunAudioLLM/SenseVoice` | CPU-friendly multilingual ASR with emotion and audio events | Quick demos, edge use cases, app integrations |
| `modelscope/FunClip` | Video clipping and transcription workflow | Creator workflows, local GUI/API recipes, showcase stories |

## North-star metrics

- GitHub stars: +20,000 across the four-repository ecosystem by 2026-09-30
- Monthly PyPI downloads: sustained growth after each release
- README quick-start success: first transcription in under 5 minutes
- Deployment success: API server, WebSocket, Docker, and vLLM examples verified on fresh machines
- Community throughput: faster issue triage, more external PRs, more user showcases

## Audience segments

| Audience | What they need | Repository surface that should serve them |
|---|---|---|
| ASR developers | Fast local transcript, model choice, Python API | README quick start, tutorial, model zoo |
| Production teams | Private deployment, streaming, API compatibility, Docker | runtime docs, OpenAI API example, vLLM guide |
| Agent builders | Speech input to Claude/Cursor/LangChain/Dify/AutoGen | MCP server and OpenAI API examples |
| Researchers | Benchmarks, reproducibility, citations, model details | benchmark report, docs, model cards |
| Chinese users | Chinese docs, ModelScope path, deployment in China | README_zh, docs/zh, ModelScope links |
| International users | English docs, Hugging Face path, comparison with Whisper | README, HF models, benchmark story |

## Current campaign snapshot

As of 2026-06-30, the ecosystem has 34,625 combined GitHub stars, or 3,401 additional stars since the 31,224 baseline. The remaining gap to the +20,000 target is 16,599 stars by 2026-09-30.

Keep this snapshot fresh during weekly planning. The ecosystem mode also reports the remaining gap, days left to 2026-09-30, and the required daily average:

```bash
python scripts/collect_growth_metrics.py --ecosystem
python scripts/collect_growth_metrics.py --ecosystem --format json
```

## Workstream 1: Convert first-time visitors

- Keep the README first screen focused on speed, supported tasks, one-call usage, and links to docs.
- Keep the benchmark table close to the quick start, with exact benchmark scope and full report link.
- Maintain a short model-selection table: SenseVoice, Paraformer, Fun-ASR-Nano, Qwen3-ASR, GLM-ASR, streaming, punctuation, VAD, speaker, emotion.
- Add visible links to deployment, agent integration, contribution, and discussions.
- Avoid burying install and first inference behind long research history.

## Workstream 2: Make deployment shareable

- Verify `funasr-server --device cuda` on a clean GPU machine and publish the exact command/output.
- Add curl examples for `/v1/audio/transcriptions` and WebSocket streaming.
- Keep Docker image tags explicit and document CPU/GPU differences.
- Add a small deployment decision table: Python API vs OpenAI API vs WebSocket vs Docker vs vLLM vs Triton.
- Turn common deployment failures into FAQ entries.

## Workstream 3: Launch content that earns stars

Use one release note or article per clear story:

- "FunASR is 170x realtime and CPU viable for long audio."
- "Self-hosted OpenAI-compatible speech transcription with one command."
- "Streaming ASR with VAD, punctuation, and speaker diarization."
- "MCP speech input for Claude/Cursor and agent workflows."
- "vLLM acceleration for Fun-ASR-Nano."
- "Chinese dialect, accent, meeting, and industrial ASR examples."

For each launch, prepare:

- GitHub release notes with GIF/screenshot or terminal output
- README update and docs link
- Hugging Face and ModelScope model-card update
- Homepage update on `modelscope.github.io/FunASR`
- Short posts for developer communities
- A pinned GitHub discussion for feedback

## Workstream 4: Community operations

- Use issue templates to collect reproducible environment, audio, and deployment details.
- Triage issues weekly with labels: bug, question, documentation, deployment, enhancement, good first issue.
- Convert repeated support answers into docs within 48 hours.
- Keep 5-10 `good first issue` tasks ready for contributors.
- Thank external contributors in release notes.

Daily maintainer loop:

- Check open issues and PRs in `modelscope/FunASR`, `FunAudioLLM/Fun-ASR`, `FunAudioLLM/SenseVoice`, and `modelscope/FunClip`.
- Prioritize blockers that stop a first transcription, an API server launch, a WebSocket deployment, or a model download.
- Keep external ecosystem PRs unblocked, especially integrations in high-visibility AI, agent, workflow, ASR, and video repositories.
- Turn repeated answers into docs or examples before closing the loop.
- Keep actionable issues labelled as `good first issue`, `help wanted`, or `ready for PR` so contributors can help without guessing.

Use the issue snapshot before each maintainer pass so waiting-on-reporter, waiting-on-contributor, and maintainer-owned items stay separate:

```bash
python scripts/collect_growth_metrics.py --issues
python scripts/collect_growth_metrics.py --issues --format json
```

## Workstream 5: External proof

- Publish reproducible benchmark scripts and raw configuration for the 184-file benchmark.
- Encourage users to share production stories, dashboards, public demos, or benchmark notes through the showcase issue template.
- Add third-party integrations and community projects to README when they are maintained and runnable.
- Keep citations and paper links visible for researchers.
- Track and unblock integrations in Hugging Face Transformers, inference runtimes, agent frameworks, workflow tools, video tools, and speech-recognition libraries.

## External integration tracker

High-visibility external integrations can create more qualified traffic than a one-off announcement because they put FunASR in the workflow users already trust. Track them as an operations queue, not as a comment-ping list.

Refresh this table before weekly planning and after any reviewer or CI change:

```bash
python scripts/collect_growth_metrics.py --integrations
python scripts/collect_growth_metrics.py --integrations --format json
```

Set `GITHUB_TOKEN` when running this from CI or a shared network so GitHub's public API rate limit does not hide the real PR state.

| Integration PR | Growth reason | Current maintainer action |
|---|---|---|
| `huggingface/transformers#46180` Fun-ASR-Nano model support | Makes Fun-ASR-Nano usable through the default HF API surface and model docs | Keep CI evidence current, address only actionable review threads, and avoid repeating maintainer pings unless new evidence appears. If `tests_processors` fails only in unrelated LightOnOCR cache setup, ask for a rerun once with the exact failing job. |
| `sgl-project/sglang-omni#898` Fun-ASR serving support | Exposes Fun-ASR-Nano through a high-visibility serving runtime for ASR benchmarks and GPU deployment | Help keep benchmark claims precise: exact checkpoint/revision, dtype, GPU, concurrency, SeedTTS EN full-set comparison against Qwen3-ASR, and a copy-paste smoke command. |
| `ray-project/ray#64053` Ray Serve FunASR ASR example | Puts FunASR in production serving docs for teams already using Ray | Monitor review, answer questions quickly, and keep the example command aligned with the current OpenAI-compatible API behavior. |
| `huggingface/optimum-intel#1801` OpenVINO support | Helps CPU and edge users evaluate Fun-ASR on Intel hardware | Watch for CI or reviewer feedback, then validate a minimal inference path before promoting it in FunASR docs. |
| `infiniflow/ragflow#16473` FunASR / SenseVoice STT provider | Adds FunASR to a high-star RAG workflow product where local STT can become a visible configuration choice | Track duplicate/conflict cleanup, keep provider naming consistent, and verify that skipped CI is expected rather than a hidden regression. |
| `pipecat-ai/pipecat#4844` FunASR local STT service | Puts SenseVoice/FunASR into realtime voice agent pipelines used by builders comparing local STT backends | Watch review, keep docs/readthedocs evidence fresh, and make sure the service degrades clearly when optional FunASR dependencies are missing. |
| `TEN-framework/ten-framework#2191` FunASR ASR extension | Places SenseVoice/FunASR in a realtime multimodal agent framework with extension discovery | Resolve review comments and failing review automation with concrete code/doc fixes rather than status pings. |
| `activepieces/activepieces#13985` FunASR speech recognition piece | Gives no-code workflow users a direct FunASR speech recognition action | The CLA status is author-controlled; monitor only for maintainer feedback or test failures until the author-side gate clears. |
| `Uberi/speech_recognition#903` FunASR recognizer | Exposes FunASR through a widely known Python speech-recognition wrapper | Keep the recognizer optional and lightweight, and be ready with a minimal install/import smoke test if maintainers ask for scope reduction. |

Operating rules:

- Comment on external PRs only when there is new evidence, a reviewer question, or a concrete unblock; avoid low-signal status bumps.
- Keep each integration's local verification command, CI link, and latest reviewer decision in the weekly notes.
- Promote an integration in FunASR README or release notes only after it is merged or has a maintained, runnable branch.
- When a third-party PR stalls, look for a smaller docs/example PR in that upstream instead of forcing a large runtime integration.

## Tracking cadence

Use the lightweight metrics script before and after major README, homepage, release, or demo updates:

```bash
python scripts/collect_growth_metrics.py
python scripts/collect_growth_metrics.py --format json
```

The script captures GitHub stars, forks, watchers, open issues, open pull requests, latest push time, and the current PyPI version using public APIs. Set `GITHUB_TOKEN` when running it from CI or a shared network to avoid public GitHub API rate limits. Paste the Markdown output into launch notes, weekly community updates, or release retrospectives so the 20k-star effort stays measurable.

## 30-day execution checklist

### Week 1: Repository conversion

- [ ] Verify README quick start on clean CPU and GPU environments.
- [ ] Verify all README links and docs links.
- [ ] Add or refresh CONTRIBUTING, PR template, and issue templates.
- [ ] Add a short FAQ for install/deployment failures.
- [ ] Confirm GitHub repo description and topics mention ASR, speech-recognition, streaming, diarization, OpenAI-compatible API, MCP, and vLLM.

### Week 2: Deployment proof

- [ ] Publish API server curl examples.
- [ ] Publish WebSocket streaming examples.
- [ ] Validate Docker CPU and GPU images.
- [ ] Validate vLLM guide on one GPU server.
- [ ] Record concise terminal outputs for release notes.

### Week 3: Launch package

- [ ] Create a GitHub release focused on one sharp value proposition.
- [ ] Update PyPI release description if needed.
- [ ] Update Hugging Face organization/model cards.
- [ ] Update ModelScope model cards and demos.
- [ ] Update homepage hero and docs entry points.

### Week 4: Distribution and feedback

- [ ] Share the release in relevant developer communities.
- [ ] Pin a GitHub discussion for the release.
- [ ] Triage all new issues within 48 hours.
- [ ] Convert top 3 support questions into docs.
- [ ] Track stars, PyPI downloads, issue volume, and docs traffic.

## Release-note template

````markdown
## Why this release matters

FunASR now lets you ...

## Try it in 60 seconds

```bash
pip install -U funasr
funasr-server --device cuda
```

## Highlights

- ...
- ...
- ...

## Verified environments

- GPU:
- CPU:
- Docker:

## Links

- Quick start:
- Deployment guide:
- Benchmark:
- Discussion:
````

## 中文摘要

目标不是单纯“求 Star”，而是让更多用户能更快完成安装、转写、部署和分享。当前增长目标按 FunASR / Fun-ASR / SenseVoice / FunClip 四仓生态统一跟踪：以 2026-06-27 的 31,224 combined stars 为基线，到 2026-09-30 累计新增 20,000 stars。最有效的抓手是：首屏转化、可复现 benchmark、OpenAI 兼容 API/流式/vLLM/Docker 部署闭环、HF/ModelScope/官网同步更新、高星生态集成，以及高效的 issue/PR 社区运营。
