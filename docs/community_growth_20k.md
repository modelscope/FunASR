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

As of 2026-06-30, the ecosystem has 34,622 combined GitHub stars, or 3,398 additional stars since the 31,224 baseline. The remaining gap to the +20,000 target is 16,602 stars by 2026-09-30.

Keep this snapshot fresh during weekly planning:

```bash
gh api repos/modelscope/FunASR --jq .stargazers_count
gh api repos/FunAudioLLM/Fun-ASR --jq .stargazers_count
gh api repos/FunAudioLLM/SenseVoice --jq .stargazers_count
gh api repos/modelscope/FunClip --jq .stargazers_count
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

## Workstream 5: External proof

- Publish reproducible benchmark scripts and raw configuration for the 184-file benchmark.
- Encourage users to share production stories, dashboards, public demos, or benchmark notes through the showcase issue template.
- Add third-party integrations and community projects to README when they are maintained and runnable.
- Keep citations and paper links visible for researchers.
- Track and unblock integrations in Hugging Face Transformers, inference runtimes, agent frameworks, workflow tools, video tools, and speech-recognition libraries.

## Tracking cadence

Use the lightweight metrics script before and after major README, homepage, release, or demo updates:

```bash
python scripts/collect_growth_metrics.py
python scripts/collect_growth_metrics.py --format json
```

The script captures GitHub stars, forks, watchers, open issues, latest push time, and the current PyPI version using public APIs. Set `GITHUB_TOKEN` when running it from CI or a shared network to avoid public GitHub API rate limits. Paste the Markdown output into launch notes, weekly community updates, or release retrospectives so the 20k-star effort stays measurable.

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
