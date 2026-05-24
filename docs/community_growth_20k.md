# FunASR Community Growth Plan: 20,000+ GitHub Stars

Baseline on 2026-05-24: `modelscope/FunASR` has about 16.2k GitHub stars. The next target is 20k+ stars, which means earning roughly 3.8k additional stars through better onboarding, stronger proof, and repeatable launches.

This plan focuses on useful adoption work rather than vanity marketing: if more users can install, evaluate, deploy, and share FunASR successfully, stars follow naturally.

## North-star metrics

- GitHub stars: 20,000+
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

## Workstream 5: External proof

- Publish reproducible benchmark scripts and raw configuration for the 184-file benchmark.
- Encourage users to share production stories, dashboards, or public demos.
- Add third-party integrations and community projects to README when they are maintained and runnable.
- Keep citations and paper links visible for researchers.

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

目标不是单纯“求 Star”，而是让更多用户能更快完成安装、转写、部署和分享。FunASR 距离 20k stars 还需要约 3.8k 增量，最有效的抓手是：首屏转化、可复现 benchmark、OpenAI 兼容 API/流式/vLLM/Docker 部署闭环、HF/ModelScope/官网同步更新，以及高效的 issue/PR 社区运营。
