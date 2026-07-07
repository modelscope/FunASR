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

As of 2026-07-07 18:47 UTC, the ecosystem has 35,076 combined GitHub stars, or 3,852 additional stars since the 31,224 baseline. The remaining gap to the +20,000 target is 16,148 stars by 2026-09-30, which requires roughly 190 stars/day across the remaining 85 days.

| Repository | Stars | Forks | Open issues | Open PRs | Last push |
|---|---:|---:|---:|---:|---|
| `modelscope/FunASR` | 19,017 | 1,913 | 6 | 0 | 2026-07-07 |
| `FunAudioLLM/Fun-ASR` | 1,360 | 132 | 0 | 0 | 2026-07-07 |
| `FunAudioLLM/SenseVoice` | 8,801 | 788 | 0 | 0 | 2026-07-07 |
| `modelscope/FunClip` | 5,898 | 709 | 0 | 0 | 2026-07-07 |

Keep this snapshot fresh during weekly planning. The ecosystem mode also reports the remaining gap, days left to 2026-09-30, and the required daily average:

```bash
python scripts/collect_growth_metrics.py --ecosystem
python scripts/collect_growth_metrics.py --ecosystem --format json
```

## Traffic funnel snapshot

The highest owned conversion surfaces are the repository overviews and Chinese READMEs. Use these surfaces for model-selection clarity, deployment proof, and official ecosystem routing before adding new long-form pages.

14-day GitHub traffic as of 2026-07-07 18:46 UTC:

| Repository | Views | Unique visitors | Clones | Unique cloners | Highest-traffic owned paths |
|---|---:|---:|---:|---:|---|
| `modelscope/FunASR` | 40,588 | 9,650 | 12,635 | 2,144 | overview: 7,243 uniques; `README_zh.md`: 2,521; releases: 469 |
| `FunAudioLLM/Fun-ASR` | 6,173 | 1,926 | 1,023 | 817 | overview: 1,584 uniques; `README_zh.md`: 581; `docs/vllm_guide.md`: 148 |
| `FunAudioLLM/SenseVoice` | 8,917 | 3,477 | 1,028 | 620 | overview: 2,802 uniques; `README_zh.md`: 923; releases: 163 |
| `modelscope/FunClip` | 2,764 | 1,366 | 622 | 455 | overview: 1,187 uniques; `README_zh.md`: 327; UI screenshots: 203 combined |

Top referrer pattern:

- Search and GitHub dominate all four repos, so SEO snippets, GitHub topics, and README first-screen clarity matter more than low-frequency social posts.
- `funasr.com`, `modelscope.github.io`, Hugging Face, ChatGPT, and `ai-bot.cn` are meaningful secondary referrers; keep model cards and homepage links synchronized with README claims.
- FunClip has unusually visible image traffic, so screenshot freshness and UI clarity can influence creator conversion.

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
| `huggingface/transformers#46180` Fun-ASR-Nano model support | Makes Fun-ASR-Nano usable through the default HF API surface and model docs | Current blocker is an unrelated LightOnOCR shared-cache failure in `tests_processors`; the account cannot rerun Hugging Face Actions jobs, so wait for a maintainer rerun and avoid duplicate comments unless new CI evidence appears. |
| `sgl-project/sglang-omni#898` Fun-ASR serving support | Exposes Fun-ASR-Nano through a high-visibility serving runtime for ASR benchmarks and GPU deployment | Help keep benchmark claims precise: exact checkpoint/revision, dtype, GPU, concurrency, SeedTTS EN full-set comparison against Qwen3-ASR, and a copy-paste smoke command. |
| `ray-project/ray#64053` Ray Serve FunASR ASR example | Puts FunASR in production serving docs for teams already using Ray | Monitor review, answer questions quickly, and keep the example command aligned with the current OpenAI-compatible API behavior. |
| `huggingface/optimum-intel#1801` OpenVINO support | Helps CPU and edge users evaluate Fun-ASR on Intel hardware | Current failed jobs are unrelated OpenVINO matrix failures in Pix2Struct, image-text quantization/export, and tiny-random T5; LauraGPT's direct mirror PR `#1856` was closed after maintainers said they will address the cleanup in a preliminary PR, so wait for that maintainer-side update before adding more comments. |
| `huggingface/speech-to-speech#319` SenseVoice STT handler | Adds SenseVoice/FunASR to local open-source voice-agent pipelines where low-latency STT is a core comparison point | Keep lint/import fixes on the PR head, explain optional `speech-to-speech[sensevoice]` install behavior, and answer handler-scope review quickly. |
| `livekit/agents#6176` FunASR/SenseVoice realtime STT plugin | Opens a path into LiveKit's realtime voice-agent ecosystem where local STT is evaluated alongside hosted providers | CI and CLA are green; avoid duplicate pings and monitor for maintainer review on plugin scope, package metadata, or optional dependency expectations. |
| `punkpeye/awesome-mcp-servers#7153` FunASR MCP server listing | Exposes FunASR to a high-star MCP discovery list used by agent-tool builders looking for local speech tools | Keep the Dockerized stdio server entrypoint ready for Glama checks, then update the external PR only when there is a valid Glama score/badge or maintainer feedback. |
| `run-llama/llama_index#21958` FunASR endpoint reader | Puts FunASR behind a LlamaIndex reader for OpenAI-compatible transcription endpoints used in RAG and agent pipelines | Keep the endpoint contract clear, avoid forcing local `funasr` dependencies into the main package, and validate one request/response example when the author updates. |
| `run-llama/llama_index#21996` local FunASR reader | Gives LlamaIndex users a local SenseVoice/FunASR reader for private transcription workflows | Keep optional dependencies isolated, verify the reader does not affect default installs, and watch for maintainer guidance on package extras. |
| `mem0ai/mem0#5571` optional FunASR transcription helper | Adds local FunASR transcription to a high-star memory layer used by agent builders | Keep FunASR optional and make examples clear about local model downloads; the current failing Vercel preview is a Mem0 team authorization gate, not a FunASR code failure, so wait for maintainer-side preview access or review. |
| `getpaseo/paseo#1634` SenseVoice local STT model support | Adds SenseVoice to a local voice/dictation app with a visible offline STT model catalog and speech CLI path | Current PR is mergeable with focused local validation posted: build server deps, format check, 3 speech Vitest files with 10 tests, and server build all passed; wait for maintainer review without duplicate pings. |
| `infiniflow/ragflow#16473` FunASR / SenseVoice STT provider | Adds FunASR to a high-star RAG workflow product where local STT can become a visible configuration choice | Track duplicate/conflict cleanup, keep provider naming consistent, and verify that skipped CI is expected rather than a hidden regression. |
| `mudler/LocalAI#10090` FunASR backend | Exposes FunASR through a high-star local AI engine used by self-hosters and edge deployments | Keep the backend registry conflict-free after upstream changes, ensure DCO stays green, and maintain one CPU smoke path plus GPU requirements notes. |
| `agno-agi/agno#8501` FunASR transcription tool | Places FunASR in a high-star agent platform as a local multilingual transcription tool | Keep issue linkage and formatting gates green, avoid heavy default dependencies, and respond to review scope requests quickly. |
| `GetStream/Vision-Agents#606` FunASR STT plugin | Adds SenseVoice/FunASR to multimodal voice and vision agent examples | Keep package include paths and tests aligned with upstream conventions, then watch review threads for optional dependency or documentation requests. |
| `TEN-framework/ten-framework#2191` FunASR ASR extension | Places SenseVoice/FunASR in a realtime multimodal agent framework with extension discovery | The failing `claude-review` action is a fork-permission gate, not a code failure; wait for maintainer-side review or rerun, and respond only to concrete code/doc feedback. |
| `activepieces/activepieces#13985` FunASR speech recognition piece | Gives no-code workflow users a direct FunASR speech recognition action | The CLA status is author-controlled and should not be signed by the operator; monitor only for maintainer feedback or test failures until the author-side/legal gate clears. |
| `speaches-ai/speaches#658` FunASR transcription backend | Adds SenseVoice/Paraformer to an OpenAI-compatible local speech API used by self-hosted voice stacks | Current `unstable` state is a fork-workflow approval gate with no check-runs/logs on the head commit; keep the local validation note current and respond only if real CI logs or maintainer review appear. |
| `Uberi/speech_recognition#903` FunASR recognizer | Exposes FunASR through a widely known Python speech-recognition wrapper | Current `unstable` state is a fork-workflow approval gate with no check-runs/logs on the head commit; keep the recognizer optional and lightweight, and be ready with a minimal install/import smoke test if maintainers ask for scope reduction. |
| `fighting41love/funNLP#478` SenseVoice and FunClip speech-section listing | Adds the newer FunASR ecosystem projects to a high-star Chinese NLP resource list with strong domestic discovery value | PR is clean and mergeable; avoid duplicate pings unless maintainers ask for category, wording, or link changes. |
| `josephmisiti/awesome-machine-learning#1339` FunASR Python speech-recognition listing | Places FunASR in a broad high-star machine-learning discovery list where users compare Python ASR toolkits | PR is open but `mergeable_state=blocked`; recheck only for maintainer feedback or a new failed status, and keep the entry description aligned with the core toolkit rather than every model variant. |
| `crownpku/Awesome-Chinese-NLP#32` Speech Recognition and Audio section | Creates a Chinese-NLP discovery path that can route users to FunASR, SenseVoice, Fun-ASR-Nano, and FunClip together | PR is clean and mergeable; monitor without status bumps unless maintainers request section naming or Chinese wording changes. |
| `mahseema/awesome-ai-tools#1403` FunASR tool listing | Complements the FunClip listing in the same AI-tools catalog with the core ASR toolkit | PR is clean and mergeable; keep FunASR and FunClip descriptions distinct so one does not look like a duplicate of the other. |
| `INTERMT/Awesome-PyTorch-Chinese#5` FunASR Chinese PyTorch listing | Adds FunASR to a Chinese PyTorch resource list used by developers looking for local speech models and toolkits | PR is clean and mergeable; monitor for maintainer style feedback and keep the link pointed at the GitHub repo rather than a transient model page. |
| `vinta/awesome-python#3246` SenseVoice listing | Places SenseVoice in the largest Python discovery list as a practical local speech-recognition option | PR is blocked by maintainer/review gate rather than a known code failure; watch for category or project-quality feedback before changing scope. |
| `RVC-Boss/GPT-SoVITS#2801` Fun-ASR-Nano fallback fix | Improves a high-star voice generation workflow by making Fun-ASR-Nano setup errors fall back cleanly instead of blocking users | PR is clean and mergeable; monitor for maintainer questions and keep the fix narrowly scoped to registration/fallback behavior. |
| `jobbole/awesome-python-cn#141` FunASR Chinese Python listing | Adds FunASR to a long-lived Chinese Python resource list used by domestic developers | PR is clean and mergeable; avoid duplicate comments unless maintainers ask for ordering or Chinese wording changes. |
| `steven2358/awesome-generative-ai#821` FunASR speech-to-text listing | Routes generative-AI readers looking for STT tools toward the core FunASR repo | PR is clean and mergeable; monitor for section placement feedback. |
| `owainlewis/awesome-artificial-intelligence#243` FunASR audio listing | Adds another broad AI-discovery path for users comparing open-source audio tools | PR is clean and mergeable; keep the description concise and avoid status pings. |
| `ai4s-research/awesome-ai-for-science#69` FunASR science toolkit listing | Creates a discovery path from scientific AI tooling lists to FunASR for transcription and field-recording workflows | Validate the link, keep the description technically accurate, and avoid extra comments unless maintainers ask for category or wording changes. |
| `lukasmasuch/best-of-ml-python#455` FunASR project listing | Adds FunASR to a high-star ranked Python ML discovery list that is refreshed weekly | The PR is mergeable and already has prior pings; keep monitoring without adding more comments unless new maintainer feedback or validation evidence appears. |
| `mahseema/awesome-ai-tools#1689` FunClip video-tool listing | Adds FunClip to a high-visibility AI tools list where video creators discover clipping and subtitle workflows | Validate the FunClip link, keep the description aligned with local transcription/SRT/AI clipping capabilities, and avoid status pings after evidence is posted. |

Completed external wins:

| Integration | Growth reason | Result |
|---|---|---|
| `xinnan-tech/xiaozhi-esp32-server#3255` configurable FunASR language | Improves a high-star ESP32 voice-agent backend by letting users pin SenseVoice/FunASR language for short utterances | Merged on 2026-06-30 after adding the multi-module management-console database migration requested by maintainers. |
| `tmoroney/auto-subs#629` SenseVoice engine for subtitle workflows | Adds SenseVoice to an on-device subtitle generation app used by video editors and creators | Merged on 2026-06-30 after adding the declarative model manifest, SenseVoice/Canary/Cohere engines, real CTC timestamps, native translation routing, and robustness fixes. |
| `pipecat-ai/pipecat#4844` FunASR local STT service | Puts SenseVoice/FunASR into realtime voice agent pipelines used by builders comparing local STT backends | Merged on 2026-07-02 after adding the local STT service, optional dependency handling, docs, and validation coverage. |

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
