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

As of 2026-07-08 07:32 UTC, the ecosystem has 35,109 combined GitHub stars, or 3,885 additional stars since the 31,224 baseline. The remaining gap to the +20,000 target is 16,115 stars by 2026-09-30, which requires roughly 192 stars/day across the remaining 84 days.

| Repository | Stars | Forks | Open issues | Open PRs | Last push |
|---|---:|---:|---:|---:|---|
| `modelscope/FunASR` | 19,041 | 1,912 | 2 | 0 | 2026-07-08 |
| `FunAudioLLM/Fun-ASR` | 1,363 | 134 | 0 | 0 | 2026-07-07 |
| `FunAudioLLM/SenseVoice` | 8,807 | 789 | 0 | 0 | 2026-07-07 |
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

### External adoption issue queue

High-star feature requests and roadmap issues are earlier in the funnel than PRs: they shape whether upstream projects expose a clean local STT contract before any code branch exists. Track them separately from merge-ready PRs and comment only when a maintainer asks for implementation detail, test scope, or provider behavior.

| Adoption issue | Growth reason | Current maintainer action |
|---|---|---|
| `openclaw/openclaw#35835` audio file support in the read tool | Extends a high-visibility agent read path from images to audio, with a natural fallback slot for private OpenAI-compatible STT when the active model cannot consume native audio | Keep native audio attachments separate from transcription fallback: audio-capable models should receive original media, while text-only models can use an explicit `POST /v1/audio/transcriptions` adapter returning transcript plus metadata. FunASR/SenseVoice are useful self-hosted regression backends for that fallback contract. |
| `openclaw/openclaw#40078` silent preflight transcription in mention-gated channels | Turns voice-note history capture into a high-visibility STT workflow where private transcription can enrich context without triggering assistant replies | Track the proposed `preflightTranscribe` gate: server-side STT should run only after channel/attachment checks, write transcript provenance into history, keep response generation mention-gated, and use an OpenAI-compatible transcription backend so hosted and local FunASR/SenseVoice services share the same contract. |
| `openclaw/openclaw#45508` self-hosted STT/TTS provider support in WebChat | Routes OpenClaw WebChat users toward private OpenAI-compatible STT providers instead of browser-only speech APIs | A batch push-to-talk path has been scoped: browser records a blob, the Gateway calls the configured `tools.media.audio` / OpenAI-compatible transcription backend, and streaming can remain a follow-up. Watch for product/API decisions before proposing code. |
| `openclaw/openclaw#46661` custom ASR server configuration | Connects a 382k-star OpenClaw native voice-input request to self-hosted/private ASR, giving FunASR, SenseVoice, and Qwen3-ASR a clean OpenAI-compatible STT provider path | LauraGPT posted the provider-contract note at https://github.com/openclaw/openclaw/issues/46661#issuecomment-4909572545. Track maintainer decisions on platform default behavior, explicit credential isolation for custom endpoints, multipart `{base_url}/audio/transcriptions`, minimum `{ "text": "..." }` responses, optional segment/timestamp metadata, timeout messaging, and fallback tests before proposing code. |
| `openclaw/openclaw#87140` pluggable macOS Push-to-Talk STT backend | Opens a native desktop path for FunASR/SenseVoice as local/private ASR behind OpenClaw's existing PTT UX | Keep Apple Speech as the default and watch for a batch-only provider boundary with explicit capability flags (`batch`, `streaming`, `languages`, `local_only`, `requires_network`). |
| `openclaw/openclaw#98027` editable Control UI dictation | Turns a 382k-star agent UI's ordinary prompt composer into a local/private STT adoption path where FunASR or SenseVoice can sit behind the existing Gateway/media-audio provider boundary | PR is open with `check-docs` failure and missing real microphone -> Gateway -> editable draft proof, so do not call it merge-ready. LauraGPT posted the downstream ASR contract checklist at https://github.com/openclaw/openclaw/pull/98027#issuecomment-4911740942; track whether proof covers local/private transcription provider dispatch, no auto-send, redacted errors, and temp-audio cleanup on success/failure. |
| `openclaw/openclaw#102028` QQBot speech transcription timeout | Hardens a very high-visibility voice-message transcription path so stalled hosted or self-hosted STT providers do not block inbound QQBot attachments indefinitely | PR is open and behind main while checks continue; LauraGPT posted a non-blocking self-hosted STT contract note at https://github.com/openclaw/openclaw/pull/102028#issuecomment-4911671891. Track whether the timeout remains configurable through the public provider/audio model path and whether timeout errors stay distinguishable from auth/model/route errors for slow FunASR/SenseVoice endpoints. |
| `NousResearch/hermes-agent#56989` local STT docs and hardening | Helps self-hosted messaging gateways document fully local voice-note transcription for private deployments | Track whether docs define a backend-neutral `local_command` and local HTTP contract; FunASR/SenseVoice should fit as wrappers or OpenAI-compatible `/v1/audio/transcriptions` services. |
| `NousResearch/hermes-agent#16185` Telegram voice-message STT provider precedence | Turns Telegram voice notes in a high-visibility messaging agent into a direct local/private STT adoption path, where FunASR/SenseVoice can fit through either `local_command` wrappers or OpenAI-compatible transcription servers | Fresh user report says removing the OpenAI STT block made local STT work, suggesting provider precedence/config resolution rather than model failure. LauraGPT posted the backend-neutral provider-boundary checklist at https://github.com/NousResearch/hermes-agent/issues/16185#issuecomment-4911698678; track for a Telegram adapter or gateway fix that logs the selected provider and prevents silent OpenAI fallback. |
| `NousResearch/hermes-agent#60624` Windows ffmpeg discovery for Discord voice | Removes a Windows setup blocker in a high-visibility Discord voice transcription/TTS path before local/private STT providers such as FunASR/SenseVoice can work reliably | PR `NousResearch/hermes-agent#60627` adds a shared ffmpeg resolver for `FFMPEG_PATH`, PATH, and winget installs. LauraGPT linked the PR and validation back to the user report on 2026-07-08 at `https://github.com/NousResearch/hermes-agent/issues/60624#issuecomment-4911355349`; now monitor checks and maintainer feedback. |
| `anomalyco/opencode#33300` desktop voice input | Puts local speech input in a high-visibility coding-agent desktop app where privacy-sensitive users may avoid browser dictation | Watch for an app-side transcription provider abstraction that inserts transcripts into the prompt draft and keeps provider credentials out of the renderer. |
| `ggml-org/llama.cpp#24375` ASR model routing for WebUI mic input | Lets Qwen3-ASR / Fun-ASR-Nano act as the dedicated transcription model for a separate text-only chat model | Track the proposed `/v1/models` input-modality discovery and explicit transcription-model selector; avoid duplicate comments while a contributor is working on the UI behavior. |
| `Tencent/ncnn#6790` Qwen3-ASR ncnn port and multi-platform deployment | Opens a 23k-star mobile/edge inference path for Qwen3-ASR-style local transcription, where successful conversion can feed Fun-ASR-Nano/Qwen3-ASR users looking for lightweight on-device ASR | This is a 2026 Rhino-Bird activity issue, so LauraGPT posted a non-claim technical validation note rather than trying to own the task: https://github.com/Tencent/ncnn/issues/6790#issuecomment-4912104458. Watch for a public `Qwen3-ASR-ncnn` repo or PR, then verify PyTorch parity on fixed preprocessing, normalized CJK text, module-level encoder/projector/decoder drift, and Linux/Windows smoke commands. |
| `Wei-Shaw/sub2api#3754` OpenAI audio endpoint passthrough | Adds a Chinese gateway path for OpenAI-compatible STT/TTS clients and self-hosted FunASR/SenseVoice endpoints | Watch for standard multipart `POST /v1/audio/transcriptions` passthrough, language/model forwarding, and a CJK regression that preserves `language=zh`. |
| `elizaOS/eliza#14807` audio PII redaction pipeline | Makes timed ASR spans part of an agent safety workflow where local CJK/private transcription can be useful | Keep FunASR/SenseVoice positioned as optional verifier backends; avoid coupling the primary redaction path to one ASR engine. |
| `modelcontextprotocol/servers#4299` FunASR speech-to-text MCP server | Keeps FunASR visible in the canonical MCP server discussion for local speech tools used by Claude/Cursor-style clients | The compact upstream server now exists in `examples/mcp_server` with `transcribe_audio`, Dockerfile, `glama.json`, and stdio smoke evidence. LauraGPT posted the upstream-status update at https://github.com/modelcontextprotocol/servers/issues/4299#issuecomment-4901476584; next action is to track whether MCP maintainers prefer an external community-server reference instead of vendoring. |
| `crewAIInc/crewAI#5983` FunASR for voice-enabled agents | Routes a 55k-star multi-agent framework toward a provider-neutral voice command path where FunASR/SenseVoice can be a local OpenAI-compatible transcription backend | LauraGPT revived the stale broad request with a concrete `voice.stt.*` config, multipart `/v1/audio/transcriptions` contract, mock endpoint test shape, and agent-command handoff at https://github.com/crewAIInc/crewAI/issues/5983#issuecomment-4901293351. Watch for maintainer direction before opening a code PR. |
| `Significant-Gravitas/AutoGPT#13347` FunASR as an open-source STT backend | Keeps the 185k-star AutoGPT voice-input discussion anchored on a generic OpenAI-compatible transcription contract rather than a heavyweight FunASR-only dependency | LauraGPT first mapped the existing hard-coded Copilot Whisper route, then opened draft PR `Significant-Gravitas/AutoGPT#13500` and linked it back at https://github.com/Significant-Gravitas/AutoGPT/issues/13347#issuecomment-4908286807. Track whether maintainers prefer env-based `/v1/audio/transcriptions` configurability, a workflow block, or a separate self-hosted transcript pipeline. |
| `chatchat-space/Langchain-Chatchat#5479` FunASR/SenseVoice voice input | Puts FunASR in a 38k-star Chinese RAG/Agent app where local audio upload can feed existing Chat and knowledge-base flows | LauraGPT proposed an OpenAI-compatible ASR endpoint slice, including `asr.base_url`, multipart request shape, minimal `{ "text": "..." }` response, and mocked CI endpoint at https://github.com/chatchat-space/Langchain-Chatchat/issues/5479#issuecomment-4901293565. Monitor stale handling and maintainer appetite for a first audio-upload transcription path. |
| `royshil/obs-localvocal#314` SenseVoice/Paraformer engine option | Opens an OBS live-captioning path where SenseVoice/Paraformer can be evaluated through a cleaner ASR-engine boundary instead of a Whisper-only runtime | Maintainer is interested but capacity-limited until mid/late August. LauraGPT mapped current Whisper coupling points and suggested a facade-only first refactor at https://github.com/royshil/obs-localvocal/issues/314#issuecomment-4909662013; wait for a refactor branch or review request before proposing Sherpa-ONNX/FunASR runtime code. |
| `SevaSk/ecoute#203` FunASR alternative ASR backend | Places FunASR/SenseVoice in a local realtime transcription app where users already compare Whisper-compatible and local ASR backends | LauraGPT recommended a two-step path at https://github.com/SevaSk/ecoute/issues/203#issuecomment-4906862462: first expose an OpenAI-compatible local STT endpoint with `stt_base_url`, optional key, model, and language; later add a native backend only with explicit streaming, language, timestamp, and batch-window capabilities. Monitor for maintainer appetite or an implementation PR before posting again. |
| `521xueweihan/HelloGitHub#3296` FunASR Chinese open-source recommendation | Gives FunASR a high-visibility Chinese developer discovery path in a long-running open-source recommendation channel | LauraGPT posted a 2026-07-08 editor-ready refresh at `https://github.com/521xueweihan/HelloGitHub/issues/3296#issuecomment-4910800381` with current ecosystem star counts, category fit, quick-start shape, OpenAI-compatible endpoint value, and the FunClip related-project angle; wait for editor triage without duplicate bumps. |
| `duixcom/Duix-Avatar#605` FunASR/SenseVoice ASR backend option | Puts FunASR/SenseVoice into a 13k-star digital-human project where low-latency ASR plus emotion/audio-event tags can drive avatar interaction | Maintainers acknowledged the current-contract suggestion and said they noted it for subsequent version iterations. Track for a Duix-side version/update invite, then verify the existing `duix-avatar-asr` `/v1/preprocess_and_tran` contract before pushing any parallel OpenAI-compatible endpoint path. |
| `sgl-project/sglang-omni#924` ASR support roadmap | Coordinates Fun-ASR-Nano serving tasks across cache behavior, batching, evaluation, and benchmark parity in a serving runtime used by ASR/GPU deployment contributors | LauraGPT posted FunASR-side guidance on full-context encoder cache scope, per-sample speech embedding lengths, batching checks, and CJK evaluation coverage; monitor F-PR follow-ups and avoid duplicate roadmap comments. |
| `pollinations/pollinations#5321` next-model voting for STT | Places FunASR/SenseVoice/Fun-ASR-Nano in a high-traffic open inference model request funnel used by app builders who want hosted or OpenAI-compatible transcription models | LauraGPT posted one exact model request on 2026-07-07 under the TTS & STT voting thread; avoid duplicate comments and monitor for maintainer model-selection signals or requests for endpoint/spec details. |
| `cjpais/Handy#1626` Qwen3-ASR Intel Mac AMD Metal fallback | Reduces a visible local-dictation failure mode in a 25k-star desktop app, keeping Qwen3-ASR / SenseVoice users from abandoning the model after a backend-specific crash | Maintainer confirmed `transcribe.cpp` will avoid Metal by default on non-Apple-Silicon Macs in the next release. Watch the release, then verify Intel macOS + AMD Metal + Qwen3-ASR routes to CPU fallback while SenseVoice keeps its working path. |
| `Zackriya-Solutions/meetily#567` Apple Voice Memos drag-to-import | Routes a 20k-star privacy-first meeting assistant toward imported-audio transcription workflows where local STT providers can be evaluated | Treat this as macOS import normalization, not an ASR swap: inspect Voice Memos drag payloads, normalize them to a local `.m4a` path plus title/start-time/duration metadata, reuse the existing import/transcription provider pipeline, and keep FunASR/SenseVoice as later optional providers behind that boundary. GitHub returned `403 Blocked` on the first public comment attempt, so monitor without retrying until there is new maintainer activity or account policy changes. |
| `altic-dev/FluidVoice#547` remote transcription backend | Gives lightweight or Intel Macs a path to consume a LAN/GPU STT server instead of running every model locally | Relevant to FunASR/SenseVoice remote endpoints, but GitHub returned `403 Blocked` on both the first comment attempt and a 2026-07-08 retry after new maintainer/user activity. Maintainers currently framed near-term work as easier local server startup plus future Windows/Linux GPU support, not a remote consumer endpoint; do not retry again unless account policy or repository write policy changes. |
| `freestyle-voice/freestyle#415` custom OpenAI-compatible STT base URL | Routes a voice-dictation app's batch STT provider toward LAN/self-hosted FunASR/SenseVoice endpoints without forcing users through hosted OpenAI URLs | Author accepted LauraGPT's compatibility scope at https://github.com/freestyle-voice/freestyle/issues/415#issuecomment-4911863210: strip trailing `/v1` or slash before appending `/v1`, keep the first pass batch-only on the existing `openai` provider, reuse the current API key/model flow, and test both bare-host and `/v1` base URLs. Wait for the implementation PR before adding more comments. |

| Integration PR | Growth reason | Current maintainer action |
|---|---|---|
| `1c7/chinese-independent-developer#1065` FunClip listing link refresh | Keeps FunClip's existing entry in a 49k-star Chinese independent-developer directory pointed at the current official GitHub repository instead of the old organization homepage | PR updates the accepted FunClip listing's "更多介绍" link from `github.com/alibaba-damo-academy` to `github.com/modelscope/FunClip`; monitor for maintainer acceptance without extra pings. |
| `huggingface/transformers#46180` Fun-ASR-Nano model support | Makes Fun-ASR-Nano usable through the default HF API surface and model docs | Current blocker is an unrelated LightOnOCR shared-cache failure in `tests_processors`; the account cannot rerun Hugging Face Actions jobs, so wait for a maintainer rerun and avoid duplicate comments unless new CI evidence appears. |
| `huggingface/transformers#47111` Qwen3-ASR hotword and language parsing fixes | Keeps the high-visibility Transformers Qwen3-ASR implementation aligned with upstream hotword/context behavior, language hints, and processor training paths used by downstream Fun-ASR-Nano/Qwen3-ASR adopters | The Qwen3-ASR slow job was explicitly rerun and reported no PR-specific failures on head `caf37300fc3d3333c3d7c2162a92a886e424eb1c`; the aggregate CI recap still has unrelated suite failures. Monitor for reviewer requests around hotword serialization, language parsing, and qwen3_asr processor tests before adding any FunASR-side comment. |
| `sgl-project/sglang-omni#898` Fun-ASR serving support | Exposes Fun-ASR-Nano through a high-visibility serving runtime for ASR benchmarks and GPU deployment | The contributor-owned branch is currently dirty; LauraGPT already posted a concrete ASR benchmark rename/docs conflict recipe, and the author replied that conflicts will be resolved during merging. Wait for contributor or maintainer conflict resolution before adding more comments. |
| `vllm-project/vllm#42478` Qwen3-ASR streaming postprocessing | Improves the upstream Qwen3-ASR streaming path used by OpenAI-compatible transcription clients, so downstream apps can consume clean SSE transcript deltas without model-specific `language ...<asr_text>` cleanup | LauraGPT posted downstream validation guidance on 2026-07-08. CI is currently blocked only by vLLM's pre-run gate: the PR needs a maintainer `ready`/`verified` label or an author history of 4+ merged PRs (current log found 2), so wait for maintainer review rather than adding code-change comments; still watch for a no-space CJK regression. |
| `vllm-project/vllm#47729` MOSS-Transcribe-Diarize support | Expands vLLM's OpenAI-compatible `/v1/audio/transcriptions` route for long-form ASR with timestamps and speaker labels, creating another high-visibility comparison point for Qwen3-ASR / Fun-ASR-Nano-style serving behavior | PR is open and `mergeable=true` with `mergeable_state=blocked`; LauraGPT posted a non-blocking downstream ASR contract note at https://github.com/vllm-project/vllm/pull/47729#issuecomment-4912065897. Watch that `response_format=json` keeps the plain `{ "text": "..." }` shape stable, any future `segments` remain additive, CJK long-form audio stays covered, and the new model registration does not alter existing Qwen3-ASR request/response behavior. |
| `tenstorrent/tt-metal#49104` Qwen3-ASR Blackhole/P150 bringup | Opens a hardware-accelerated Qwen3-ASR path with an OpenAI-compatible `/v1/audio/transcriptions` server for Tenstorrent users evaluating speech workloads | PR is open, approved, and process-blocked rather than waiting on FunASR feedback; LauraGPT posted a non-blocking API-contract note at https://github.com/tenstorrent/tt-metal/pull/49104#issuecomment-4911526665. Track merge/release docs so downstream FunASR users can compare TT, CPU, vLLM, and H100 timing scopes correctly. |
| `ray-project/ray#64053` Ray Serve FunASR ASR example | Puts FunASR in production serving docs for teams already using Ray | Monitor review, answer questions quickly, and keep the example command aligned with the current OpenAI-compatible API behavior. |
| `huggingface/optimum-intel#1801` OpenVINO support | Helps CPU and edge users evaluate Fun-ASR on Intel hardware | Current failed jobs are unrelated OpenVINO matrix failures in Pix2Struct, image-text quantization/export, and tiny-random T5; LauraGPT's direct mirror PR `#1856` was closed after maintainers said they will address the cleanup in a preliminary PR, so wait for that maintainer-side update before adding more comments. |
| `huggingface/speech-to-speech#319` SenseVoice STT handler | Adds SenseVoice/FunASR to local open-source voice-agent pipelines where low-latency STT is a core comparison point | Keep lint/import fixes on the PR head, explain optional `speech-to-speech[sensevoice]` install behavior, and answer handler-scope review quickly. |
| `OpenBMB/VoxCPM#349` Windows CUDA installer with SenseVoice fallback | Puts SenseVoice into a 32k-star multilingual TTS app's Windows install and first-run ASR path as the fallback when local Parakeet/CUDA is unavailable | Current head `9f34141cc81e04028c4e62ac652f2a66dd453dfa` is dirty only in `app.py`; LauraGPT posted the conflict recipe and light validation at https://github.com/OpenBMB/VoxCPM/pull/349#issuecomment-4905293176. Wait for author rebase or maintainer action without duplicate comments. |
| `livekit/agents#6176` FunASR/SenseVoice realtime STT plugin | Opens a path into LiveKit's realtime voice-agent ecosystem where local STT is evaluated alongside hosted providers | CI and CLA are green; avoid duplicate pings and monitor for maintainer review on plugin scope, package metadata, or optional dependency expectations. |
| `datajuicer/data-juicer#938` HumanVBench audio/video operators | Places FunASR/SenseVoice-style speech understanding into a 6k-star data processing toolkit used to evaluate human-centric video and multimodal datasets | Current PR is open and mergeable but process-blocked while unit-test jobs are still waiting; LauraGPT posted the FunASR/SenseVoice dependency and validation follow-up at https://github.com/datajuicer/data-juicer/pull/938#issuecomment-4905235851. Wait for CI or maintainer feedback before adding more comments. |
| `mahimairaja/voiceai#16` FunASR and SenseVoice STT resource listing | Places FunASR/SenseVoice in a curated voice-AI resource map for builders comparing STT providers and local speech stacks | Maintainer approved. A pre-existing Stars badge target caused `link-check` to fail; pushed `16c451d` to retarget the badge to the repo homepage, verified `lychee 0.23.0` with the CI args (`480 Total`, `0 Errors`), and posted the green-status note after GitHub reported combined status `success`. Wait for maintainer merge without further pings. |
| `punkpeye/awesome-mcp-servers#7153` FunASR MCP server listing | Exposes FunASR to a high-star MCP discovery list used by agent-tool builders looking for local speech tools | Glama manifest now points at the current `server.json` schema and declares the maintainer for verification; the MCP Dockerfile also carries the official Registry OCI ownership label. Next step is maintainer OAuth/Glama ingestion or a public OCI image plus matching MCP Registry `server.json`, then update the external PR only after a valid score badge or maintainer feedback. |
| `zts212653/clowder-ai#1083` Qwen3-ASR service unification | Puts Qwen3-ASR into a user-facing local STT service slot by making it a `whisper-stt` backend variant instead of a separate, easier-to-misconfigure service | Head `28067864e313cd03dd3d6f4ce0a72ec5b5026b47` reports green CI after fixes for Rosetta hardware detection, Qwen3-ASR install/server dispatch, async backend locking, temp WAV cleanup, and stale setup docs; author status update: https://github.com/zts212653/clowder-ai/pull/1083#issuecomment-4910797452. Wait for maintainer re-review without duplicate comments. |
| `run-llama/llama_index#21958` FunASR endpoint reader | Puts FunASR behind a LlamaIndex reader for OpenAI-compatible transcription endpoints used in RAG and agent pipelines | LauraGPT rechecked head `07a8599deaebe5e7a559d62174e7a872870c2f7e` at https://github.com/run-llama/llama_index/pull/21958#issuecomment-4905345545; author acknowledged the shared-reader pytest caveat at https://github.com/run-llama/llama_index/pull/21958#issuecomment-4905384536. Keep the endpoint contract clear and avoid forcing local `funasr` dependencies into the main package. |
| `run-llama/llama_index#21996` local FunASR reader | Gives LlamaIndex users a local SenseVoice/FunASR reader for private transcription workflows | Keep optional dependencies isolated, verify the reader does not affect default installs, and watch for maintainer guidance on package extras. |
| `mem0ai/mem0#5571` optional FunASR transcription helper | Adds local FunASR transcription to a high-star memory layer used by agent builders | Keep FunASR optional and make examples clear about local model downloads; the current failing Vercel preview is a Mem0 team authorization gate, not a FunASR code failure, so wait for maintainer-side preview access or review. |
| `Significant-Gravitas/AutoGPT#13500` configurable transcription endpoints | Opens a path from AutoGPT's high-visibility agent platform to local FunASR/SenseVoice gateways through the existing OpenAI-compatible transcription route | GitHub Actions are green after the typed `HeadersInit` fix; remaining blockers are the author-controlled CLA, AutoGPT-team Vercel preview authorization, and maintainer review, not FunASR-side code. |
| `harry0703/MoneyPrinterTurbo#1006` shareable video-generation presets | Puts subtitle, language, voice, and provider settings in a 96k-star short-video generator's reusable preset schema, creating the right safety boundary for later local/offline subtitle ASR or FunClip handoff workflows | LauraGPT posted a non-blocking preset-contract note at https://github.com/harry0703/MoneyPrinterTurbo/pull/1006#issuecomment-4912521164: export only stable non-secret choices such as subtitle provider, language, subtitle style, TTS/voice selection, and local/offline capability; keep API keys, bearer tokens, signed URLs, temporary media paths, cached model files, and unsanitized endpoint URLs out of shareable presets. Track review without asking this preset PR to add a FunASR/SenseVoice engine. |
| `harry0703/MoneyPrinterTurbo#911` OpenVINO local subtitle provider | Adds a hardware-accelerated local transcription provider to a 96k-star short-video generator's subtitle pipeline, creating the same `audio -> SRT/timestamped segments` seam that later FunASR/SenseVoice or FunClip workflows can reuse | LauraGPT posted a non-blocking provider-boundary note at https://github.com/harry0703/MoneyPrinterTurbo/pull/911#issuecomment-4912751845. Track whether the PR keeps `subtitle_provider` generic, handles missing OpenVINO packages/model directories before subtitle correction, preserves valid UTF-8 SRT and CJK punctuation, and avoids another settings migration for future `funasr` / `sensevoice` providers. |
| `maximhq/bifrost#5020` diarized OpenAI transcription compatibility | Keeps a high-visibility LLM gateway's OpenAI-compatible `/v1/audio/transcriptions` path safe for diarized STT responses, including downstream clients that compare hosted models with self-hosted FunASR/SenseVoice endpoints | PR is open with green checks and bot review confidence; LauraGPT posted a non-blocking downstream contract note at https://github.com/maximhq/bifrost/pull/5020#issuecomment-4911586742. The fix covers string diarized segment IDs, explicit empty `segments: []`, multipart transcription params, and speaker passthrough; watch merge/release feedback before adding more comments. |
| `getpaseo/paseo#313` FunASR streaming STT provider | Adds a standalone FunASR/SenseVoice server and streaming dictation provider to a 10k-star agent orchestration app | Current head `19d9270f1b55d5a0d59288ff2c4a4e707bd60c9a` is dirty against main; LauraGPT triage at https://github.com/getpaseo/paseo/pull/313#issuecomment-4905606566 documents the conflict set and a `needsFunASR` provider-wiring bug to fix during rebase. Wait for author rebase or maintainer guidance without duplicate pings. |
| `getpaseo/paseo#1634` SenseVoice local STT model support | Adds SenseVoice to a local voice/dictation app with a visible offline STT model catalog and speech CLI path | Current PR is mergeable at head `4cc65ac6fbc9735ac487648bbbd02ee487050f31`; posted focused local validation at https://github.com/getpaseo/paseo/pull/1634#issuecomment-4905550193 covering server deps build, format check, 3 speech Vitest files with 10 tests, and server build. Wait for maintainer review without duplicate pings. |
| `infiniflow/ragflow#16473` FunASR / SenseVoice STT provider | Adds FunASR to a high-star RAG workflow product where local STT can become a visible configuration choice | Track duplicate/conflict cleanup, keep provider naming consistent, and verify that skipped CI is expected rather than a hidden regression. |
| `mudler/LocalAI#10090` FunASR backend | Exposes FunASR through a high-star local AI engine used by self-hosters and edge deployments | Upstream DCO is green; the fork-side `tests` failure is a CodeQL/SARIF upload permission gate on a fork, not a FunASR backend regression. Local CPU smoke validation remains `backend/python/funasr/test.py` with four passing tests, so wait for maintainer-side review or rerun without duplicate pings. |
| `agno-agi/agno#8501` FunASR transcription tool | Places FunASR in a high-star agent platform as a local multilingual transcription tool | Keep issue linkage and formatting gates green, avoid heavy default dependencies, and respond to review scope requests quickly. |
| `GetStream/Vision-Agents#606` FunASR STT plugin | Adds SenseVoice/FunASR to multimodal voice and vision agent examples | Keep package include paths and tests aligned with upstream conventions, then watch review threads for optional dependency or documentation requests. |
| `TEN-framework/ten-framework#2191` FunASR ASR extension | Places SenseVoice/FunASR in a realtime multimodal agent framework with extension discovery | The failing `claude-review` action is a fork-permission gate, not a code failure; wait for maintainer-side review or rerun, and respond only to concrete code/doc feedback. |
| `activepieces/activepieces#13985` FunASR speech recognition piece | Gives no-code workflow users a direct FunASR speech recognition action | Replacement PR for conflicted `#13450` is open and mergeable at head `f9d22ee03c58199c7236e2f1008f083d6f80b4a2`; Greptile marks the additive community piece safe to merge and normal checks are green/skipped. Latest LauraGPT recheck: https://github.com/activepieces/activepieces/pull/13985#issuecomment-4904635178. Remaining blocker is the `license/cla` status, which is author/account-controlled and should not be signed by the operator. |
| `omnigent-ai/omnigent#2093` server-side streaming dictation | Opens a model-agnostic dictation backend in a multi-agent desktop/mobile harness where FunASR/SenseVoice can later fit behind the same local worker or OpenAI-compatible fallback contract | LauraGPT posted the non-blocking FunASR/SenseVoice backend-contract follow-up at https://github.com/omnigent-ai/omnigent/pull/2093#issuecomment-4907140875. Current PR is open and mergeable but blocked by `npm test`, pre-commit, and maintainer approval checks; wait for concrete review or CI logs before more comments. |
| `speaches-ai/speaches#658` FunASR transcription backend | Adds SenseVoice/Paraformer to an OpenAI-compatible local speech API used by self-hosted voice stacks | Current `unstable` state is a fork-workflow approval gate with no check-runs/logs on the head commit; keep the local validation note current and respond only if real CI logs or maintainer review appear. |
| `Uberi/speech_recognition#903` FunASR recognizer | Exposes FunASR through a widely known Python speech-recognition wrapper | Current `unstable` state is a fork-workflow approval gate with no check-runs/logs on the head commit; keep the recognizer optional and lightweight, and be ready with a minimal install/import smoke test if maintainers ask for scope reduction. |
| `fighting41love/funNLP#478` SenseVoice and FunClip speech-section listing | Adds the newer FunASR ecosystem projects to a high-star Chinese NLP resource list with strong domestic discovery value | PR is clean and mergeable; avoid duplicate pings unless maintainers ask for category, wording, or link changes. |
| `josephmisiti/awesome-machine-learning#1339` FunASR Python speech-recognition listing | Places FunASR in a broad high-star machine-learning discovery list where users compare Python ASR toolkits | PR is open but `mergeable_state=blocked`; recheck only for maintainer feedback or a new failed status, and keep the entry description aligned with the core toolkit rather than every model variant. |
| `crownpku/Awesome-Chinese-NLP#32` Speech Recognition and Audio section | Creates a Chinese-NLP discovery path that can route users to FunASR, SenseVoice, Fun-ASR-Nano, and FunClip together | PR is clean and mergeable; monitor without status bumps unless maintainers request section naming or Chinese wording changes. |
| `mahseema/awesome-ai-tools#1403` FunASR tool listing | Complements the FunClip listing in the same AI-tools catalog with the core ASR toolkit | PR is clean and mergeable; keep FunASR and FunClip descriptions distinct so one does not look like a duplicate of the other. |
| `INTERMT/Awesome-PyTorch-Chinese#5` FunASR Chinese PyTorch listing | Adds FunASR to a Chinese PyTorch resource list used by developers looking for local speech models and toolkits | PR is clean and mergeable; monitor for maintainer style feedback and keep the link pointed at the GitHub repo rather than a transient model page. |
| `krzjoa/awesome-python-data-science#99` FunASR computer-audition listing | Adds FunASR to a Python data-science discovery list where speech/audio tooling is compared with broader ML stacks | PR is clean and mergeable; maintain a short toolkit-focused description and avoid duplicate comments unless maintainers ask for category changes. |
| `zzw922cn/awesome-speech-recognition-speech-synthesis-papers#27` FunAudioLLM paper listing | Places Paraformer, FunASR, SenseVoice, and Fun-ASR-Nano papers in a speech-recognition/synthesis paper list used by researchers | PR is clean and mergeable; keep paper links stable and be ready to adjust ordering or paper metadata if requested. |
| `Osmantic/ODS#1639` SenseVoice/FunASR STT backend | Adds a direct local STT backend to an app surface where users can choose open speech engines | PR is mergeable but blocked by review/process state; wait for maintainer feedback and keep any follow-up focused on optional dependency and endpoint behavior. |
| `faroit/awesome-python-scientific-audio#85` FunASR scientific-audio listing | Exposes FunASR to Python users browsing scientific audio and speech-processing toolkits | PR is clean and mergeable; monitor for taxonomy or wording feedback without status bumps. |
| `joewongjc/type4me#207` Qwen3-only local ASR mode | Helps a local dictation app use Qwen3-ASR final transcription without keeping SenseVoice/VAD preview models resident | Keep as draft until a full Xcode/XCTest run or real Qwen3-only ASR smoke test validates the runtime path; do not request review yet. |
| `ga642381/speech-trident#31` SenseVoice model-list entry | Adds SenseVoice to a speech/audio language-model catalog that routes model researchers toward FunASR ecosystem components | PR is clean and mergeable; monitor for citation or category feedback. |
| `vinta/awesome-python#3246` SenseVoice listing | Places SenseVoice in the largest Python discovery list as a practical local speech-recognition option | PR is blocked by maintainer/review gate rather than a known code failure; watch for category or project-quality feedback before changing scope. |
| `RVC-Boss/GPT-SoVITS#2801` Fun-ASR-Nano fallback fix | Improves a high-star voice generation workflow by making Fun-ASR-Nano setup errors fall back cleanly instead of blocking users | PR is clean and mergeable; monitor for maintainer questions and keep the fix narrowly scoped to registration/fallback behavior. |
| `jobbole/awesome-python-cn#141` FunASR Chinese Python listing | Adds FunASR to a long-lived Chinese Python resource list used by domestic developers | PR is clean and mergeable; avoid duplicate comments unless maintainers ask for ordering or Chinese wording changes. |
| `yuekaizhang/Fun-ASR-vllm#21` vLLM prompt embedding dtype fix | Keeps a community Fun-ASR-vLLM GPU inference path stable for users trying Fun-ASR-Nano with vLLM | PR is clean and has a validation note. Monitor for maintainer feedback, keep the change limited to float32 prompt embeddings, and avoid extra status bumps unless runtime questions appear. |
| `openvino-agent/optimum-intel#5` FunASR OpenVINO review cleanup | Preserves an OpenVINO review-structure cleanup branch while the upstream optimum-intel path is maintainer-owned | Keep as a reference branch for review cleanup and do not ping externally unless maintainers ask for the generated-file or modularization details. |
| `EmulationAI/awesome-large-audio-models#19` FunAudioLLM paper listing | Adds Paraformer, FunASR, SenseVoice, and Fun-ASR-Nano papers to a large-audio-models discovery list | README-only PR with validation posted; wait for maintainer review and be ready to adjust paper ordering or citation text. |
| `ddlBoJack/Awesome-Speech-Language-Model#6` Fun-ASR-Nano listing | Routes speech language model readers toward Fun-ASR-Nano as an audio-LLM component | Validation was posted on 2026-07-07; PR is clean with no exposed checks, so monitor without duplicate pings. |
| `LqNoob/Neural-Codec-and-Speech-Language-Models#4` Fun-ASR-Nano and SenseVoice listing | Adds the FunAudioLLM speech understanding models to a neural codec and speech-language-model resource list | Validation was posted on 2026-07-07; wait for maintainer review and keep any follow-up focused on model categorization. |
| `PyTorchKR/oss-landscape#688` FunASR OSS landscape entry | Adds FunASR to a Korean PyTorch and OSS discovery map that can route regional users to the toolkit | PR is README-only, clean, and has no check contexts; avoid low-signal comments unless maintainers ask for metadata or category changes. |
| `metame-ai/awesome-audio-plaza#10` FunASR, SenseVoice, and FunClip ASR listing | Presents the full FunASR ecosystem together in an audio tools plaza used by speech and creator-tool readers | Prior pings are already present; monitor for maintainer feedback and keep future replies limited to link or category changes. |
| `ChristosChristofidis/awesome-deep-learning#317` FunASR framework listing | Adds FunASR to a 28k-star deep-learning discovery list where users compare frameworks and toolkits | Validation was refreshed on 2026-07-07; wait for maintainer review without another status bump. |
| `Hannibal046/Awesome-LLM#623` Fun-ASR-Nano Alibaba-section listing | Routes a large LLM audience to Fun-ASR-Nano as a speech understanding model in the Alibaba ecosystem | Validation was refreshed on 2026-07-07; wait for maintainer review and adjust section placement only if requested. |
| `AiHubCN/Awesome-Chinese-LLM#103` SenseVoice and Fun-ASR-Nano Chinese LLM listing | Adds FunAudioLLM speech models to a high-star Chinese LLM catalog with strong domestic discovery value | Validation was refreshed on 2026-07-07; PR is currently blocked by maintainer review/process state rather than a code failure. |
| `pluja/awesome-privacy#836` FunASR privacy-focused STT listing | Puts fully self-hosted FunASR in front of privacy-conscious users looking for local speech-to-text models | Validation was refreshed on 2026-07-07; wait for maintainer review and keep privacy wording concise. |
| `BradyFU/Awesome-Multimodal-Large-Language-Models#280` Fun-ASR-Nano MLLM listing | Adds Fun-ASR-Nano to a multimodal-LLM discovery list where speech understanding models are compared | Validation was refreshed on 2026-07-07; monitor for maintainer category feedback. |
| `mahmoud/awesome-python-applications#227` FunASR audio application listing | Adds FunASR to a broad Python applications list used by developers looking for usable audio tooling | Validation was refreshed on 2026-07-07; wait for maintainer review without extra pings. |
| `bharathgs/Awesome-pytorch-list#164` FunASR PyTorch speech listing | Adds FunASR to a PyTorch NLP and speech processing list with a broad ML practitioner audience | Validation was refreshed on 2026-07-07; PR is clean and should only need maintainer review. |
| `WangRongsheng/awesome-LLM-resources#162` FunASR and SenseVoice resources | Adds FunASR/SenseVoice to a Chinese LLM resources list where multimodal and speech tool readers compare projects | Validation was refreshed on 2026-07-07; wait for maintainer review and keep future replies focused on link stability. |
| `steven2358/awesome-generative-ai#821` FunASR speech-to-text listing | Routes generative-AI readers looking for STT tools toward the core FunASR repo | PR is clean and mergeable; monitor for section placement feedback. |
| `owainlewis/awesome-artificial-intelligence#243` FunASR audio listing | Adds another broad AI-discovery path for users comparing open-source audio tools | PR is clean and mergeable; keep the description concise and avoid status pings. |
| `ai4s-research/awesome-ai-for-science#69` FunASR science toolkit listing | Creates a discovery path from scientific AI tooling lists to FunASR for transcription and field-recording workflows | Validate the link, keep the description technically accurate, and avoid extra comments unless maintainers ask for category or wording changes. |
| `lukasmasuch/best-of-ml-python#455` FunASR project listing | Adds FunASR to a high-star ranked Python ML discovery list that is refreshed weekly | The PR is mergeable and already has prior pings; keep monitoring without adding more comments unless new maintainer feedback or validation evidence appears. |
| `tensorchord/Awesome-LLMOps#533` FunASR audio foundation model listing | Adds FunASR to an LLMOps discovery list where users look for model-serving and operations-ready audio foundation models | Current PR is README-only, clean, DCO-passing, and has fresh FunASR-side link validation; wait for maintainer review without duplicate pings. |
| `rafska/awesome-local-llm#118` FunASR local CPU ASR listing | Exposes the FunASR llama.cpp/GGUF path to local-LLM users looking for fully local CPU speech recognition | Current PR is README-only, clean, and has fresh link validation; monitor for maintainer feedback without adding another status comment. |
| `krzemienski/awesome-video#102` FunClip AI video tools listing | Adds FunClip to a curated streaming/video tools list where video engineers and creator-tool builders discover AI-assisted clipping and subtitle generation workflows | Sourcery's rerun passes on head `f90be00`; the PR is README-only, mergeable clean, and uses owner/repo-style link text `[modelscope/FunClip]`. LauraGPT posted a concise ready/validation note on 2026-07-08 at `https://github.com/krzemienski/awesome-video/pull/102#issuecomment-4911378040`; now wait for maintainer review without duplicate pings. |
| `CopilotKit/CopilotKit#5863` voice runtime transcription docs | Puts self-hosted OpenAI-compatible ASR behind a 35k-star agent UI voice surface by clarifying that custom `TranscriptionService` implementations can proxy CopilotKit `/transcribe` requests to local or LAN STT endpoints | LauraGPT posted a 2026-07-08 downstream compatibility note at `https://github.com/CopilotKit/CopilotKit/pull/5863#issuecomment-4911282937`, suggesting an optional self-hosted ASR callout for FunASR, SenseVoice, and Qwen3-ASR-style `/v1/audio/transcriptions` deployments. Wait for maintainer/doc-author feedback; do not turn this into a blocking review. |
| `manaflow-ai/cmux#7314` provider-neutral iOS voice backend | Puts a 23k-star AI terminal's iOS Voice Mode and composer dictation behind a recognition-backend seam where local/on-device Apple or Parakeet engines can later coexist with LAN/self-hosted FunASR/SenseVoice/Qwen3-ASR transcription gateways | LauraGPT posted a non-blocking backend-contract note at https://github.com/manaflow-ai/cmux/pull/7314#issuecomment-4912323678: keep engine capability metadata provider-neutral, send exactly one final transcript to `mobile.voice.input`, preserve fail-closed target checks outside the ASR engine, sanitize provider errors, and keep downloaded model state separate from any future remote endpoint settings. Track review without asking this PR to add another engine. |
| `QwenLM/qwen-code#6516` VoiceButton i18n and provider-error boundary | Makes voice dictation states and retry affordances translatable in the same high-visibility Qwen Web Shell surface, which matters for non-English users running local FunASR/SenseVoice STT backends | LauraGPT posted the local/private STT note at https://github.com/QwenLM/qwen-code/pull/6516#issuecomment-4911909587. Latest head `79f3294e651211de4095947eec35e33d94da5c7a` adds locale-cache invalidation after the earlier missed placeholder fix; Ubuntu Node 22 tests and coverage comment passed, while `review-pr` is still in progress. Reviewer `wenshao` downgraded to non-blocking comment at https://github.com/QwenLM/qwen-code/pull/6516#issuecomment-4912381983 and suggested i18n interpolation/timeline tests plus sentinel cleanup. Monitor checks/review; avoid another FunASR comment unless provider-error or voice insertion tests are requested. |
| `agent-of-empires/agent-of-empires#2585` composer action extension point | Adds a provider-agnostic composer action API in an agent UI, creating the right plugin boundary for local FunASR/SenseVoice dictation without hard-coding STT into core | Maintainer agreed with LauraGPT's provider-neutral boundary at https://github.com/agent-of-empires/agent-of-empires/pull/2585#issuecomment-4911979551 and asked the author to rebase. Keep STT-provider code out of this PR; after the extension point is clean, propose or review a plugin that calls an OpenAI-compatible FunASR/SenseVoice `/v1/audio/transcriptions` endpoint. |
| `tmoroney/auto-subs#652` Whisper download failure with SenseVoice fallback | Protects a live subtitle workflow after AutoSubs added SenseVoice: a Chinese-video user is blocked by `large-v3-turbo` whisper.cpp model download failure, not by audio normalization or ASR execution | LauraGPT posted safe triage at https://github.com/tmoroney/auto-subs/issues/652#issuecomment-4911843184: avoid unverified ZIP/bypass scripts, keep the Whisper fix in the model-manager/cache path, and use the built-in SenseVoice path from #629 plus the app translation route as the immediate `lang=zh` fallback. A new user asked for 1:1 help, so LauraGPT moved support back into the public issue at https://github.com/tmoroney/auto-subs/issues/652#issuecomment-4912472340; after they asked which option is SenseVoice, LauraGPT identified the UI label/model id at https://github.com/tmoroney/auto-subs/issues/652#issuecomment-4912653724. Track whether they can see `SenseVoice` / `sense-voice`, whether their build includes the newer engine, and whether Whisper remains isolated to download/cache handling. |
| `OpenWhispr/openwhispr#769` custom endpoint transcription contract | Keeps a privacy-first dictation app's BYOK/custom STT route compatible with OpenAI-style multipart transcription endpoints, the same wire shape used by self-hosted FunASR/SenseVoice gateways | LauraGPT posted the regression-contract checklist at https://github.com/OpenWhispr/openwhispr/issues/769#issuecomment-4909852583 after a user confirmed the OpenRouter fix branch worked. Track for a main/release-side fix that preserves `file`, `model`, optional `language`, optional `response_format=verbose_json`, exact `/audio/transcriptions` base-url joining, safe `audio/webm` handling, and provider-specific JSON/base64 only when required. |
| `mahseema/awesome-ai-tools#1689` FunClip video-tool listing | Adds FunClip to a high-visibility AI tools list where video creators discover clipping and subtitle workflows | Validate the FunClip link, keep the description aligned with local transcription/SRT/AI clipping capabilities, and avoid status pings after evidence is posted. |

Completed external wins:

| Integration | Growth reason | Result |
|---|---|---|
| `xinnan-tech/xiaozhi-esp32-server#3255` configurable FunASR language | Improves a high-star ESP32 voice-agent backend by letting users pin SenseVoice/FunASR language for short utterances | Merged on 2026-06-30 after adding the multi-module management-console database migration requested by maintainers. |
| `tmoroney/auto-subs#629` SenseVoice engine for subtitle workflows | Adds SenseVoice to an on-device subtitle generation app used by video editors and creators | Merged on 2026-06-30 after adding the declarative model manifest, SenseVoice/Canary/Cohere engines, real CTC timestamps, native translation routing, and robustness fixes. |
| `pipecat-ai/pipecat#4844` FunASR local STT service | Puts SenseVoice/FunASR into realtime voice agent pipelines used by builders comparing local STT backends | Merged on 2026-07-02 after adding the local STT service, optional dependency handling, docs, and validation coverage. |
| `liusongxiang/Large-Audio-Models#26` FunAudioLLM audio-language-model listing | Adds FunAudioLLM to a focused audio language model catalog where researchers compare speech and audio LLM projects | Merged on 2026-07-03 with a README-only FunAudioLLM entry. |
| `yuekaizhang/Fun-ASR-vllm#20` deterministic vLLM sampling for ASR | Improves a community Fun-ASR-vLLM inference path by making ASR generation deterministic enough for repeatable evaluation and demos | Merged on 2026-07-07 after switching the vLLM sampling path to deterministic decoding for ASR. Follow-up `#21` continues the prompt-embedding dtype cleanup. |
| `xorbitsai/inference#5140` Fun-ASR-Nano Xinference dependency fix | Unblocks Fun-ASR-Nano deployment through Xinference by moving model specs off an old FunASR git pin that predates `FunASRNano` registration | Merged on 2026-07-07 as `4a625fafa55827ee932f1e2cfcd362b9a7b4aabe`; LauraGPT synced FunASR deployment docs in https://github.com/modelscope/FunASR/pull/3117 and posted the merge follow-up at https://github.com/xorbitsai/inference/pull/5140#issuecomment-4901893926. Track Xinference release notes so FunASR docs can point users at a fixed build. |
| `debpalash/OmniVoice-Studio#1003` OpenAI-compatible ASR backend | Gives an 8k-star local voice/video studio a generic `POST /v1/audio/transcriptions` path that can point at self-hosted FunASR, SenseVoice, or Qwen3-ASR servers before direct Transformers support lands | Merged by maintainer `debpalash` on 2026-07-08 as `5a7d9cc05cda45682b11b5ece35190a5c39bbb6f` after LauraGPT's timeout, fallback-scope, key-clear, and error-sanitization review note. Watch release/user feedback for real FunASR/SenseVoice endpoint smoke results; do not post another follow-up unless those hardening items resurface. |
| `OpenWhispr/openwhispr#1093` transcript-retention fix | Strengthens a privacy-first dictation and meeting transcription app so local/private STT providers do not lose genuine microphone speech when echo evidence is audio-only | Merged on 2026-07-08 as `4d6f2d2e8b4602feb72f1493aff1f8f12de9e3f3` after LauraGPT highlighted the final-text retention contract; watch release feedback and future OpenAI-compatible/local provider work without posting duplicate follow-ups. |
| `elizaOS/eliza#15426` streaming ASR live transcript contract | Lands a high-star agent OS voice UX path where segment preview and final batch transcript semantics can support local/private FunASR or SenseVoice-style ASR providers | Merged on 2026-07-08 as `e8d780cd31d1b5e8a870c20e370da0010ce8eb2d` after LauraGPT called out the final-transcript safety contract at https://github.com/elizaOS/eliza/pull/15426#issuecomment-4911110675. Watch follow-up provider work and keep any FunASR bridge aligned with the final-only fallback contract. |
| `QwenLM/qwen-code#6510` pane-scoped voice dictation restore | Restores voice dictation in a 25k-star Qwen coding-agent Web Shell split-view, creating a visible local/private STT surface when the daemon advertises `voice_transcribe` | Merged on 2026-07-08 as `d8dc8043d6cfd4d7605f83b8a4838fee9dacdffa` after LauraGPT's pane-scoped local/private ASR boundary note at https://github.com/QwenLM/qwen-code/pull/6510#issuecomment-4911877569 and maintainer confirmation at https://github.com/QwenLM/qwen-code/pull/6510#issuecomment-4912022943: capability-gated mic visibility, editable transcript draft without auto-submit, pane-specific routing, and sanitized provider errors for FunASR/SenseVoice/OpenAI-compatible STT backends. |
| `chatmcp/mcpso#1` MCP server directory submission | Routes MCP builders browsing mcp.so toward FunASR's local speech-recognition MCP server | Listed on mcp.so on 2026-07-08 at https://mcp.so/server/mcp-server-funasr/radial-hks after LauraGPT submitted the server in the public directory thread. The MCP README now links the live directory page so Claude/Cursor-style users can discover `transcribe_audio` without reading the growth tracker. |

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
