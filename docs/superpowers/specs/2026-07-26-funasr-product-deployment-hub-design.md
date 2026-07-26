# FunASR Product Deployment Hub Design

Status: approved direction (Option A) on 2026-07-26

## 1. Purpose

Turn `www.funasr.com` from a collection of project and tutorial pages into the
product entry point for deploying FunASR in production. The primary audience is
developers, platform engineers, and technical leads preparing a private speech
deployment. Researchers and first-time evaluators remain supported through
secondary paths.

The site must help a visitor answer four questions without searching the GitHub
repository:

1. Which runtime fits my workload and hardware?
2. What is the shortest verified command that proves it works?
3. What must I add before exposing it to production traffic?
4. Where are the exact release assets, evidence, and limitations?

This work supports the September community-growth objective by converting the
site's existing traffic into successful deployments, repeat visits, release
downloads, and GitHub engagement.

## 2. Current-State Evidence

The live site already receives meaningful traffic. On 2026-07-26 it reported
5,527 page views and 2,182 unique visitors. The home page had 1,485 views, while
`quickstart.html` had 82 and `models.html` had 76. This indicates that the site
has distribution but the path from landing to deployment is weak.

The content surface is fragmented:

- The home page mentions industrial deployment but does not provide a deployment
  decision flow.
- `llama-cpp.html` exists, but vLLM, WebSocket, OpenAI API, containers, ONNX,
  Triton, and operations do not form a coherent product family.
- The GitHub `web-pages` Vue source and the manually maintained live static HTML
  have diverged.
- The live `tracker.js` posts to `/stats/log`, but Nginx has no matching route, so
  `github_clicks` remains zero and conversion cannot be measured.
- Nginx serves directly from a mutable `dist` directory, so deployment and
  rollback are not atomic.
- The configured TLS protocol list still includes TLS 1.0 and 1.1.

The repository already contains strong source material: a deployment matrix,
model-selection guides, an OpenAI-compatible API with health and model-list
checks, Docker Compose and Kubernetes examples, production WebSocket guidance,
vLLM benchmarks and concurrency notes, ONNX/C++ and Triton runtimes, and nine
cross-platform llama.cpp/GGUF release assets.

## 3. Scope

### 3.1 Included in the first release

- Redesign the Chinese and English home pages around deployment selection.
- Add a bilingual `/deploy/` product center and seven deployment detail pages.
- Introduce a single data model for deployment status, hardware, models,
  commands, evidence, limitations, and release links.
- Normalize primary navigation across generated and legacy pages while keeping
  `功德榜` / `Donors` as the final navigation item.
- Preserve the current blog, demos, voice assets, and high-value indexed URLs.
- Import the public live-site corpus into source control, excluding backups,
  access statistics, logs, and generated output, so a release can be rebuilt
  without reading mutable files from the production host.
- Preserve `/llama-cpp.html` and `/en/llama-cpp.html` as permanent compatibility
  entry points with canonical links to the new deployment pages.
- Add privacy-preserving first-party conversion measurement for fixed GitHub,
  documentation, release, and deployment calls to action.
- Replace mutable in-place deployment with versioned releases and an atomic
  `current` symlink.
- Harden Nginx for TLS 1.2/1.3, security headers, cache policy, and deterministic
  redirects.

### 3.2 Explicitly excluded from the first release

- Rewriting all existing blog articles.
- Hosting a public inference API on `www.funasr.com`.
- Claiming a runtime is production-ready without repository evidence.
- Adding built-in authentication, rate limiting, or Prometheus metrics to the
  FunASR server in the website change. The site will state which controls are
  native and which belong at an API gateway.
- Migrating the entire site to Astro or another full content framework.

## 4. Product Positioning

The first viewport uses the literal product name as the heading:

> FunASR

Supporting copy:

> 可私有化部署的语音智能基础设施。覆盖 GPU 高吞吐、实时流式服务、
> OpenAI 兼容 API，以及 CPU 与边缘独立运行。

English:

> Private-deployment speech infrastructure for high-throughput GPU inference,
> real-time streaming, OpenAI-compatible APIs, and standalone CPU or edge use.

The primary call to action is `选择部署方案` / `Choose a deployment`. The
secondary action is `5 分钟验证` / `Verify in five minutes`. GitHub remains
visible in the header but does not replace the deployment workflow.

## 5. Information Architecture

Primary navigation order:

1. Product / 产品
2. Deploy / 部署
3. Models / 模型
4. Benchmarks / 性能
5. Ecosystem / 生态
6. Blog / 博客
7. Docs / 文档
8. Donors / 功德榜

Language selection and the GitHub action remain separate header controls.

New routes:

| Chinese route | English route | Purpose |
|---|---|---|
| `/deploy/` | `/en/deploy/` | Workload and hardware deployment selector |
| `/deploy/vllm.html` | `/en/deploy/vllm.html` | Fun-ASR-Nano GPU batch and high-throughput inference |
| `/deploy/llama-cpp.html` | `/en/deploy/llama-cpp.html` | CPU, desktop, and edge GGUF runtime |
| `/deploy/openai-api.html` | `/en/deploy/openai-api.html` | OpenAI-compatible private transcription API |
| `/deploy/realtime.html` | `/en/deploy/realtime.html` | WebSocket live captions and streaming ASR |
| `/deploy/containers.html` | `/en/deploy/containers.html` | Docker Compose and Kubernetes service deployment |
| `/deploy/cpu-runtime.html` | `/en/deploy/cpu-runtime.html` | ONNX/C++ high-concurrency CPU deployment |
| `/deploy/production.html` | `/en/deploy/production.html` | Security, readiness, capacity, observability, and rollout checklist |
| `/benchmarks.html` | `/en/benchmarks.html` | Reproducible benchmark evidence and methodology |

Existing `quickstart.html`, `models.html`, `ecosystem.html`, `/blog/`, and donor
routes remain stable.

## 6. Page Design

### 6.1 Home page

The home page is an operational product surface, not a marketing collage.

1. Full-width brand hero with one custom deployment-topology bitmap and real
   terminal/API output layered as inspectable HTML, not baked into the image.
2. A compact deployment selector with segmented controls for workload and
   hardware. It recommends one path and explains why.
3. A proof band for stable facts: Apache-2.0, supported operating-system families,
   OpenAI-compatible endpoint, verified release date, and benchmark evidence.
4. A deployment matrix organized by workload, hardware, interface, and maturity.
5. A real API contract example showing `/health`, `/v1/models`, and
   `/v1/audio/transcriptions`.
6. A production-readiness band separating native capability from gateway or
   infrastructure responsibility.
7. Verified runtime downloads and ecosystem adoption.
8. A final deployment and GitHub call to action.

The next section remains visible at common desktop and mobile viewport heights.

### 6.2 Deployment selector

Inputs:

- Workload: file batch, real-time stream, private API, or edge application.
- Hardware: NVIDIA GPU, general CPU, desktop/edge GPU, or Kubernetes cluster.
- Priority: throughput, latency, portability, or compatibility.

Output:

- Recommended runtime.
- Supported model family.
- Why it matches.
- Primary limitation.
- Direct link to the detailed deployment page.

The selector is deterministic and encoded in the deployment registry. It does
not send data to a server and works without JavaScript by exposing the full
matrix below it.

### 6.3 Deployment detail template

Every deployment page uses the same scan-friendly structure:

1. Status, last verified date, tested FunASR version, and evidence links.
2. `适合` / `不适合` guidance.
3. Supported models, hardware, operating systems, and interfaces.
4. Copy-ready installation and startup commands.
5. Health or smoke-test command with expected output.
6. Architecture and request flow.
7. Capacity-planning variables and benchmark method.
8. Security and network boundary.
9. Operations: readiness, logs, model cache, graceful restart, and rollback.
10. Known limitations and troubleshooting.
11. Release assets, deeper documentation, issue template, and GitHub action.

### 6.4 Benchmark page

No unqualified performance number may appear. Each metric must include model,
runtime, hardware, dataset or audio duration, batch/concurrency settings, whether
download and warmup were excluded, source link, and verification date. The page
distinguishes batch RTFx from real-time streaming capacity and explicitly warns
against treating one traffic profile as a universal concurrency promise.

## 7. Visual System

- Base: white and near-black with cobalt blue for actions, green for verified
  status, and amber for limitations. Avoid a one-hue dark/slate presentation.
- Type: system fonts with `Inter`, `PingFang SC`, `Microsoft YaHei`, and sans-serif
  fallbacks. Remove the Google Fonts dependency for reliable access in China.
- Corners: 4-8 px. Do not nest cards or turn page sections into floating cards.
- Icons: bundled Lucide icons with text labels and tooltips for unfamiliar icons.
  The selected SVG files and Lucide license are checked into the site assets from
  one pinned upstream release; no icon is redrawn locally.
- Commands: familiar copy icon buttons, not text-filled controls where an icon is
  sufficient.
- Layout: dense, calm, and technical. Tables and unframed bands take priority over
  decorative cards.
- Motion: only selector transitions, copy confirmation, and reduced-motion-safe
  topology movement. No decorative orbs, bokeh, or gradient background art.
- Hero asset: a custom bitmap that depicts the actual deployment spectrum from
  GPU server to CPU/edge clients. It must remain legible under the text overlay
  and may not contain generated text.

## 8. Content and Evidence Model

The source of truth is a bilingual deployment registry checked into the FunASR
repository. Each entry contains:

- stable id and route
- Chinese and English name, summary, fit, and limitations
- maturity: `production-verified`, `community-verified`, or `experimental`
- supported models, hardware, operating systems, and interfaces
- install, launch, health, and smoke-test commands
- tested FunASR/runtime versions and verification date
- benchmark records with complete conditions
- release assets and documentation evidence
- operations and security responsibilities

The build fails if a production-verified entry lacks a verification date,
evidence link, tested version, smoke test, or explicit limitation. URLs and model
aliases are validated. Chinese and English entries must have the same structure.

## 9. Static Build Architecture

Add a self-contained product-site source under `web-pages/product-site/`:

```text
web-pages/product-site/
  assets/
  content/
  data/deployments.json
  legacy/
  templates/
  build.py
  validate.py
  requirements-site.txt
  tests/
```

`build.py` runs on Python 3.11 or newer, uses pinned Jinja2 and Beautiful Soup
build dependencies, and emits dependency-free static HTML, CSS, JavaScript,
sitemap entries, redirects, and a deployment manifest. The live server remains
Nginx-only. Existing blog and demo assets are copied from the tracked `legacy/`
snapshot, never from the mutable production directory.

A structured HTML parser updates primary navigation in legacy pages. It must be
idempotent and preserve article content. The same navigation manifest feeds all
new pages and the legacy normalizer.

Generated assets use versioned filenames. HTML is no-cache; immutable assets are
cached long term. The build output is complete in a staging directory before any
live file changes.

## 10. Deployment and Rollback

Deploy to:

```text
/root/FunASR/web-pages/releases/<UTC timestamp>/
```

After validation, atomically switch:

```text
/root/FunASR/web-pages/current -> releases/<UTC timestamp>
```

Nginx serves `current`. Deployment steps:

1. Build in an isolated worktree.
2. Validate generated HTML, links, sitemap, language pairs, and asset hashes.
3. Upload to a new release directory.
4. Run live-root checks against the staged directory.
5. Back up Nginx configuration, run `nginx -t`, and reload only after success.
6. Switch the symlink atomically.
7. Run public HTTPS and browser smoke tests.
8. Roll back by switching to the previous release if any required check fails.

No command mutates or deletes an earlier release during deployment.

## 11. Nginx and Measurement

Nginx changes:

- Serve the atomic `current` directory.
- Allow TLS 1.2 and 1.3 only.
- Add a content security policy compatible with the local assets and the existing
  approved video embed.
- Add `Permissions-Policy`, retain HSTS, frame, content-type, and referrer headers.
- Use no-cache for HTML and immutable caching for versioned assets.
- Add fixed 302 routes for GitHub, docs, releases, and deployment calls to action.
  These routes are not open redirects.
- Write the fixed redirect routes to a dedicated conversion access log.

The broken beacon endpoint is removed. Conversion reporting is derived from
first-party access logs and stores no audio, form input, cookies, or persistent
visitor identifier. Reports include page views, deployment-detail views, and
fixed outbound actions.

## 12. Failure Handling

- Build errors, missing evidence, invalid bilingual structure, broken links, or
  sitemap mismatch stop deployment.
- Legacy navigation normalization runs against a copy and aborts if an expected
  structure cannot be parsed.
- The deployment selector always has a no-JavaScript matrix fallback.
- Unknown routes receive a bilingual 404 with deployment, docs, and home links.
- Existing indexed routes remain available or use permanent redirects with
  canonical and hreflang metadata.
- A failed Nginx check or public smoke test leaves the previous release active.

## 13. Verification

### Build and content

- Unit tests for registry validation, route generation, selector decisions, and
  navigation normalization.
- Generated-output tests for all Chinese/English route pairs.
- HTML parsing and internal-link checks across every generated and legacy page.
- Sitemap, canonical, hreflang, robots, structured-data, and redirect checks.
- Exact model alias, package version, release asset, and documentation link checks.
- Evidence checks for every benchmark and production-verified status.

### Browser

- Playwright on 390x844, 768x1024, 1440x900, and 1920x1080.
- Screenshots for home, deployment index, all detail templates, mobile navigation,
  selector states, and bilingual pages.
- No horizontal overflow, overlapping controls, hidden headings, or text clipping.
- Keyboard navigation, visible focus, semantic landmarks, contrast, alt text, and
  reduced-motion behavior.
- The hero asset loads and the first viewport reveals the next section.
- Copy buttons, language links, selector recommendations, old llama.cpp routes,
  and fixed outbound redirects work.

### Live

- HTTPS 200 for all canonical routes and assets.
- Correct cache and security headers.
- TLS 1.0/1.1 rejected; TLS 1.2/1.3 accepted.
- Public sitemap routes equal the generated manifest.
- Conversion log records fixed outbound actions.
- Existing blog and demo contracts remain green.

## 14. Rollout Order

1. Add generator, registry, validation, and tests.
2. Build the bilingual home page and deployment index.
3. Build the seven deployment pages and benchmark page.
4. Normalize legacy navigation and preserve old routes.
5. Add versioned deployment, Nginx hardening, and conversion logs.
6. Run full local and staged-browser verification.
7. Deploy atomically, run public verification, and monitor the first 24 hours.

## 15. Success Criteria

The first release is complete only when:

- All planned bilingual routes are public and pass the generated manifest.
- A visitor can reach a justified deployment recommendation in two interactions.
- Every production claim has exact repository or release evidence.
- vLLM and llama.cpp/GGUF each have a complete production-oriented page.
- Old high-value URLs and all blog/demo contracts remain valid.
- Deployment is atomic and rollback is proven.
- GitHub, docs, release, and deployment CTA conversion is measurable.
- Desktop and mobile browser checks pass with no overlap or overflow.
- The live site passes security-header, TLS, sitemap, and link verification.
