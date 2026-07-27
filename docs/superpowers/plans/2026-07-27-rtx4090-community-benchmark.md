# RTX 4090 Community Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a clearly scoped, source-linked summary of the Fun-ASR Nano RTX 4090 community validation on the English and Chinese FunASR benchmark pages.

**Architecture:** Extend the product-site Jinja benchmark source template with one language-aware section and verify both generated routes. Reuse existing benchmark-card CSS patterns, keep all claims traceable to `sgl-project/sglang-omni#1170`, and clearly distinguish community validation from released upstream support.

**Tech Stack:** Jinja2 templates, pytest/BeautifulSoup generated-output tests, product-site validator, local HTTP server, Playwright screenshots.

## Global Constraints

- Do not claim that the SGLang-Omni integration has merged or shipped.
- Report only measurements published in `sgl-project/sglang-omni#1170`.
- Keep existing navigation and CSS unchanged.
- Preserve English and Chinese content parity.

---

### Task 1: Define the generated-output contract

**Files:**
- Modify: `web-pages/product-site/tests/test_output.py`

**Interfaces:**
- Consumes: generated `/benchmarks.html` and `/en/benchmarks.html`
- Produces: a regression test for the RTX 4090 section, evidence links, metrics, and release caveat

- [ ] **Step 1: Write the failing generated-output test**

Add `test_rtx4090_community_benchmark_is_bilingual_and_qualified`. For both language routes, require `[data-community-benchmark="rtx4090"]`, `105,067`, `16.11 GiB`, `0.0175`, `0.0164`, the four source URLs, and language-specific text stating the integration is not released.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
python -m pytest web-pages/product-site/tests/test_output.py -k rtx4090 -q
```

Expected: fail because the section is absent.

### Task 2: Add the bilingual community benchmark

**Files:**
- Modify: `web-pages/product-site/templates/benchmarks.html`

**Interfaces:**
- Consumes: `language`, existing `benchmark-record`, `benchmark-fields`, and `text-link` template styles
- Produces: `[data-community-benchmark="rtx4090"]` in both generated benchmark routes

- [ ] **Step 1: Add the section and evidence card**

Insert a new unframed section after the public-record list. Use Jinja language branches for headings and explanatory copy, while keeping model names, metrics, revisions, and URLs identical.

- [ ] **Step 2: Include complete evidence and boundaries**

Render hardware/software versions, SeedTTS EN/ZH c=1 and c=16 throughput/latency/quality, the 30-minute soak, memory, 30-second request limit, terminal-event-only behavior, dirty-worktree provenance, and pending upstream integration.

- [ ] **Step 3: Run the focused test**

```bash
python -m pytest web-pages/product-site/tests/test_output.py -k rtx4090 -q
```

Expected: pass.

### Task 3: Validate and publish the isolated branch

**Files:**
- Test: `web-pages/product-site/templates/benchmarks.html`
- Test: `web-pages/product-site/tests/test_output.py`

**Interfaces:**
- Consumes: product-site source and tests from Tasks 1-2
- Produces: a pushed `codex/rtx4090-benchmark-20260727` branch and reviewable pull request

- [ ] **Step 1: Run the product-site suite**

```bash
python -m pytest web-pages/product-site/tests -q
```

Expected: all tests pass.

- [ ] **Step 2: Build and validate generated output**

Run two builds, compare them recursively, and run `validate.py` on the output. Request the four linked GitHub/Gist sources and require HTTP status below 400.

- [ ] **Step 3: Inspect rendered pages**

Serve the generated output with:

```bash
python3 -m http.server 8765
```

Capture desktop `1440x1000` and mobile `390x844` screenshots of both pages. Confirm the new evidence card does not overflow and text does not overlap.

- [ ] **Step 4: Run repository checks**

Run:

```bash
git diff --check origin/main...HEAD
git status --short
```

Expected: no whitespace errors and only the intended files changed.

- [ ] **Step 5: Commit and publish**

```bash
git add web-pages/product-site/templates/benchmarks.html web-pages/product-site/tests/test_output.py
git commit -m "docs: add RTX 4090 Fun-ASR benchmark"
git push -u origin codex/rtx4090-benchmark-20260727
```

Open a ready pull request against `main` with the validation evidence and explicit community/unmerged caveat.
