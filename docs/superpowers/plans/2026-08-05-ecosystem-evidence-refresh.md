# Ecosystem Evidence Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the bilingual FunASR ecosystem page with current owned-repository scale, the latest stable FunClip release, and the merged audio.cpp Fun-ASR-Nano runtime.

**Architecture:** Keep the existing legacy ecosystem pages as the source surface consumed by the approved product-site builder. Add output-level assertions before editing content, update only the two bilingual pages, then refresh their existing snapshot hashes in the legacy manifest.

**Tech Stack:** Static HTML, Python 3.11+, Beautiful Soup, pytest, the existing product-site builder and validator, Playwright.

## Global Constraints

- Follow the approved Option A product deployment hub design in `docs/superpowers/specs/2026-07-26-funasr-product-deployment-hub-design.md`.
- Do not claim an integration is merged or released without exact public evidence.
- Keep Chinese and English ecosystem cards structurally equivalent.
- Preserve all existing routes, navigation, legacy content, and attributed `/go/funclip` repository link.
- Do not add dependencies or change unrelated site content.

---

### Task 1: Lock the ecosystem evidence contract

**Files:**
- Modify: `web-pages/product-site/tests/test_output.py`

**Interfaces:**
- Consumes: the existing `built_site` fixture and `read_soup()` helper.
- Produces: a bilingual regression gate for the ecosystem scale, FunClip release, and audio.cpp integration evidence.

- [x] **Step 1: Write the failing output test**

Add a parametrized test for `ecosystem.html` and `en/ecosystem.html`. It must assert that the rendered page contains `36K+`, links to `https://github.com/modelscope/FunClip/releases/tag/v2.1.1`, and contains an audio.cpp card linking to all of:

```text
https://github.com/0xShug0/audio.cpp
https://github.com/0xShug0/audio.cpp/pull/155
https://github.com/0xShug0/audio.cpp/blob/1778b23a5f6a4951c788e4bb0e7baa04f20012a2/docs/models/fun_asr_nano.md
```

The card text must include `Fun-ASR-Nano`, `CPU`, `CUDA`, `CLI`, and `OpenAI`.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
python -m pytest web-pages/product-site/tests/test_output.py -k ecosystem_refresh -q
```

Expected: fail because the current page says `35K+`, links FunClip `v2.1.0`, and has no audio.cpp card.

### Task 2: Refresh the bilingual ecosystem source

**Files:**
- Modify: `web-pages/product-site/legacy/ecosystem.html`
- Modify: `web-pages/product-site/legacy/en/ecosystem.html`
- Modify: `web-pages/product-site/content/legacy-manifest.json`

**Interfaces:**
- Consumes: the output contract from Task 1 and public GitHub evidence for FunClip `v2.1.1` and audio.cpp PR #155.
- Produces: bilingual legacy pages accepted by the unchanged product-site builder.

- [x] **Step 1: Apply the minimal bilingual content update**

Change the aggregate ecosystem figure from `35K+` to `36K+`, update the FunClip release link and label from `v2.1.0` to `v2.1.1`, and add one audio.cpp card under cross-platform inference. Describe only the merged native Fun-ASR-Nano implementation: pure C++/GGML, CPU/CUDA, CLI, and OpenAI-compatible local HTTP. Link the repository, merged PR, and pinned merged guide.

- [x] **Step 2: Refresh only the two snapshot hashes**

Compute SHA-256 for the two modified legacy pages and replace only their existing values in `legacy-manifest.json`. Keep `captured` and every unrelated file hash unchanged.

- [x] **Step 3: Run the focused test and verify GREEN**

Run:

```bash
python -m pytest web-pages/product-site/tests/test_output.py -k ecosystem_refresh -q
```

Expected: both language cases pass.

- [x] **Step 4: Run complete site verification**

Run:

```bash
python -m pytest web-pages/product-site/tests -q
python web-pages/product-site/build.py --output /tmp/funasr-site-ecosystem-refresh
python web-pages/product-site/validate.py /tmp/funasr-site-ecosystem-refresh
cd web-pages/product-site/tests/browser && npm ci && npx playwright test
git diff --check
```

Expected: all tests and validators pass, Playwright desktop/mobile coverage passes, and the diff is whitespace-clean.

- [ ] **Step 5: Commit and publish for review**

Create a signed+DCO commit containing exactly the test, two pages, manifest hashes, and this plan. Push `codex/site-ecosystem-refresh-20260805`, open a ready PR, wait for required checks, and merge only after the exact pushed head is green and mergeable. After deployment, run `python scripts/check_funasr_website_static.py` and direct HTTPS checks against both ecosystem routes.
