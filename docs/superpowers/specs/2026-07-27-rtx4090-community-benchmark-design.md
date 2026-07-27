# RTX 4090 Community Benchmark Design

## Goal

Add a trustworthy, production-oriented summary of the completed Fun-ASR Nano single-RTX-4090 community validation to the FunASR benchmark pages without presenting unmerged SGLang-Omni integration work as an official supported runtime.

## Scope

- Update the product-site benchmark source template on `main`; do not edit generated `gh-pages` output.
- Render one self-contained community-validation section in both English and Chinese.
- Link the upstream benchmark issue, integration roadmap, draft PR, and raw JSON artifacts.
- Add a focused generated-output regression test.
- Leave navigation, styling, deployment registry, and all other pages unchanged.

## Content

The new section will identify the environment as a single RTX 4090 24 GB and report only measurements published in `sgl-project/sglang-omni#1170`:

- SeedTTS English and Chinese throughput, latency, and WER/CER at concurrency 1 and 16.
- The 30-minute mixed-concurrency soak result of 105,067 successful requests and zero unexpected errors.
- Startup and post-graph-capture memory.
- Operational constraints: 30-second request limit, no low-latency delta claim, dirty feature worktree provenance, and pending upstream integration.

The section will call this a community validation and will not claim support in a released SGLang-Omni version.

## Presentation

Use the product site's existing `benchmark-record` and `benchmark-fields` components so the data remains readable without horizontal scrolling on mobile. No new CSS or visual asset is needed because this is an evidence card inside the existing benchmark experience, not a new page or hero.

## Verification

- Run the product-site pytest suite, including a focused assertion for both generated language routes.
- Run the deterministic product-site build and validator.
- Check the source URLs return successful HTTP responses.
- Run `git diff --check`.
- Serve the generated site and inspect desktop and mobile screenshots for overflow or overlap.

## Rollback

The work is isolated on `codex/rtx4090-benchmark-20260727` from recorded `main` commit `1e0200916488bc749565aaf67818d056efbc57e8`. The generated-page experiment and original pages are retained in the patrol evidence directory and an unpushed `gh-pages` backup branch.
