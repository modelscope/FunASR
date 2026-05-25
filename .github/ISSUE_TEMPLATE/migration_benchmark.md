---
name: 📊 Migration Benchmark Report
about: Share FunASR results when comparing against Whisper, OpenAI audio APIs, or cloud ASR
labels: 'benchmark, showcase, needs triage'
---

Thanks for benchmarking FunASR on your own audio. Migration reports help new users decide whether FunASR fits their language, domain, hardware, and deployment constraints.

If you need help debugging a failure, please use Bug Report or Deployment Help instead.

## Summary

<!-- One or two sentences: what did you compare, and what was the headline result? -->

## Baseline

- Baseline ASR (`Whisper`, `Whisper large-v3`, cloud provider, internal system, other):
- Baseline runtime or API:
- Baseline hardware or pricing tier:

## FunASR setup

- FunASR version:
- Model(s):
- Runtime path (`Python API`, `funasr-server`, `OpenAI API`, `Docker`, `WebSocket`, `vLLM`, other):
- Device (`cuda`, `cpu`, `mps`):
- GPU / CPU:
- CUDA / PyTorch versions:
- Command or script used:

```bash

```

## Audio set

- Number of files:
- Total audio duration:
- Language(s) / dialect(s):
- Domain (`meeting`, `call-center`, `subtitle`, `lecture`, `media`, `noisy field audio`, other):
- Speaker count range:
- Sample rate / format:
- Can any sample be shared publicly? yes/no

## Results

<!-- Paste the aggregate section from examples/migration/benchmark_funasr.py summary.md, or summarize your own measurement. -->

```text

```

## Quality notes

<!-- Share WER/CER if available, or human-review notes about names, numbers, punctuation, timestamps, speaker labels, and domain terms. -->

## Operational notes

- Model download / warmup time:
- Steady-state throughput:
- Memory usage:
- Failed files or error rate:
- Deployment blockers:

## Links or artifacts

<!-- Public repo, demo, blog, benchmark sheet, screenshot, anonymized transcript snippet, or architecture diagram. Do not include private audio, credentials, or customer data. -->

## What should FunASR improve next?

<!-- Missing docs, rough edges, model gaps, deployment friction, benchmark needs, etc. -->
