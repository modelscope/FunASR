# Contributing to FunASR

Thanks for helping improve FunASR. We especially welcome contributions that make the first successful transcription faster, improve deployment reliability, or make benchmarks easier to reproduce.

## High-impact areas

- **Quick start reliability:** installation notes, CPU/GPU/MPS compatibility, dependency fixes, and runnable examples.
- **Deployment recipes:** OpenAI-compatible API, WebSocket streaming, Docker, vLLM, Triton, Android, browser, and agent integration.
- **Benchmarks:** reproducible speed, WER/CER, memory, and hardware comparison scripts.
- **Model examples:** multilingual ASR, speaker diarization, punctuation, VAD, emotion recognition, hotwords, timestamps, and fine-tuning.
- **Documentation:** shorter paths from README to working code, clearer troubleshooting, and verified links.

## Development setup

```bash
git clone https://github.com/modelscope/FunASR.git
cd FunASR
python -m venv .venv
source .venv/bin/activate
pip install -e ./
```

For docs work:

```bash
pip install -U "funasr[docs]"
cd docs
make html
```

## Before opening a pull request

Run the checks that match your change. For Python-only changes, start with:

```bash
python -m compileall funasr examples tests
```

For docs-only changes, preview the Markdown or generated HTML and verify relative links. For runtime changes, include the exact command, image tag, device, and endpoint you validated.

## Pull request checklist

- The PR has a focused scope and explains the user-facing value.
- New examples include command lines and the expected output shape.
- Bug fixes include reproduction steps or a short failure-mode explanation.
- Deployment docs list hardware, OS, Python/CUDA versions, and network endpoints.
- Large model, dataset, audio, or video files are not committed directly.

## Issue reports

Please use the templates and include environment details, exact commands, logs, and whether the audio can be shared. If audio is private, describe duration, sample rate, language, speaker count, format, and noise level.

## Maintainer focus for 20k+ stars

When reviewing changes, prioritize work that helps new users reach a good result in under five minutes, helps teams deploy FunASR privately, or gives external writers a clear story to share.
