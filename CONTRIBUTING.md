# Contributing to FunASR

Thanks for helping improve FunASR. We especially welcome contributions that make the first successful transcription faster, improve deployment reliability, or make benchmarks easier to reproduce.

## Repository scope

FunASR spans four repositories. Before opening an issue or PR, check which repo owns the area you are working on:

| Repository | Owns |
|---|---|
| `modelscope/FunASR` | Toolkit, inference pipelines, deployment services, `funasr` PyPI package |
| `QwenAudio/Fun-ASR` | Fun-ASR-Nano / MLT model family, checkpoints, model cards, benchmarks, model-level integrations (Transformers, vLLM, GGUF) |
| `QwenAudio/SenseVoice` | SenseVoice model: ASR + emotion + audio events |
| `modelscope/FunClip` | Video transcription, subtitles, Gradio UI |

**Quick test:** does the problem persist with a different model? If yes, it belongs in `modelscope/FunASR`. If it only affects one model, file it in that model's repo.

Full details: [Repository roles & roadmap](./docs/repository_roles.md) ([中文](./docs/repository_roles_zh.md))

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

## Maintainer issue lifecycle

Issue closure records a verified outcome, not only that a related pull request was merged or a package was released.

- Keep a bug open with the `needs feedback` label while the reporter tests the fix in the environment that reproduced it.
- Close after the reporter confirms the problem is resolved, or after a maintainer reproduces the original failure, verifies the fix under equivalent conditions, and leaves a reasonable feedback window.
- For long-running, hardware-specific, accuracy, or deployment bugs, synthetic and unit tests are supporting evidence; they do not replace a reporter's real workload unless the original workload is available and reproduced.
- If a reporter says the problem still occurs, reopen the issue and record the new evidence, even when the associated code change has already shipped.
- When closing, link the exact fix or documentation, the released version when applicable, and the verification evidence. State any remaining boundary explicitly.

## Maintainer focus for 20k+ stars

When reviewing changes, prioritize work that helps new users reach a good result in under five minutes, helps teams deploy FunASR privately, or gives external writers a clear story to share.
