# FunASR Colab Quickstart

English | [简体中文](README_zh.md) | [日本語](README_ja.md) | [한국어](README_ko.md)

Run FunASR in a browser without preparing a local Python environment.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modelscope/FunASR/blob/main/examples/colab/funasr_quickstart.ipynb)

## What the notebook covers

- Install FunASR and runtime dependencies in Colab.
- Pick `cuda:0` automatically when a Colab GPU is available, otherwise use CPU.
- Transcribe a public sample audio file with `paraformer-zh`, VAD, and punctuation.
- Upload your own audio file and run the same model.
- Save the transcript JSON for sharing or issue reports.

## Notes

- The first run downloads model files and may take a few minutes.
- CPU runtime works for a quick smoke test; GPU runtime is faster for longer audio.
- If you are evaluating production deployment, use the [deployment matrix](../../docs/deployment_matrix.md) after the notebook works.
- For OpenAI-compatible HTTP service testing, use [examples/openai_api](../openai_api/).

## Troubleshooting

| Symptom | What to try |
|---|---|
| Colab runtime disconnects or resets | Reconnect the runtime, rerun the install cell, then rerun the model cell. The notebook does not persist Python packages across runtime resets. |
| GPU is unavailable | Use `Runtime > Change runtime type > GPU`. If no GPU is assigned, the notebook still works on CPU for a short smoke test. |
| Model download is slow | Rerun the cell after the network recovers. The first run downloads model files and later runs in the same runtime are faster. |
| Uploaded audio fails or is too large | Try a short WAV/MP3 first. For long files, trim a representative sample before using Colab. |
| Output is unexpected | Save the transcript JSON cell output and include it when opening an issue. |

Notebook source: [funasr_quickstart.ipynb](https://github.com/modelscope/FunASR/blob/main/examples/colab/funasr_quickstart.ipynb).
