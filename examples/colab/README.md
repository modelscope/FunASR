# FunASR Colab Quickstart

English | [简体中文](README_zh.md)

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

Notebook source: [funasr_quickstart.ipynb](https://github.com/modelscope/FunASR/blob/main/examples/colab/funasr_quickstart.ipynb).
