# FunASR Colab 快速体验

[English](README.md) | 简体中文

无需提前配置本地 Python 环境，直接在浏览器里运行 FunASR。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modelscope/FunASR/blob/main/examples/colab/funasr_quickstart.ipynb)

## Notebook 覆盖内容

- 在 Colab 中安装 FunASR 和运行依赖。
- 如果 Colab runtime 有 GPU，自动使用 `cuda:0`；否则使用 CPU。
- 使用 `paraformer-zh`、VAD 和标点模型转写公开样例音频。
- 上传自己的音频文件，并用同一套模型运行转写。
- 保存 transcript JSON，便于分享、对比或提交 issue。

## 使用建议

- 第一次运行会下载模型文件，可能需要几分钟。
- CPU runtime 适合快速 smoke test；长音频建议切换到 GPU runtime。
- 如果要评估生产部署，先跑通 notebook，再阅读 [部署选型表](../../docs/deployment_matrix_zh.md)。
- 如果要测试 OpenAI 兼容 HTTP 服务，请使用 [examples/openai_api](../openai_api/README_zh.md)。

Notebook 源文件：[funasr_quickstart.ipynb](https://github.com/modelscope/FunASR/blob/main/examples/colab/funasr_quickstart.ipynb)。
