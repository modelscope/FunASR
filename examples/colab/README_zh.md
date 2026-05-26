# FunASR Colab 快速体验

[English](README.md) | 简体中文 | [日本語](README_ja.md) | [한국어](README_ko.md)

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

## 故障排查

| 现象 | 处理方式 |
|---|---|
| Colab runtime 断开或重置 | 重新连接 runtime，先重跑安装单元，再重跑模型单元。runtime 重置后不会保留已安装的 Python 包。 |
| 没有 GPU | 通过 `Runtime > Change runtime type > GPU` 切换。没有分配到 GPU 时，短音频 smoke test 仍可用 CPU 跑通。 |
| 模型下载很慢 | 网络恢复后重跑该单元。第一次运行会下载模型文件，同一个 runtime 后续运行会更快。 |
| 上传音频失败或文件太大 | 先用短 WAV/MP3 验证。长音频建议截取有代表性的片段再放到 Colab。 |
| 输出不符合预期 | 保存 transcript JSON 单元输出，提交 issue 时一起附上。 |

Notebook 源文件：[funasr_quickstart.ipynb](https://github.com/modelscope/FunASR/blob/main/examples/colab/funasr_quickstart.ipynb)。
