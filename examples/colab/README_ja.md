# FunASR Colab クイックスタート

[English](README.md) | [简体中文](README_zh.md) | 日本語 | [한국어](README_ko.md)

ローカルの Python 環境を準備せずに、ブラウザだけで FunASR を実行できます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modelscope/FunASR/blob/main/examples/colab/funasr_quickstart.ipynb)

## Notebook で試せること

- Colab に FunASR と実行時依存関係をインストールします。
- Colab の GPU が使える場合は自動で `cuda:0` を選び、なければ CPU を使います。
- `paraformer-zh`、VAD、句読点モデルで公開サンプル音声を文字起こしします。
- 自分の音声ファイルをアップロードし、同じモデルで文字起こしします。
- transcript JSON を保存し、共有、比較、issue 報告に利用できます。

## 利用メモ

- 初回実行ではモデルファイルをダウンロードするため、数分かかる場合があります。
- CPU runtime は短い smoke test に使えます。長い音声では GPU runtime の方が高速です。
- 本番デプロイを評価する場合は、notebook が動いた後に [deployment matrix](../../docs/deployment_matrix.md) を確認してください。
- OpenAI 互換 HTTP サービスを試す場合は [examples/openai_api](../openai_api/README_ja.md) を利用してください。

Notebook source: [funasr_quickstart.ipynb](https://github.com/modelscope/FunASR/blob/main/examples/colab/funasr_quickstart.ipynb).
