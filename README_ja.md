([English](./README.md)|[简体中文](./README_zh.md)|日本語|[한국어](./README_ko.md))

<p align="center">
<a href="https://github.com/modelscope/FunASR"><img src="https://svg-banners.vercel.app/api?type=origin&text1=FunASR🤠&text2=💖%20A%20Fundamental%20End-to-End%20Speech%20Recognition%20Toolkit&width=800&height=210" alt="FunASR"></a>
</p>

<p align="center">
  <strong>産業グレードの音声認識。最大340倍リアルタイム、Whisperより26倍高速。50以上の言語に対応。</strong><br>
  <em>話者分離 · 感情認識 · ストリーミング · ワンコールで完結</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/funasr/"><img src="https://img.shields.io/pypi/v/funasr" alt="PyPI"></a>
  <a href="https://github.com/modelscope/FunASR"><img src="https://img.shields.io/github/stars/modelscope/FunASR?style=social" alt="Stars"></a>
  <a href="https://pypi.org/project/funasr/"><img src="https://img.shields.io/pypi/dm/funasr" alt="Downloads"></a>
  <a href="https://modelscope.github.io/FunASR/"><img src="https://img.shields.io/badge/ドキュメント-オンライン-blue" alt="Docs"></a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/10479" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10479" alt="modelscope%2FFunASR | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="#クイックスタート">クイックスタート</a> · <a href="./examples/colab/README_ja.md">Colab</a> · <a href="./docs/model_selection_ja.md">モデル選択</a> · <a href="#ベンチマーク">ベンチマーク</a> · <a href="./docs/migration_from_whisper.md">Migration guide</a> · <a href="./docs/use_case_showcase.md">Use cases</a> · <a href="./docs/deployment_matrix_ja.md">Deployment matrix</a> · <a href="#モデル一覧">モデル一覧</a> · <a href="https://modelscope.github.io/FunASR/agent.html">Agent連携</a> · <a href="https://modelscope.github.io/FunASR/">ドキュメント</a>
</p>

---

<a name="クイックスタート"></a>

## クイックスタート

```bash
pip install funasr
```

```python
from funasr import AutoModel

model = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad", spk_model="cam++", device="cuda")
result = model.generate(input="meeting.wav")
```

**出力** — 話者ラベル・タイムスタンプ・句読点付きの構造化テキスト：
```
[00:00.4 → 00:03.8] 話者0: Q3の計画について話し合いましょう。
[00:04.2 → 00:07.1] 話者1: いいですね。3つのポイントがあります。
[00:07.5 → 00:12.3] 話者0: どうぞ。あと30分あります。
```

1つのモデル、1回の呼び出し — VADセグメンテーション、音声認識、句読点復元、話者分離がすべて自動で実行されます。

初めて使う場合は [Colab クイックスタート](./examples/colab/README_ja.md) から試せます。どのモデルを選ぶか迷う場合は [モデル選択ガイド](./docs/model_selection_ja.md) を参照してください。

> **APIサーバーとしてデプロイ：** `funasr-server --device cuda` → localhost:8000でOpenAI互換エンドポイント
>
> **AIエージェント連携：** [MCPサーバー](examples/mcp_server/) Claude/Cursor対応 · [OpenAI API](examples/openai_api/) LangChain/Dify/AutoGen対応

### なぜFunASRを選ぶのか？

| | FunASR | Whisper | クラウドAPI |
|---|---|---|---|
| 速度 | **170倍リアルタイム** | 13倍リアルタイム | 〜1倍リアルタイム |
| 話者認識 | ✅ 内蔵 | ❌ pyannoteが必要 | ✅ 追加料金 |
| 感情認識 | ✅ 喜び/悲しみ/怒り | ❌ | ❌ |
| 言語数 | 50以上 | 57 | サービスにより異なる |
| ストリーミング | ✅ WebSocket | ❌ | ✅ |
| セルフホスト | ✅ MITライセンス | ✅ MITライセンス | ❌ クラウドのみ |
| コスト | 無料 | 無料 | $0.006/分〜 |
| CPU対応 | ✅ 17倍リアルタイム | ❌ 遅すぎる | 該当なし |

---

<a name="ベンチマーク"></a>

## ベンチマーク

> 184件の長時間音声（計192分）。[詳細レポート →](https://modelscope.github.io/FunASR/benchmark.html)

| モデル | 中国語 CER ↓ | GPU速度 | CPU速度 | Whisper-large-v3比 |
|--------|------|---------|---------|-------------------|
| **Fun-ASR-Nano**（vLLM） | **8.20%** | **340倍**リアルタイム | — | 🚀 **26倍高速** |
| **SenseVoice-Small** | **7.81%** | **170倍**リアルタイム | **17倍**リアルタイム | 🚀 **13倍高速** |
| **Paraformer-Large** | 10.18% | **120倍**リアルタイム | **15倍**リアルタイム | 🚀 **9倍高速** |
| Whisper-large-v3-turbo | 21.71% | 46倍リアルタイム | ❌ | 3.4倍高速 |
| Whisper-large-v3 | 20.02% | 13倍リアルタイム | ❌ | ベースライン |

> **ポイント：** FunASRのCPU速度は、WhisperのGPU速度より速い。

---

## 最新情報

- 2026/05/24：**v1.3.3** — `funasr-server` CLI、OpenAI互換API、MCPサーバー。`pip install --upgrade funasr`
- 2026/05/20：Qwen3-ASR (0.6B/1.7B) 追加 — 52言語対応。
- 2026/05/20：GLM-ASR-Nano (1.5B) 追加 — 17言語、方言対応。
- 2025/12/15：[Fun-ASR-Nano-2512](https://github.com/FunAudioLLM/Fun-ASR) — 31言語対応。

---

## インストール

```bash
pip install funasr
```

要件：Python ≥ 3.8、PyTorch ≥ 1.13、torchaudio

---

<a name="モデル一覧"></a>

## モデル一覧

| モデル | タスク | 言語 | パラメータ | リンク |
|--------|--------|------|-----------|--------|
| **Fun-ASR-Nano** | 認識 + タイムスタンプ | 31言語 | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) |
| **SenseVoiceSmall** | 認識 + 感情 + イベント | 中/英/日/韓/粤 | 234M | [⭐](https://www.modelscope.cn/models/iic/SenseVoiceSmall) [🤗](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) |
| **Paraformer-zh** | 認識 + タイムスタンプ | 中/英 | 220M | [⭐](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) [🤗](https://huggingface.co/funasr/paraformer-zh) |
| Qwen3-ASR | 認識、52言語 | 多言語 | 1.7B | [使用法](examples/industrial_data_pretraining/qwen3_asr) |
| GLM-ASR-Nano | 認識、17言語 | 多言語 | 1.5B | [使用法](examples/industrial_data_pretraining/glm_asr) |
| Whisper-large-v3-turbo | 認識 + 翻訳 | 多言語 | 809M | [使用法](examples/industrial_data_pretraining/whisper) |

---

## デプロイ

```bash
# OpenAI互換API（推奨）
pip install funasr fastapi uvicorn python-multipart
funasr-server --device cuda

# Dockerストリーミングサービス
docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.12
```

[Colab quickstart →](./examples/colab/README_ja.md) · [OpenAI API example →](./examples/openai_api/README_ja.md) · [Client recipes →](./examples/openai_api/CLIENTS.md) · [Workflow recipes →](./examples/openai_api/WORKFLOWS.md) · [Postman collection →](./examples/openai_api/POSTMAN.md) · [OpenAPI spec →](./examples/openai_api/OPENAPI.md) · [Security guide →](./examples/openai_api/SECURITY.md) · [Deployment matrix →](./docs/deployment_matrix_ja.md) · [デプロイドキュメント →](./runtime/readme.md) · [Agent連携 →](https://modelscope.github.io/FunASR/agent.html)

---

## コミュニティ

|  |  |
|---|---|
| 📖 [ドキュメント](https://modelscope.github.io/FunASR/) | 🐛 [Issues](https://github.com/modelscope/FunASR/issues) |
| 💬 [Discussions](https://github.com/modelscope/FunASR/discussions) | 🤗 [HuggingFace](https://huggingface.co/funasr) |

## ライセンス

[MIT License](./LICENSE)
