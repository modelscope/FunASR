([English](./README.md)|[简体中文](./README_zh.md)|日本語|[한국어](./README_ko.md))

<p align="center">
<a href="https://github.com/modelscope/FunASR"><img src="https://svg-banners.vercel.app/api?type=origin&text1=FunASR🤠&text2=💖%20A%20Fundamental%20End-to-End%20Speech%20Recognition%20Toolkit&width=800&height=210" alt="FunASR"></a>
</p>

<p align="center">
  <strong>オフライン、ストリーミング、エッジ展開に対応する産業グレードの音声認識ツールキット。</strong><br>
  <em>ASR · VAD · 句読点 · 話者パイプライン · 感情/音声イベントモデル · OpenAI互換配信</em>
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
  <a href="#クイックスタート">クイックスタート</a> · <a href="./docs/model_selection_ja.md">モデル選択</a> · <a href="#モデル一覧">モデル一覧</a> · <a href="./docs/deployment_matrix_ja.md">デプロイ方式</a> · <a href="https://www.funasr.com/en/">デプロイセンター</a> · <a href="https://www.funasr.com/en/docs/">ドキュメント</a> · <a href="#ベンチマーク">ベンチマーク</a>
</p>

---

<a name="クイックスタート"></a>

## クイックスタート

### ネイティブ Transformers

Hugging Face API で Fun-ASR-Nano を使う場合は [Transformers 5.17.0 CPU ガイド（英語）](./docs/transformers_native.md) から開始できます。FunASR toolkit とリモート Python コードは不要です。

[Space](https://huggingface.co/spaces/FunAudioLLM/Fun-ASR-Nano) · [Notebook](https://colab.research.google.com/github/QwenAudio/Fun-ASR/blob/main/examples/colab/fun_asr_nano_transformers.ipynb) · [Python / batch examples](https://github.com/QwenAudio/Fun-ASR/tree/main/examples/transformers)

### FunASR toolkit とパイプライン

```bash
python -m pip install torch torchaudio
python -m pip install funasr
```

以下は公開サンプルを使う CPU 向けの例です。GPU を使う場合は
[インストールガイド](./docs/installation/installation.md) に従って互換性のある
PyTorch/CUDA 環境を用意し、`torch.cuda.is_available()` を確認してから
`device="cuda"` に変更してください。

```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad", spk_model="cam++", device="cpu")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")

for seg in result[0]["sentence_info"]:
    print(f"[{seg['start']/1000:.1f}s] 話者{seg['spk']}: {rich_transcription_postprocess(seg['sentence'])}")
```

実際に返された VAD 区間の開始時刻（秒）、匿名話者番号、SenseVoice タグを
除いたテキストを表示します。テキストや区間は音声と checkpoint に依存し、
ここでは固定の認識結果を示していません。

CAM++ は `spk_embedding` ベクトルを抽出し、`AutoModel` がクラスタリングと
VAD 区間への話者割り当てを行います。番号は録音内だけで有効で、既知の人物の
識別や SenseVoiceSmall 単体の出力ではありません。
詳細は [SDK 契約](./docs/python_api.md) を参照してください。

初めて使う場合は [Colab クイックスタート](./examples/colab/README_ja.md) から試せます。どのモデルを選ぶか迷う場合は [モデル選択ガイド](./docs/model_selection_ja.md) を参照してください。

> **APIサーバーとしてデプロイ：** [ローカル SenseVoice CPU 手順](#デプロイ) · [Nano GPU 配信と固定版 vLLM 環境](./docs/vllm_guide.md)
>
> **AIエージェント連携：** [MCPサーバー](examples/mcp_server/) Claude/Cursor対応 · [OpenAI API](examples/openai_api/) LangChain/Dify/AutoGen対応

### なぜFunASRを選ぶのか？

FunASR はツールキットです。タスク、checkpoint、ランタイムを別々に選びます。
あるモデルやアダプターの機能が、すべての配信バックエンドで使えるとは限りません。

| タスク | Checkpoint またはパイプライン | ランタイムの入口 | 主な制約 |
|---|---|---|---|
| ファイル認識と感情・イベントタグ | SenseVoiceSmall | Python `AutoModel`、CPU または GPU | 5言語の checkpoint。タグは話者の身元を示しません。 |
| LLM によるファイル認識 | Fun-ASR-Nano | `AutoModel`、または文書化された GPU 分離エンジン `AutoModelVLLM` | 基本 Nano は中・英・日と中国語方言・アクセント。timestamp は checkpoint と経路に依存します。 |
| より多くの言語のファイル認識 | Fun-ASR-MLT-Nano | Python `AutoModel` | 独立した31言語 checkpoint。その対応言語を基本 Nano に当てはめないでください。 |
| チャンク単位のライブ認識 | Paraformer-zh-streaming | ストリーミング SDK または runtime WebSocket | ストリーミング checkpoint とセッション別 cache が必要です。 |
| 話者付きファイル認識 | SenseVoiceSmall + FSMN-VAD + CAM++ | `AutoModel` の VAD と埋め込みクラスタリング | 番号は録音内の匿名ラベルで、登録済み人物の識別ではありません。 |
| テキスト・時刻・話者の同時生成 | 第三者 OpenMOSS の MOSS-Transcribe-Diarize | MOSS ガイドの FunASR adapter または上流バックエンド | オフライン、録音内の匿名ラベル。統合経路に外部 VAD/話者モデルは付けません。 |
| ネイティブ CPU/エッジ認識 | Fun-ASR-Nano または SenseVoiceSmall GGUF | llama.cpp runtime | 対応する変換済み重みが必要。GGUF は Python `AutoModel` 用 checkpoint ではありません。 |

[Model Zoo](./model_zoo/readme.md) と [デプロイ方式](./docs/deployment_matrix_ja.md)
でインターフェースとライセンスの制約を確認し、対象音声・ハードウェアで評価してください。

---

<a name="ベンチマーク"></a>

## ベンチマーク

[過去の評価レポート](https://modelscope.github.io/FunASR/benchmark.html) と
[分離エンジンの測定](./docs/vllm_guide.md#benchmark) に元の結果を残しています。
別々の記録であり、一般的な速度順位や本番容量を保証するものではありません。

[RTFx と再現性の説明](./docs/benchmark/rtf_reproducibility.md) に従い、
checkpoint/revision、音声セット、ハードウェア、バッチ、ウォームアップ、計測範囲、
CER/WER をそろえて比較してください。オフラインのスループットはストリーミングの
遅延ではありません。[移行評価の例](./examples/migration/) で自分の録音を評価できます。

---

## 最新情報

- **MOSS-Transcribe-Diarize** を FunASR service、Docker、Kubernetes、vLLM/SGLang workflow、FunClip に統合し、長時間 ASR、timestamp、匿名 speaker label を一度に処理できます。[MOSS をデプロイ ->](./docs/moss_transcribe_diarize.md)
- **FunASR 1.4.15** はテスト済みの NumPy 2 互換性を追加し、ストリーミング KWS/VAD の境界処理と checkpoint の順位付けを修正します。`python -m pip install -U "funasr==1.4.15"`。[リリースと検証範囲 ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.15)
- **ネイティブ Transformers：** 正式版 **5.17.0** が Fun-ASR-Nano に対応。公式 `-hf` checkpoint、CPU サンプル、Notebook：[導入ガイド（英語） ->](./docs/transformers_native.md)

> 完全な変更履歴と download asset は [GitHub Releases](https://github.com/modelscope/FunASR/releases) を参照してください。

---

## インストール

```bash
pip install funasr
```

要件：Python ≥ 3.8、PyTorch ≥ 1.13、torchaudio

---

<a name="モデル一覧"></a>

## モデル一覧

第三者モデルも含みます。MOSS-Transcribe-Diarize の公開元は **OpenMOSS** で、
FunASR はアダプターを提供します。統合経路はオフラインで、匿名話者ラベルは
録音内だけで有効です。リアルタイム処理や既知の人物の識別ではありません。
モデルのライセンスはツールキットの MIT ライセンスとは別に確認してください。

| モデル | タスク | 言語 | パラメータ | リンク |
|--------|--------|------|-----------|--------|
| **Fun-ASR-Nano** | 認識 | 中/英/日 + 中国語方言 | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) [GGUF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF) |
| **Fun-ASR-MLT-Nano** | 認識 | 31言語 | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) |
| **SenseVoiceSmall** | 認識 + 感情 + イベント | 中/英/日/韓/粤 | 234M | [⭐](https://www.modelscope.cn/models/iic/SenseVoiceSmall) [🤗](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) [GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF) |
| **MOSS-Transcribe-Diarize** | 第三者 OpenMOSS：オフライン認識 + タイムスタンプ + 匿名話者 | 公式モデルカードを参照 | 公式モデルカードを参照 | [🤗](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) [ガイド](./docs/moss_transcribe_diarize.md) |
| **Paraformer-zh** | 認識 + タイムスタンプ | 中/英 | 220M | [⭐](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) [🤗](https://huggingface.co/funasr/paraformer-zh) |
| Qwen3-ASR | 認識、52言語 | 多言語 | 1.7B | [使用法](examples/industrial_data_pretraining/qwen3_asr) |
| GLM-ASR-Nano | 認識、17言語 | 多言語 | 1.5B | [使用法](examples/industrial_data_pretraining/glm_asr) |
| Whisper-large-v3-turbo | 認識 + 翻訳 | 多言語 | 809M | [使用法](examples/industrial_data_pretraining/whisper) |

---

## デプロイ

新しいディレクトリで POSIX shell と Python 3.11 を使い、ローカルの SenseVoice
CPU サービスを起動します。独立環境に入るのは PyPI のリリースであり、現在の
ソース checkout ではありません。認証を内蔵しないため loopback で待ち受け、
他のクライアントへ公開する前に[セキュリティガイド](./examples/openai_api/SECURITY.md)を確認してください。

```bash
python3.11 -m venv .venv-funasr-http
. .venv-funasr-http/bin/activate
python -m pip install torch torchaudio
python -m pip install funasr fastapi uvicorn python-multipart
python -m pip check
funasr-server --host 127.0.0.1 --port 8000 --model sensevoice --device cpu
```

モデルのダウンロードと起動を待ちます。別のターミナルで同じディレクトリに入り、
curl 7.76+ で公開中国語サンプルを取得して認識します。リクエストは事前ロードした
モデルと一致しますが、固定テキストや話者ラベルを保証する例ではありません。

```bash
curl --fail --location https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav && \
curl --fail-with-body http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

オフライン ASR と匿名話者ラベルの同時生成（`moss-transcribe-diarize`）には、
[MOSS service / Docker / Kubernetes / vLLM / SGLang / LocalAI / FunClip guide →](./docs/moss_transcribe_diarize.md)
の独立環境を用意してください。CPU 環境で続けて起動するものではなく、代替サービスです。
8000 番ポートを再利用する前に CPU サービスを停止します。Nano GPU 配信は
[固定版の分離エンジンガイド](./docs/vllm_guide.md)に従い、実際の backend ログを確認してください。
モデルを指定しただけでは vLLM の使用を確認できません。

```bash
# Dockerストリーミングサービス
docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.12
```

CPU/エッジで Python なしのオフライン ASR が必要な場合は、llama.cpp / GGUF ランタイムを使えます：[funasr.com/deploy/llama-cpp](https://www.funasr.com/en/deploy/llama-cpp.html) · [Fun-ASR-Nano-GGUF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF) · [SenseVoiceSmall-GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF)。

**事前ビルド済みバイナリ：** [Releases](https://github.com/modelscope/FunASR/releases) · [v0.2.6](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.6) · [Linux Vulkan tarball](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-linux-x64-vulkan.tar.gz) · [Windows Vulkan zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-vulkan.zip) · [Windows CUDA zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-cuda.zip) · [Windows Blackwell CUDA zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-cuda-blackwell.zip) · **ダウンロードとクイックスタート：** [funasr.com/deploy/llama-cpp](https://www.funasr.com/en/deploy/llama-cpp.html) · **GGUF モデル：** [Hugging Face](https://huggingface.co/FunAudioLLM) · **ドキュメントとベンチマーク：** [runtime/llama.cpp/](./runtime/llama.cpp/)

Windows GPU では、[runtime-llamacpp-v0.2.6](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.6) の `windows-x64-vulkan`、`windows-x64-cuda` または `windows-x64-cuda-blackwell` パッケージを選択してください。RTX 50 / Blackwell (`sm_120`) には専用の `windows-x64-cuda-blackwell` パッケージがあります。CI のアーカイブ検証は実機での Blackwell 推論を保証しません。詳細は [llama.cpp 配布ガイド](https://www.funasr.com/en/deploy/llama-cpp.html) を参照してください。

[Colab quickstart →](./examples/colab/README_ja.md) · [OpenAI API example →](./examples/openai_api/README_ja.md) · [Client recipes →](./examples/openai_api/CLIENTS.md) · [Workflow recipes →](./examples/openai_api/WORKFLOWS.md) · [Postman collection →](./examples/openai_api/POSTMAN.md) · [OpenAPI spec →](./examples/openai_api/OPENAPI.md) · [Security guide →](./examples/openai_api/SECURITY.md) · [Deployment matrix →](./docs/deployment_matrix_ja.md) · [デプロイドキュメント →](./runtime/readme.md) · [Agent連携 →](https://modelscope.github.io/FunASR/agent.html)

---

## コミュニティ

|  |  |
|---|---|
| 📖 [ドキュメント](https://modelscope.github.io/FunASR/) | 🐛 [Issues](https://github.com/modelscope/FunASR/issues) |
| 💬 [Discussions](https://github.com/modelscope/FunASR/discussions) | 🤗 [HuggingFace](https://huggingface.co/funasr) |

## ライセンス

- このリポジトリの FunASR ツールキットのソースコード: [MIT License](./LICENSE)。
- 事前学習済みモデルの重みは個別にライセンスされます。各モデルカードに記載されたライセンスを確認してください。モデルカードがこのリポジトリの [FunASR Model Open Source License Agreement](./MODEL_LICENSE) を参照している場合、その条件が適用されます。
