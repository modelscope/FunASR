# FunASR デプロイ選択マトリクス

## Transformers で Nano を試す

中国語・英語・日本語の文字起こしには [Transformers 5.17.0 ガイド（英語）](./transformers_native.md) と公式 `FunAudioLLM/Fun-ASR-Nano-2512-hf` checkpoint を使えます。CPU サンプルがあり、toolkit とサービスの経路は別です。ネイティブ出力はテキストで、タイムスタンプ・話者・HTTP サーバーを追加しません。

プロダクト、デモ、ベンチマーク、社内ワークフローに合わせて最短のデプロイ経路を選ぶためのガイドです。まずは要件を満たす最小構成から始め、throughput、latency、integration 要件が明確になったら重い runtime に移行してください。

## クイック判断表

| Path | 向いている用途 | 最初に読むもの | 運用メモ |
|---|---|---|---|
| Colab notebook | ブラウザ smoke test、初回評価、共有 demo | [Colab クイックスタート](../examples/colab/README_ja.md) | ローカル環境不要。初回はモデルをダウンロードし、GPU runtime の方が高速です。 |
| Python API | Notebook、offline job、最初の model evaluation | [README quick start](../README_ja.md#クイックスタート) | 最小構成。batching、retry、file 管理は呼び出し側で扱います。 |
| OpenAI 互換 API | Private speech API、Agent、Dify/LangChain/AutoGen style clients | [OpenAI API example](../examples/openai_api/README_ja.md) | OpenAI audio API に対応した既存 app に最も接続しやすい経路です。 |
| Docker Compose API | 再現可能な local smoke test、小さな internal service | [OpenAI API Docker docs](../examples/openai_api/README_ja.md#docker-デプロイ) | デフォルトは CPU。CUDA を使う前に CUDA-capable image へ調整してください。 |
| Kubernetes API | Cluster service 向け internal speech API | [Kubernetes template](../examples/openai_api/kubernetes/) | private `ClusterIP` から開始。公開範囲を広げる前に auth、TLS、network policy、GPU scheduling を追加します。 |
| Runtime WebSocket service | Live captions、meeting、call-center stream | [Runtime service docs](../runtime/readme.md) | partial result、endpointing、long-lived audio stream が重要な場合に使います。 |
| vLLM acceleration | Fun-ASR-Nano の native ファイル文字起こし、または split-engine デコード | [公式 native 検証（英語）](./vllm_official_native_validation.md)、[split-engine guide](./vllm_guide.md) | 2 つの経路は checkpoint と API が異なります。非自己回帰の Paraformer には適用しません。 |
| MOSS-Transcribe-Diarize | 長時間の複数話者 transcription、timestamp、speaker label | [Third-party MOSS guide](./moss_transcribe_diarize.md) | OpenMOSS の Apache-2.0 model を FunASR `AutoModel` に統合済みです。local HF（`backend="hf"`）、vLLM（`backend="vllm"`）、または SGLang Omni（`backend="sglang"`）を選択できます。model の公開・保守主体は OpenMOSS のままです。 |
| MCP server | Claude/Cursor/desktop agent の speech tool | [MCP example](../examples/mcp_server/) | ASR 結果を local tool として Agent に渡したい場合に便利です。 |
| Subtitle generator | 長時間 audio/video から SRT/VTT 作成 | [Subtitle example](../examples/subtitle/) | readability が重要な場合は verbose segment と speaker label を使います。 |
| Batch ASR script | Archive、meeting、dataset、繰り返し offline run | [Batch example](../examples/batch_asr_improved.py) | production では queue、manifest、retry log を追加してください。 |

## よくある選択

### Fun-ASR-Nano を vLLM で動かしたい

- **Split-engine**: 基本モデル `FunAudioLLM/Fun-ASR-Nano-2512` を使い、FunASR 側で音声を処理して LLM decoder を vLLM で実行します。[split-engine guide](./vllm_guide.md)を参照してください。
- **Native**: vLLM 自身が音声モデルを実行します。公式 checkpoint は `FunAudioLLM/Fun-ASR-Nano-2512-vllm` です。基本モデル用の設定や checkpoint と相互に置き換えないでください。

[公式 native 検証記録（英語）](./vllm_official_native_validation.md)は、モデル revision
`a4362c943d48951f98ca2a62181cc028970270c5`、vLLM 0.27.1、既存の H100 環境での機能確認です。
モデル revision は FunASR パッケージのバージョンではありません。依存関係を含む準備・起動手順は検証記録を参照してください。

確認した API はファイル入力の `/v1/audio/transcriptions` です。`/v1/realtime`、クリーンインストール、長時間音声、話者分離、本番の処理容量はこの検証の対象ではありません。FunASR SDK や別の checkpoint の検証として扱わず、実運用の音声・負荷で別途評価してください。

### 5分で FunASR を試したい

ブラウザだけで試すなら [Colab クイックスタート](../examples/colab/README_ja.md) を使います。ローカルで作業する場合は README の Python API から始めます。どのモデルを使うか迷う場合は [モデル選択ガイド](./model_selection_ja.md) を参照してください。

### Cloud transcription の local replacement が欲しい

OpenAI 互換 API を使います。主な入口は次のとおりです。

- `/v1/audio/transcriptions`: ファイル文字起こし
- `/v1/models`: モデル一覧
- `/health`: ヘルスチェック
- Swagger docs: API の確認

まず `sensevoice` で smoke test を実行し、既存 SDK や HTTP client を [OpenAI API example](../examples/openai_api/README_ja.md) に合わせて接続してください。

### 再現可能な container demo が欲しい

Docker Engine と Docker Compose plugin を用意し、**FunASR リポジトリのルート**からローカルの SenseVoice CPU service を起動します。このコマンドは既存の `.env` を上書きしません。明示した port、device、model は、この起動に限り環境から継承した値より優先されます。ホスト側は loopback のみで待ち受けますが、認証 gateway ではありません。共有前に [security guide（英語）](../examples/openai_api/SECURITY.md) を確認してください。

```bash
FUNASR_HOST_PORT=127.0.0.1:8000 FUNASR_DEVICE=cpu FUNASR_MODEL=sensevoice \
  docker compose -f examples/openai_api/docker-compose.yml up --build
```

Compose はそのターミナルで実行したままにします。**別のターミナルを開き、同じリポジトリのルート**で Python 3.10 以降を使って確認します。smoke client は標準ライブラリのみを使うため、ホストへの FunASR のインストールは不要です。

```bash
python3 examples/openai_api/smoke_test.py --base-url http://127.0.0.1:8000 --model sensevoice --response-format verbose_json
```

現在のディレクトリに `sample.wav` がなければ公開の中国語音声をダウンロードし、既存ファイルがあれば再利用します。health、model metadata、文字起こし JSON が表示されます。テキストは自分で確認してください。終了コードの成功だけでは認識精度や同時実行性能は検証できません。client は Authorization を送信せず、security guide の文字起こし専用 gateway はこの smoke が使う metadata route を拒否します。ローカル endpoint で実行し、機密音声や未加工の出力を残さないでください。

CPU image は example server をコピーしますが、FunASR は PyPI からインストールします。依存バージョンは固定されていません。再現性を主張する前に実際の package version と image digest を記録してください。`FUNASR_DEVICE` の変更だけでは CUDA 依存や container の GPU access は追加されません。GPU image と scheduling は別途用意して検証し、[HTTP deployment guide（英語）](../examples/openai_api/README.md#docker-deployment) とモデルの要件を確認してください。

### Streaming または live captioning が必要

Runtime WebSocket service を使います。本番投入前に chunk size、VAD、endpointing、punctuation、speaker diarization、reconnect、client backpressure を実音声で検証してください。

## Readiness checklist

- model alias を決め、deployment note に固定します。
- FunASR version、model version、device、CUDA/PyTorch version、Docker image tag、command line を記録します。
- public smoke sample と realistic private sample を少なくとも 1 つずつ実行します。
- request ごとに audio duration、model、device、latency、response format、error type をログ化します。
- trusted network の外へ API を出す前に upload-size limit、authentication、TLS、rate limit を入れます。[Security guide](../examples/openai_api/SECURITY.md) も確認してください。
- 詰まったら deployment path、command/config、logs、model、device、audio characteristics を添えて [Deployment Help issue](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md) を開いてください。
