# FunASR デプロイ選択マトリクス

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
| vLLM acceleration | Fun-ASR-Nano の LLM-based ASR throughput 向上 | [vLLM guide](./vllm_guide.md) | LLM decoder throughput 向け。non-autoregressive Paraformer には適用しません。 |
| MCP server | Claude/Cursor/desktop agent の speech tool | [MCP example](../examples/mcp_server/) | ASR 結果を local tool として Agent に渡したい場合に便利です。 |
| Subtitle generator | 長時間 audio/video から SRT/VTT 作成 | [Subtitle example](../examples/subtitle/) | readability が重要な場合は verbose segment と speaker label を使います。 |
| Batch ASR script | Archive、meeting、dataset、繰り返し offline run | [Batch example](../examples/batch_asr_improved.py) | production では queue、manifest、retry log を追加してください。 |

## よくある選択

### 5分で FunASR を試したい

ブラウザだけで試すなら [Colab クイックスタート](../examples/colab/README_ja.md) を使います。ローカルで作業する場合は README の Python API から始めます。どのモデルを使うか迷う場合は [モデル選択ガイド](./model_selection_ja.md) を参照してください。

### Cloud transcription の local replacement が欲しい

OpenAI 互換 API を使います。`/v1/audio/transcriptions`、`/v1/models`、`/health`、Swagger docs を提供します。まず `sensevoice` で smoke test を実行し、既存 SDK や HTTP client を [OpenAI API example](../examples/openai_api/README_ja.md) に合わせて接続してください。

### 再現可能な container demo が欲しい

`examples/openai_api/docker-compose.yml` を CPU mode の smoke test として使います。

```bash
cd examples/openai_api
cp .env.example .env
docker compose up --build
```

CUDA を使う場合は CUDA-capable PyTorch/FunASR image を作成してから `FUNASR_DEVICE=cuda` に変更し、同じ smoke test で確認します。

### Streaming または live captioning が必要

Runtime WebSocket service を使います。本番投入前に chunk size、VAD、endpointing、punctuation、speaker diarization、reconnect、client backpressure を実音声で検証してください。

## Readiness checklist

- model alias を決め、deployment note に固定します。
- FunASR version、model version、device、CUDA/PyTorch version、Docker image tag、command line を記録します。
- public smoke sample と realistic private sample を少なくとも 1 つずつ実行します。
- request ごとに audio duration、model、device、latency、response format、error type をログ化します。
- trusted network の外へ API を出す前に upload-size limit、authentication、TLS、rate limit を入れます。[Security guide](../examples/openai_api/SECURITY.md) も確認してください。
- 詰まったら deployment path、command/config、logs、model、device、audio characteristics を添えて [Deployment Help issue](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md) を開いてください。
