([English](README.md)|[简体中文](README_zh.md)|日本語|[한국어](README_ko.md))

# FunASR OpenAI 互換 API サーバー

FunASR OpenAI 互換 API は `/v1/audio/transcriptions` を提供します。OpenAI スタイルの SDK、エージェントフレームワーク、Dify、n8n、HTTP ノード、社内システムから、プライベートな音声認識サービスとして利用できます。

## クイックスタート

```bash
pip install funasr fastapi uvicorn python-multipart
python server.py --model sensevoice --device cuda --port 8000
```

モデルのロード後にサービスが起動します。ヘルスチェックは `GET /health` です。

コピーして使える連携例が必要な場合は、[クライアントレシピ](CLIENTS.md)、[JavaScript/TypeScript レシピ](JAVASCRIPT.md)、[Gradio ブラウザデモ](GRADIO.md)、[ワークフローレシピ](WORKFLOWS.md)、[Postman コレクション](POSTMAN.md)、[OpenAPI 仕様](OPENAPI.md)、[セキュリティとゲートウェイガイド](SECURITY.md)、[Kubernetes デプロイテンプレート](kubernetes/README.md)を参照してください。

### エンドツーエンド smoke test

別のターミナルで実行します。

```bash
bash smoke_test.sh
# curl/bash を使わないクロスプラットフォーム版:
python smoke_test.py
```

同等の手動コマンド:

```bash
curl -L https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav
curl http://localhost:8000/health
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

## Gradio ブラウザデモ

ローカルブラウザで音声ファイルのアップロードやマイク入力を試したい場合は、先に API サーバーを起動し、オプションの Gradio フロントエンドを起動します。

```bash
pip install gradio
python gradio_app.py --base-url http://localhost:8000
```

このブラウザデモは smoke test と同じ OpenAI 互換 API エンドポイントを呼び出します。Docker、Kubernetes、本番利用の注意点は [Gradio ブラウザデモ](GRADIO.md)を参照してください。

## OpenAI SDK で使う

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

result = client.audio.transcriptions.create(
    model="sensevoice",  # "paraformer", "paraformer-en", "fun-asr-nano" も利用できます
    file=open("meeting.wav", "rb"),
)
print(result.text)

verbose = client.audio.transcriptions.create(
    model="sensevoice",
    file=open("meeting.wav", "rb"),
    response_format="verbose_json",
)
print(verbose.segments)
```

## curl で使う

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=sensevoice

curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

## 利用できるモデル

| Model | GPU 速度 | CPU 速度 | 言語 | 特徴 |
|---|---|---|---|---|
| `sensevoice` | 170x realtime | 17x realtime | zh/en/ja/ko/yue | 感情・イベントタグ |
| `paraformer` | 120x realtime | 15x realtime | zh/en | 句読点復元 |
| `paraformer-en` | 120x realtime | 15x realtime | en | 英語認識 |
| `fun-asr-nano` | 17x realtime | 3.6x realtime | 中英日 + 中国語方言・地域アクセント | LLM-based、タイムスタンプ |

## API エンドポイント

| Endpoint | Method | 説明 |
|---|---|---|
| `/v1/audio/transcriptions` | POST | OpenAI 互換の音声文字起こし |
| `/v1/models` | GET | モデルエイリアスの一覧 |
| `/health` | GET | ヘルスチェック、ロード済みモデル、利用可能モデル |
| `/docs` | GET | FastAPI Swagger ドキュメント |

コードを書かずに確認したい場合は、[Gradio ブラウザデモ](GRADIO.md)でローカルアップロードやマイク入力を試すか、[Postman コレクション](POSTMAN.md)をインポートしてください。API ゲートウェイ、開発者ポータル、社内クライアント生成には [OpenAPI 仕様](OPENAPI.md)を利用できます。

## エージェントとローコードワークフロー

**LangChain**、**LlamaIndex**、**AutoGen**、**CrewAI**、**Semantic Kernel**、**Dify**、**n8n**、OpenAI audio API または multipart HTTP を使える任意のシステムで利用できます。

- SDK、JavaScript/TypeScript、Agent tool の書き方は [クライアントレシピ](CLIENTS.md) と [JavaScript/TypeScript レシピ](JAVASCRIPT.md)を参照してください。
- Dify、n8n、HTTP ノード、webhook worker は [ワークフローレシピ](WORKFLOWS.md)を参照してください。
- GUI smoke test は [Postman コレクション](POSTMAN.md)を参照してください。
- schema-driven import には [OpenAPI 仕様](OPENAPI.md)を使えます。

## Docker デプロイ

デフォルトのイメージは CPU モードで起動し、再現しやすい smoke test として使えます。

```bash
cd examples/openai_api
cp .env.example .env

docker compose up --build
```

同等の `docker run`:

```bash
docker build -t funasr-api .

docker run --rm -p 8000:8000 \
  -e FUNASR_DEVICE=cpu \
  -e FUNASR_MODEL=sensevoice \
  funasr-api
```

GPU ホストでは NVIDIA Container Toolkit と CUDA 対応の PyTorch/FunASR イメージが必要です。CUDA 依存関係に合わせてイメージを調整した後、次のように起動できます。

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e FUNASR_DEVICE=cuda \
  -e FUNASR_MODEL=sensevoice \
  funasr-api
```

コンテナの検証:

```bash
BASE_URL=http://localhost:8000 bash smoke_test.sh
python smoke_test.py --base-url http://localhost:8000
```

## Kubernetes デプロイ

チーム内で共有したりゲートウェイ経由で公開したりする前に、[セキュリティとゲートウェイガイド](SECURITY.md)を確認し、TLS、認証、アップロード制限、レート制限、ログ方針を整えてください。

永続化されたモデルキャッシュ、ヘルスプローブ、プライベート `ClusterIP` を持つ内部クラスタサービスが必要な場合は、[Kubernetes デプロイテンプレート](kubernetes/README.md)から始めてください。サンプルイメージをビルドして push し、manifests を適用した後、`kubectl port-forward` と `python smoke_test.py --base-url http://localhost:8000` で検証します。

CUDA 対応イメージと GPU スケジューリング設定が整うまでは、デフォルトの CPU モードを維持してください。

## 設定

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--host` | `0.0.0.0` | バインドアドレス |
| `--port` | `8000` | ポート |
| `--device` | `cuda` | `cuda`、`cpu`、`mps` |
| `--model` | `sensevoice` | 起動時にプリロードするモデル |

Docker 環境変数:

| Env | デフォルト | 説明 |
|---|---|---|
| `FUNASR_PORT` | `8000` | `server.py` に渡すコンテナポート |
| `FUNASR_DEVICE` | `cpu` | コンテナのデバイスモード。CUDA 対応依存関係を持つイメージでのみ `cuda` に設定してください |
| `FUNASR_MODEL` | `sensevoice` | コンテナ起動時にロードするモデルエイリアス |

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| CUDA が利用できない | まず `--device cpu` で smoke test を通します。 |
| 8000 ポートが使用中 | `--port 9000` に変更し、`BASE_URL=http://localhost:9000 bash smoke_test.sh` または `python smoke_test.py --base-url http://localhost:9000` を実行します。 |
| モデルのダウンロードが遅い | 安定したネットワークで再試行するか、ModelScope/Hugging Face から事前にモデルをダウンロードします。 |
| Dify/n8n コンテナから `localhost` に接続できない | ワークフロー実行環境から到達できるホスト名、Compose service name、または Kubernetes service name を使います。 |
| 応答に `segments` がない | `response_format=verbose_json` を設定します。 |
