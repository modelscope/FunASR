# FunASR モデル選択ガイド

初めて FunASR を試すとき、Whisper やクラウド ASR から移行するとき、または OpenAI 互換 API で公開するモデル alias を決めるときに使ってください。

## 迷ったらここから

まずは **SenseVoice-Small** から始めるのがおすすめです。

```python
from funasr import AutoModel

model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    spk_model="cam++",
    device="cuda",  # 手元の smoke test では "cpu" でも可
)
result = model.generate(input="meeting.wav")
```

デモ、プライベート API、多言語文字起こし、話者付き会議録、Agent 音声入力の最初の選択肢として使いやすいモデルです。中国語本番精度、ストリーミング遅延、LLM-based ASR 評価など明確な要件が出たときだけ切り替えてください。

## 判断表

| やりたいこと | 最初に試すもの | 理由 | 次に読むもの |
|---|---|---|---|
| 高速な多言語プライベート文字起こし | SenseVoice-Small | ASR、感情タグ、音声イベントタグ、CPU/GPU の扱いやすさがそろった標準ルート。 | [README quick start](../README_ja.md#クイックスタート) |
| 中国語中心の本番 ASR | Paraformer-Large | VAD と句読点復元を組み合わせた成熟した中国語 ASR ルート。 | [Tutorial](./tutorial/README.md) |
| OpenAI API 例の英語ルート | `paraformer-en` alias | OpenAI-style client で互換性を確認しやすい軽量な英語ルート。 | [OpenAI API example](../examples/openai_api/README_ja.md) |
| LLM-based ASR や中英日 + 中国語方言・地域アクセントの評価 | Fun-ASR-Nano | LLM-based モデル。decoder throughput が重要なら vLLM を使います。 | [vLLM guide](./vllm_guide.md) |
| ライブ字幕やコールセンターストリーム | Runtime WebSocket service | 長時間接続、部分結果、エンドポイント検出に向いたランタイム。 | [Runtime service docs](../runtime/readme.md) |
| Whisper / cloud ASR からの移行 | SenseVoice-Small で baseline を作り、必要に応じて比較 | まず強い標準ルートで評価してから、用途別に詰めるのが安全です。 | [Migration guide](./migration_from_whisper.md) |

## OpenAI 互換 API alias

`examples/openai_api` server は短い alias を提供します。アプリケーション側はモデル repository ID を知らなくても利用できます。

| Alias | 中身 | 使う場面 |
|---|---|---|
| `sensevoice` | `iic/SenseVoiceSmall` | 多言語 ASR、イベントタグ、CPU/GPU 両対応の標準プライベート音声 API。 |
| `paraformer` | `paraformer-zh` + VAD + punctuation | 中国語中心の本番ルート。 |
| `paraformer-en` | `paraformer-en` + VAD | OpenAI-style client の英語互換性チェック。 |
| `fun-asr-nano` | `FunAudioLLM/Fun-ASR-Nano-2512` | LLM-based ASR の中英日・中国語方言/地域アクセント評価、または vLLM acceleration の確認。 |

接続前にサービスを確認します。

```bash
curl http://localhost:8000/v1/models
python examples/openai_api/smoke_test.py --base-url http://localhost:8000 --model sensevoice
```

SDK、JavaScript、workflow、Postman、OpenAPI、Docker、Kubernetes は [OpenAI API example](../examples/openai_api/README_ja.md) から始めてください。

## ベンチマークしてから決める

きれいな demo 音声 1 つだけでモデルを決めないでください。まず小さな代表セットで確認します。

- 短いクリップ、長い会議、無音、ノイズ、話者重なり、専門用語、対象言語を含む 20-50 ファイルを用意します。
- model name、model revision、FunASR version、device、CPU/GPU、CUDA/PyTorch、runtime path、batch size、download/warmup の扱いを記録します。
- 読みやすさだけでなく、通常使う WER/CER または人手レビューで品質を見ます。
- latency、throughput、memory、failure、upload size limit をまとめて比較します。
- 困ったときは model、device、command、logs、audio duration、runtime path を添えて [Deployment Help issue](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md) を開いてください。
