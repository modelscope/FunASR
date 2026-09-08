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

デモ、プライベート API、多言語文字起こし、Agent 音声入力の評価をここから始められます。上の会議録の例では、SenseVoice の ASR・感情/イベントタグとは別に、`fsmn-vad` で音声区間を検出し、`cam++` の話者埋め込みをクラスタリングします。話者ラベルは録音内の匿名番号で、登録済み人物の識別や録音間で固定された ID ではありません。対象言語と実際の音声でモデルを比較してください。

**Fun-ASR-Nano-2512** は中国語・英語・日本語と中国語方言/地域アクセントの評価候補です。**Fun-ASR-MLT-Nano** は別の checkpoint です。必要な言語の対応を各モデルカードで確認し、Nano の対応範囲と混同しないでください。

## 判断表

| やりたいこと | 最初に試すもの | 理由 | 次に読むもの |
|---|---|---|---|
| 高速な多言語プライベート文字起こし | SenseVoice-Small | ASR、感情タグ、音声イベントタグ、CPU/GPU の扱いやすさがそろった標準ルート。 | [README quick start](../README_ja.md#クイックスタート) |
| 中国語中心の本番 ASR | Paraformer-Large | VAD と句読点復元を組み合わせた成熟した中国語 ASR ルート。 | [Tutorial](./tutorial/README.md) |
| OpenAI API 例の英語ルート | `paraformer-en` alias | OpenAI-style client で互換性を確認しやすい軽量な英語ルート。 | [OpenAI API example](../examples/openai_api/README_ja.md) |
| LLM-based ASR や中英日 + 中国語方言・地域アクセントの評価 | Fun-ASR-Nano | Python で評価してから、checkpoint と interface に合わせて vLLM の経路を選びます。 | [vLLM の経路](#vllm-checkpoint-paths) |
| オフライン長時間 ASR と匿名話者ラベル | MOSS-Transcribe-Diarize | 1 回のオフライン request で文字起こし、timestamps、録音内の匿名話者ラベルを返します。既知人物の識別ではなく、外部 VAD / speaker model も不要です。 | [MOSS deployment guide](./moss_transcribe_diarize.md) |
| ライブ字幕やコールセンターストリーム | Runtime WebSocket service | 長時間接続、部分結果、エンドポイント検出に向いたランタイム。 | [Runtime service docs](../runtime/readme.md) |
| Whisper / cloud ASR からの移行 | SenseVoice-Small で baseline を作り、必要に応じて比較 | まず強い標準ルートで評価してから、用途別に詰めるのが安全です。 | [Migration guide](./migration_from_whisper.md) |

## OpenAI 互換 API alias

`examples/openai_api` server は短い alias を提供します。アプリケーション側はモデル repository ID を知らなくても利用できます。

- **`sensevoice`**: `iic/SenseVoiceSmall` による CPU/GPU での多言語 HTTP 文字起こしです。返却テキストからリッチタグは除去されます。
- **`paraformer`**: `paraformer-zh` に VAD と句読点復元を組み合わせた中国語向けの経路です。
- **`paraformer-en`**: `paraformer-en` と VAD を使う、OpenAI-style client 向けの英語文字起こしです。
- **`fun-asr-nano`**: `FunAudioLLM/Fun-ASR-Nano-2512` による中英日・中国語方言/地域アクセントの評価経路です。vLLM acceleration を試す場合は互換性のある runtime を選んでください。
- **`moss-transcribe-diarize`**: 第三者の `OpenMOSS-Team/MOSS-Transcribe-Diarize` によるオフライン文字起こしと録音内の匿名話者ラベルです。[MOSS guide（英語）](./moss_transcribe_diarize.md)で専用の依存関係と remote code を確認し、構造化 segment には `verbose_json` を指定します。外部 VAD / speaker model は不要で、既知人物の識別ではありません。

ここでの alias は `AutoModel` を読み込む[サンプル server](../examples/openai_api/server.py)のものです。
native vLLM や `AutoModelVLLM` を自動選択する設定ではありません。
パッケージの `funasr-server` は別の loader / backend 選択を持つため、
サービス間で alias や性能結果をそのまま流用しないでください。

この HTTP サンプルはトップレベルの `text` と `verbose_json` の各 segment の
`text` を整形するため、形式を変えても感情/イベントタグは復元されません。
元のタグが必要なら Python SDK を使い、表示用の後処理より前に返却された `text`
を保存してください。[元のタグを保存するレシピ（英語）](./speaker_emotion.md)を参照してください。

接続前にサービスを確認します。

```bash
curl http://localhost:8000/v1/models
python examples/openai_api/smoke_test.py --base-url http://localhost:8000 --model sensevoice
```

SDK、JavaScript、workflow、Postman、OpenAPI、Docker、Kubernetes は [OpenAI API example](../examples/openai_api/README_ja.md) から始めてください。

<a id="vllm-checkpoint-paths"></a>

## vLLM の checkpoint と interface

| 経路 | checkpoint と interface | 次の資料 |
| --- | --- | --- |
| FunASR split-engine | 基本の `FunAudioLLM/Fun-ASR-Nano-2512` を `AutoModelVLLM` で読み込み、音声側は FunASR、decoder は vLLM が処理します。 | [Split-engine（英語）](./vllm_guide.md) |
| 公式 native vLLM | 変換済み `FunAudioLLM/Fun-ASR-Nano-2512-vllm` を vLLM の native 実装で読み込み、`/v1/audio/transcriptions` を利用します。`AutoModelVLLM` ではありません。 | [公式機能検証（英語）](./vllm_official_native_validation.md) |
| 過去の community native vLLM | `allendou/Fun-ASR-Nano-2512-vllm`、2026-08-13 の検証です。時間計測は当時の checkpoint と環境に限定されます。 | [過去の community 記録](./vllm_native_funasr_validation.md) |

公式記録は固定 revision と既存環境での機能検証であり、新規インストール手順、
持続負荷の benchmark、`/v1/realtime` の streaming 検証ではありません。
過去の community 計測を公式モデルに付け替えないでください。
MOSS は専用ガイドに従ってください。Nano の checkpoint と検証は MOSS の互換性を証明しません。

## ベンチマークしてから決める

きれいな demo 音声 1 つだけでモデルを決めないでください。まず小さな代表セットで確認します。

- 短いクリップ、長い会議、無音、ノイズ、話者重なり、専門用語、対象言語を含む 20-50 ファイルを用意します。
- model name、model revision、FunASR version、device、CPU/GPU、CUDA/PyTorch、runtime path、batch size、download/warmup の扱いを記録します。
- 読みやすさだけでなく、通常使う WER/CER または人手レビューで品質を見ます。
- latency、throughput、memory、failure、upload size limit をまとめて比較します。
- 困ったときは model、device、command、logs、audio duration、runtime path を添えて [Deployment Help issue](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md) を開いてください。
