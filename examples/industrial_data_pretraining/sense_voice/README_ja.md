# SenseVoice

「[简体中文](./README_zh.md)」|「[English](./README.md)」|「日本語」

SenseVoiceは、音声認識（ASR）、言語識別（LID）、音声感情認識（SER）、および音響イベント分類（AEC）または音響イベント検出（AED）を含む音声理解能力を備えた音声基盤モデルです。本プロジェクトでは、SenseVoiceモデルの紹介と、複数のタスクテストセットでのベンチマーク、およびモデルの体験に必要な環境のインストールと推論方法を提供します。

<div align="center">  
<img src="image/sensevoice2.png">
</div>
[//]: # (<div align="center"><img src="image/sensevoice2.png" width="700"/> </div>)
 
<div align="center">  
<h4>
<a href="https://funaudiollm.github.io/"> ホームページ </a>
｜<a href="#最新动态"> 最新情報 </a>
｜<a href="#性能评测"> 性能評価 </a>
｜<a href="#环境安装"> 環境インストール </a>
｜<a href="#用法教程"> 使用方法チュートリアル </a>
｜<a href="#联系我们"> お問い合わせ </a>
</h4>

モデルリポジトリ：[modelscope](https://www.modelscope.cn/models/iic/SenseVoiceSmall)，[huggingface](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)

オンライン体験：
[modelscope demo](https://www.modelscope.cn/studios/iic/SenseVoice), [huggingface space](https://huggingface.co/spaces/FunAudioLLM/SenseVoice)

</div>

<a name="核心功能"></a>
# コア機能 🎯
**SenseVoice**は、高精度な多言語音声認識、感情認識、および音声イベント検出に焦点を当てています。
- **多言語認識：** 40万時間以上のデータを使用してトレーニングされ、50以上の言語をサポートし、認識性能はWhisperモデルを上回ります。
- **リッチテキスト認識：** 
  - 優れた感情認識能力を持ち、テストデータで現在の最良の感情認識モデルの効果を達成および上回ります。
  - 音声イベント検出能力を提供し、音楽、拍手、笑い声、泣き声、咳、くしゃみなどのさまざまな一般的な人間とコンピュータのインタラクションイベントを検出します。
- **効率的な推論：** SenseVoice-Smallモデルは非自己回帰エンドツーエンドフレームワークを採用しており、推論遅延が非常に低く、10秒の音声の推論に70msしかかかりません。Whisper-Largeより15倍高速です。
- **簡単な微調整：** 便利な微調整スクリプトと戦略を提供し、ユーザーがビジネスシナリオに応じてロングテールサンプルの問題を簡単に解決できるようにします。
- **サービス展開：** マルチコンカレントリクエストをサポートする完全なサービス展開パイプラインを提供し、クライアントサイドの言語にはPython、C++、HTML、Java、C#などがあります。

<a name="最新动态"></a>
# 最新情報 🔥
- 2024/7：新しく[ONNX](./demo_onnx.py)と[libtorch](./demo_libtorch.py)のエクスポート機能を追加し、Pythonバージョンのランタイム：[funasr-onnx-0.4.0](https://pypi.org/project/funasr-onnx/)、[funasr-torch-0.1.1](https://pypi.org/project/funasr-torch/)も提供開始。
- 2024/7: [SenseVoice-Small](https://www.modelscope.cn/models/iic/SenseVoiceSmall) 多言語音声理解モデルがオープンソース化されました。中国語、広東語、英語、日本語、韓国語の多言語音声認識、感情認識、およびイベント検出能力をサポートし、非常に低い推論遅延を実現しています。
- 2024/7: CosyVoiceは自然な音声生成に取り組んでおり、多言語、音色、感情制御をサポートします。多言語音声生成、ゼロショット音声生成、クロスランゲージ音声クローン、および指示に従う能力に優れています。[CosyVoice repo](https://github.com/FunAudioLLM/CosyVoice) and [CosyVoice オンライン体験](https://www.modelscope.cn/studios/iic/CosyVoice-300M).
- 2024/7: [FunASR](https://github.com/modelscope/FunASR) は、音声認識（ASR）、音声活動検出（VAD）、句読点復元、言語モデル、話者検証、話者分離、およびマルチトーカーASRなどの機能を提供する基本的な音声認識ツールキットです。

<a name="Benchmarks"></a>
# ベンチマーク 📝

## 多言語音声認識

オープンソースのベンチマークデータセット（AISHELL-1、AISHELL-2、Wenetspeech、Librispeech、Common Voiceを含む）でSenseVoiceとWhisperの多言語音声認識性能と推論効率を比較しました。中国語と広東語の認識効果において、SenseVoice-Smallモデルは明らかな効果の優位性を持っています。

<div align="center">  
<img src="image/asr_results1.png" width="400" /><img src="image/asr_results2.png" width="400" />
</div>

## 感情認識

現在、広く使用されている感情認識のテスト指標と方法が不足しているため、複数のテストセットでさまざまな指標をテストし、最近のベンチマークの複数の結果と包括的に比較しました。選択されたテストセットには、中国語/英語の両方の言語と、パフォーマンス、映画、自然な会話などのさまざまなスタイルのデータが含まれています。ターゲットデータの微調整を行わない前提で、SenseVoiceはテストデータで現在の最良の感情認識モデルの効果を達成および上回ることができました。

<div align="center">  
<img src="image/ser_table.png" width="1000" />
</div>

さらに、テストセットで複数のオープンソースの感情認識モデルを比較し、結果はSenseVoice-Largeモデルがほぼすべてのデータで最良の効果を達成し、SenseVoice-Smallモデルも多数のデータセットで他のオープンソースモデルを上回る効果を達成したことを示しています。

<div align="center">  
<img src="image/ser_figure.png" width="500" />
</div>

## イベント検出

SenseVoiceは音声データのみでトレーニングされていますが、イベント検出モデルとして単独で使用することもできます。環境音分類ESC-50データセットで、現在業界で広く使用されているBEATSおよびPANNモデルの効果と比較しました。SenseVoiceモデルはこれらのタスクで良好な効果を達成しましたが、トレーニングデータとトレーニング方法の制約により、イベント分類の効果は専門のイベント検出モデルと比較してまだ一定の差があります。

<div align="center">  
<img src="image/aed_figure.png" width="500" />
</div>

## 推論効率

SenseVoice-smallモデルは非自己回帰エンドツーエンドアーキテクチャを採用しており、推論遅延が非常に低いです。Whisper-Smallモデルと同等のパラメータ量で、Whisper-Smallモデルより5倍高速で、Whisper-Largeモデルより15倍高速です。同時に、SenseVoice-smallモデルは音声の長さが増加しても、推論時間に明らかな増加はありません。

<div align="center">  
<img src="image/inference.png" width="1000" />
</div>

<a name="环境安装"></a>
# 環境インストール 🐍

```shell
pip install -r requirements.txt
```

<a name="用法教程"></a>
# 使用方法 🛠️

## 推論

任意の形式の音声入力をサポートし、任意の長さの入力をサポートします。

```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"


model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",  
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

# en
res = model.generate(
    input=f"{model.model_path}/example/en.mp3",
    cache={},
    language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)
```

<details><summary>パラメータの説明（クリックして展開）</summary>

- `model_dir`：モデル名、またはローカルディスク上のモデルパス。
- `trust_remote_code`：
  - `True`は、modelコードの実装が`remote_code`からロードされることを意味し、`remote_code`は`model`コードの正確な位置を指定します（例：現在のディレクトリの`model.py`）。絶対パス、相対パス、およびネットワークURLをサポートします。
  - `False`は、modelコードの実装が[FunASR](https://github.com/modelscope/FunASR)内部に統合されたバージョンであることを意味し、この場合、現在のディレクトリの`model.py`を変更しても効果がありません。FunASR内部バージョンがロードされるためです。モデルコード[こちらを参照](https://github.com/modelscope/FunASR/tree/main/funasr/models/sense_voice)。
- `vad_model`：VAD（音声活動検出）を有効にすることを示します。VADの目的は、長い音声を短いクリップに分割することです。この場合、推論時間にはVADとSenseVoiceの合計消費が含まれ、エンドツーエンドの遅延を表します。SenseVoiceモデルの推論時間を個別にテストする場合は、VADモデルを無効にできます。
- `vad_kwargs`：VADモデルの設定を指定します。`max_single_segment_time`：`vad_model`による音声セグメントの最大長を示し、単位はミリ秒（ms）です。
- `use_itn`：出力結果に句読点と逆テキスト正規化が含まれるかどうか。
- `batch_size_s`：動的バッチの使用を示し、バッチ内の音声の合計長を秒（s）で測定します。
- `merge_vad`：VADモデルによって分割された短い音声フラグメントをマージするかどうか。マージ後の長さは`merge_length_s`で、単位は秒（s）です。
- `ban_emo_unk`：emo_unkラベルを無効にする。
</details>

すべての入力が短い音声（30秒未満）であり、バッチ推論が必要な場合、推論効率を向上させるためにVADモデルを削除し、`batch_size`を設定できます。

```python
model = AutoModel(model=model_dir, trust_remote_code=True, device="cuda:0")

res = model.generate(
    input=f"{model.model_path}/example/en.mp3",
    cache={},
    language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size=64, 
)
```

詳細な使用方法については、[ドキュメント](https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README.md)を参照してください。

### 直接推論

任意の形式の音声入力をサポートし、入力音声の長さは30秒以下に制限されます。

```python
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")
m.eval()

res = m.inference(
    data_in=f"{kwargs['model_path']}/example/en.mp3",
    language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
    ban_emo_unk=False,
    **kwargs,
)

text = rich_transcription_postprocess(res[0][0]["text"])
print(text)
```

## サービス展開

未完了

### エクスポートとテスト
<details><summary>ONNXとLibtorchのエクスポート</summary>

#### ONNX
```python
# pip3 install -U funasr funasr-onnx
from pathlib import Path
from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess


model_dir = "iic/SenseVoiceSmall"

model = SenseVoiceSmall(model_dir, batch_size=10, quantize=True)

# inference
wav_or_scp = ["{}/.cache/modelscope/hub/{}/example/en.mp3".format(Path.home(), model_dir)]

res = model(wav_or_scp, language="auto", use_itn=True)
print([rich_transcription_postprocess(i) for i in res])
```
備考：ONNXモデルは元のモデルディレクトリにエクスポートされます。

#### Libtorch
```python
from pathlib import Path
from funasr_torch import SenseVoiceSmall
from funasr_torch.utils.postprocess_utils import rich_transcription_postprocess


model_dir = "iic/SenseVoiceSmall"

model = SenseVoiceSmall(model_dir, batch_size=10, device="cuda:0")

wav_or_scp = ["{}/.cache/modelscope/hub/{}/example/en.mp3".format(Path.home(), model_dir)]

res = model(wav_or_scp, language="auto", use_itn=True)
print([rich_transcription_postprocess(i) for i in res])
```
備考：Libtorchモデルは元のモデルディレクトリにエクスポートされます。

</details>

### 展開

### FastAPIを使った展開
```shell
export SENSEVOICE_DEVICE=cuda:0
fastapi run --port 50000
```

## 微調整

### トレーニング環境のインストール

```shell
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip3 install -e ./
```

### データ準備

データ例
```text
{"key": "YOU0000008470_S0000238_punc_itn", "text_language": "<|en|>", "emo_target": "<|NEUTRAL|>", "event_target": "<|Speech|>", "with_or_wo_itn": "<|withitn|>", "target": "Including legal due diligence, subscription agreement, negotiation.", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/industrial_data/english_all/audio/YOU0000008470_S0000238.wav", "target_len": 7, "source_len": 140}
{"key": "AUD0000001556_S0007580", "text_language": "<|en|>", "emo_target": "<|NEUTRAL|>", "event_target": "<|Speech|>", "with_or_wo_itn": "<|woitn|>", "target": "there is a tendency to identify the self or take interest in what one has got used to", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/industrial_data/english_all/audio/AUD0000001556_S0007580.wav", "target_len": 18, "source_len": 360}
```
詳細は `data/train_example.jsonl` を参照してください。

<details><summary>データ準備の詳細</summary>

説明：
- `key`：音声ファイルのユニークID
- `source`：音声ファイルのパス
- `source_len`：音声ファイルのfbankフレーム数
- `target`：文字起こし結果
- `target_len`：target（文字起こし）の長さ
- `text_language`：音声ファイルの言語ID
- `emo_target`：音声ファイルの感情ラベル
- `event_target`：音声ファイルのイベントラベル
- `with_or_wo_itn`：句読点と逆テキスト正規化を含むかどうか

`train_text.txt`
```bash
BAC009S0764W0121 甚至出现交易几乎停滞的情况
BAC009S0916W0489 湖北一公司以员工名义贷款数十员工负债千万
asr_example_cn_en 所有只要处理 data 不管你是做 machine learning 做 deep learning 做 data analytics 做 data science 也好 scientist 也好通通都要都做的基本功啊那 again 先先对有一些>也许对
ID0012W0014 he tried to think how it could be
```
`train_wav.scp`
```bash
BAC009S0764W0121 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav
BAC009S0916W0489 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0916W0489.wav
asr_example_cn_en https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_cn_en.wav
ID0012W0014 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_en.wav
```
`train_text_language.txt`
言語IDは `<|zh|>`、`<|en|>`、`<|yue|>`、`<|ja|>`、および `<|ko|>`を含みます。
```bash
BAC009S0764W0121 <|zh|>
BAC009S0916W0489 <|zh|>
asr_example_cn_en <|zh|>
ID0012W0014 <|en|>
```
`train_emo.txt`
感情ラベルは、`<|HAPPY|>`、`<|SAD|>`、`<|ANGRY|>`、`<|NEUTRAL|>`、`<|FEARFUL|>`、`<|DISGUSTED|>` および `<|SURPRISED|>`を含みます。
```bash
BAC009S0764W0121 <|NEUTRAL|>
BAC009S0916W0489 <|NEUTRAL|>
asr_example_cn_en <|NEUTRAL|>
ID0012W0014 <|NEUTRAL|>
```
`train_event.txt`
イベントラベルは、 `<|BGM|>`、`<|Speech|>`、`<|Applause|>`、`<|Laughter|>`、`<|Cry|>`、`<|Sneeze|>`、`<|Breath|>` および `<|Cough|>`を含みます。
```bash
BAC009S0764W0121 <|Speech|>
BAC009S0916W0489 <|Speech|>
asr_example_cn_en <|Speech|>
ID0012W0014 <|Speech|>
```
`コマンド`
```shell
# wav.scp、text.txt、text_language.txt、emo_target.txt、event_target.txt から train.jsonl と val.jsonl を生成します
sensevoice2jsonl \
++scp_file_list='["../../../data/list/train_wav.scp", "../../../data/list/train_text.txt", "../../../data/list/train_text_language.txt", "../../../data/list/train_emo.txt", "../../../data/list/train_event.txt"]' \
++data_type_list='["source", "target", "text_language", "emo_target", "event_target"]' \
++jsonl_file_out="../../../data/list/train.jsonl"
```
`train_text_language.txt`、`train_emo_target.txt`、`train_event_target.txt` がない場合、`SenseVoice` モデルを使用して言語、感情、およびイベントラベルが自動的に予測されます。
```shell
# wav.scp と text.txt から train.jsonl と val.jsonl を生成します
sensevoice2jsonl \
++scp_file_list='["../../../data/list/train_wav.scp", "../../../data/list/train_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="../../../data/list/train.jsonl"
```
</details>

### トレーニングの開始

`finetune.sh`の`train_tool`を、前述のFunASRパス内の`funasr/bin/train_ds.py`の絶対パスに変更することを忘れないでください。

```shell
bash finetune.sh
```

## WebUI

```shell
python webui.py
```

<div align="center"><img src="image/webui.png" width="700"/> </div>

## 注目すべきサードパーティの取り組み
- Triton (GPU) デプロイメントのベストプラクティス：Triton + TensorRT を使用し、FP32 でテスト。V100 GPU で加速比 526 を達成。FP16 のサポートは進行中です。[リポジトリ](https://github.com/modelscope/FunASR/blob/main/runtime/triton_gpu/README.md)
- Sherpa-onnx デプロイメントのベストプラクティス：SenseVoice を10種類のプログラミング言語（C++, C, Python, C#, Go, Swift, Kotlin, Java, JavaScript, Dart）で使用可能。また、iOS, Android, Raspberry Pi などのプラットフォームでも SenseVoice をデプロイできます。[リポジトリ](https://k2-fsa.github.io/sherpa/onnx/sense-voice/index.html)
- [SenseVoice.cpp](https://github.com/lovemefan/SenseVoice.cpp) GGMLに基づいて純粋なC/C++でSenseVoiceを推測し、3ビット、4ビット、5ビット、8ビット量子化などをサポートし、サードパーティの依存関係はありません。
- [streaming-sensevoice](https://github.com/pengzhendong/streaming-sensevoice) ストリーム型SenseVoiceは、チャンク（chunk）方式で推論を行います。擬似ストリーミング処理を実現するために、一部の精度を犠牲にして切り捨て注意機構（truncated attention）を採用しています。さらに、この技術はCTCプレフィックスビームサーチ（CTC prefix beam search）とホットワード強化機能もサポートしています。
- [OmniSenseVoice](https://github.com/lifeiteng/OmniSenseVoice) は、超高速推論とバッチ処理のために最適化されています。

# お問い合わせ

使用中に問題が発生した場合は、githubページで直接Issuesを提起できます。音声に興味のある方は、以下のDingTalkグループQRコードをスキャンしてコミュニティグループに参加し、交流と議論を行ってください。

|                          FunASR                          |
|:--------------------------------------------------------:|
| <img src="image/dingding_funasr.png" width="250"/></div> |

