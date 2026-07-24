# SenseVoice

「简体中文」|「[English](./README.md)」|「[日本語](./README_ja.md)」

SenseVoice 是具有音频理解能力的音频基础模型，包括语音识别（ASR）、语种识别（LID）、语音情感识别（SER）和声学事件分类（AEC）或声学事件检测（AED）。本项目提供 SenseVoice 模型的介绍以及在多个任务测试集上的 benchmark，以及体验模型所需的环境安装的与推理方式。

<div align="center">  
<img src="image/sensevoice2.png">
</div>

<div align="center">  
<h4>
<a href="https://funaudiollm.github.io/"> Homepage </a>
｜<a href="#最新动态"> 最新动态 </a>
｜<a href="#性能评测"> 性能评测 </a>
｜<a href="#环境安装"> 环境安装 </a>
｜<a href="#用法教程"> 用法教程 </a>
｜<a href="#联系我们"> 联系我们 </a>

</h4>

模型仓库：[modelscope](https://www.modelscope.cn/models/iic/SenseVoiceSmall)，[huggingface](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)

在线体验：
[modelscope demo](https://www.modelscope.cn/studios/iic/SenseVoice), [huggingface space](https://huggingface.co/spaces/FunAudioLLM/SenseVoice)

</div>

<a name="核心功能"></a>

# 核心功能 🎯

**SenseVoice** 专注于高精度多语言语音识别、情感辨识和音频事件检测

- **多语言识别：** 采用超过 40 万小时数据训练，支持超过 50 种语言，识别效果上优于 Whisper 模型。
- **富文本识别：**
  - 具备优秀的情感识别，能够在测试数据上达到和超过目前最佳情感识别模型的效果。
  - 支持声音事件检测能力，支持音乐、掌声、笑声、哭声、咳嗽、喷嚏等多种常见人机交互事件进行检测。
- **高效推理：** SenseVoice-Small 模型采用非自回归端到端框架，推理延迟极低，10s 音频推理仅耗时 70ms，15 倍优于 Whisper-Large。
- **微调定制：** 具备便捷的微调脚本与策略，方便用户根据业务场景修复长尾样本问题。
- **服务部署：** 具有完整的服务部署链路，支持多并发请求，支持客户端语言有，python、c++、html、java 与 c# 等。

<a name="最新动态"></a>

# 最新动态 🔥

- 2024/7：新增加导出 [ONNX](./demo_onnx.py) 与 [libtorch](./demo_libtorch.py) 功能，以及 python 版本 runtime：[funasr-onnx-0.4.0](https://pypi.org/project/funasr-onnx/)，[funasr-torch-0.1.1](https://pypi.org/project/funasr-torch/)
- 2024/7: [SenseVoice-Small](https://www.modelscope.cn/models/iic/SenseVoiceSmall) 多语言音频理解模型开源，支持中、粤、英、日、韩语的多语言语音识别，情感识别和事件检测能力，具有极低的推理延迟。。
- 2024/7: CosyVoice 致力于自然语音生成，支持多语言、音色和情感控制，擅长多语言语音生成、零样本语音生成、跨语言语音克隆以及遵循指令的能力。[CosyVoice repo](https://github.com/QwenAudio/CosyVoice) and [CosyVoice 在线体验](https://www.modelscope.cn/studios/iic/CosyVoice-300M).
- 2024/7: [FunASR](https://github.com/modelscope/FunASR) 是一个基础语音识别工具包，提供多种功能，包括语音识别（ASR）、语音端点检测（VAD）、标点恢复、语言模型、说话人验证、说话人分离和多人对话语音识别等。

<a name="Benchmarks"></a>

# 性能评测 📝

## 多语言语音识别

我们在开源基准数据集（包括 AISHELL-1、AISHELL-2、Wenetspeech、Librispeech 和 Common Voice）上比较了 SenseVoice 与 Whisper 的多语言语音识别性能和推理效率。在中文和粤语识别效果上，SenseVoice-Small 模型具有明显的效果优势。

<div align="center">  
<img src="image/asr_results1.png" width="400" /><img src="image/asr_results2.png" width="400" />
</div>

## 情感识别

由于目前缺乏被广泛使用的情感识别测试指标和方法，我们在多个测试集的多种指标进行测试，并与近年来 Benchmark 上的多个结果进行了全面的对比。所选取的测试集同时包含中文 / 英文两种语言以及表演、影视剧、自然对话等多种风格的数据，在不进行目标数据微调的前提下，SenseVoice 能够在测试数据上达到和超过目前最佳情感识别模型的效果。

<div align="center">  
<img src="image/ser_table.png" width="1000" />
</div>

同时，我们还在测试集上对多个开源情感识别模型进行对比，结果表明，SenseVoice-Large 模型可以在几乎所有数据上都达到了最佳效果，而 SenseVoice-Small 模型同样可以在多数数据集上取得超越其他开源模型的效果。

<div align="center">  
<img src="image/ser_figure.png" width="500" />
</div>

## 事件检测

尽管 SenseVoice 只在语音数据上进行训练，它仍然可以作为事件检测模型进行单独使用。我们在环境音分类 ESC-50 数据集上与目前业内广泛使用的 BEATS 与 PANN 模型的效果进行了对比。SenseVoice 模型能够在这些任务上取得较好的效果，但受限于训练数据与训练方式，其事件分类效果专业的事件检测模型相比仍然有一定的差距。

<div align="center">  
<img src="image/aed_figure.png" width="500" />
</div>

## 推理效率

SenseVoice-small 模型采用非自回归端到端架构，推理延迟极低。在参数量与 Whisper-Small 模型相当的情况下，比 Whisper-Small 模型推理速度快 5 倍，比 Whisper-Large 模型快 15 倍。同时 SenseVoice-small 模型在音频时长增加的情况下，推理耗时也无明显增加。

<div align="center">  
<img src="image/inference.png" width="1000" />
</div>

<a name="环境安装"></a>

# 安装依赖环境 🐍

```shell
pip install -r requirements.txt
```

<a name="用法教程"></a>

# 用法 🛠️

## 推理

### 使用 funasr 推理

支持任意格式音频输入，支持任意时长输入

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
    merge_vad=True,
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)
```

<details><summary> 参数说明（点击展开）</summary>

- `model_dir`：模型名称，或本地磁盘中的模型路径。
- `trust_remote_code`：
  - `True` 表示 model 代码实现从 `remote_code` 处加载，`remote_code` 指定 `model` 具体代码的位置（例如，当前目录下的 `model.py`），支持绝对路径与相对路径，以及网络 url。
  - `False` 表示，model 代码实现为 [FunASR](https://github.com/modelscope/FunASR) 内部集成版本，此时修改当前目录下的 `model.py` 不会生效，因为加载的是 funasr 内部版本，模型代码 [点击查看](https://github.com/modelscope/FunASR/tree/main/funasr/models/sense_voice)。
- `vad_model`：表示开启 VAD，VAD 的作用是将长音频切割成短音频，此时推理耗时包括了 VAD 与 SenseVoice 总耗时，为链路耗时，如果需要单独测试 SenseVoice 模型耗时，可以关闭 VAD 模型。
- `vad_kwargs`：表示 VAD 模型配置，`max_single_segment_time`: 表示 `vad_model` 最大切割音频时长，单位是毫秒 ms。
- `use_itn`：输出结果中是否包含标点与逆文本正则化。
- `batch_size_s` 表示采用动态 batch，batch 中总音频时长，单位为秒 s。
- `merge_vad`：是否将 vad 模型切割的短音频碎片合成，合并后长度为 `merge_length_s`，单位为秒 s。
- `ban_emo_unk`：禁用 emo_unk 标签，禁用后所有的句子都会被赋与情感标签。默认 `False`

</details>

如果输入均为短音频（小于 30s），并且需要批量化推理，为了加快推理效率，可以移除 vad 模型，并设置 `batch_size`

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

更多详细用法，请参考 [文档](https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README.md)

### 直接推理

支持任意格式音频输入，输入音频时长限制在 30s 以下

```python
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")
m.eval()

res = m.inference(
    data_in=f"{kwargs ['model_path']}/example/en.mp3",
    language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
    ban_emo_unk=False,
    **kwargs,
)

text = rich_transcription_postprocess(res [0][0]["text"])
print(text)
```

## 服务部署

Undo

### 导出与测试

<details><summary>ONNX 与 Libtorch 导出 </summary>

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

备注：ONNX 模型导出到原模型目录中

#### Libtorch

```python
from pathlib import Path
from funasr_torch import SenseVoiceSmall
from funasr_torch.utils.postprocess_utils import rich_transcription_postprocess


model_dir = "iic/SenseVoiceSmall"

model = SenseVoiceSmall(model_dir, batch_size=10, device="cuda:0")

wav_or_scp = ["{}/.cache/modelscope/hub/{}/example/en.mp3".format(Path.home(), model_dir)]

res = model(wav_or_scp, language="auto", use_itn=True)
print([rich_transcription_postprocess (i) for i in res])
```

备注：Libtorch 模型导出到原模型目录中

</details>

### 部署

### 使用 FastAPI 部署

```shell
export SENSEVOICE_DEVICE=cuda:0
fastapi run --port 50000
```

## 微调

### 安装训练环境

```shell
git clone https://github.com/modelscope/FunASR.git && cd FunASR
pip3 install -e ./
```

### 数据准备

数据格式需要包括如下几个字段：

```text
{"key": "YOU0000008470_S0000238_punc_itn", "text_language": "<|en|>", "emo_target": "<|NEUTRAL|>", "event_target": "<|Speech|>", "with_or_wo_itn": "<|withitn|>", "target": "Including legal due diligence, subscription agreement, negotiation.", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/industrial_data/english_all/audio/YOU0000008470_S0000238.wav", "target_len": 7, "source_len": 140}
{"key": "AUD0000001556_S0007580", "text_language": "<|en|>", "emo_target": "<|NEUTRAL|>", "event_target": "<|Speech|>", "with_or_wo_itn": "<|woitn|>", "target": "there is a tendency to identify the self or take interest in what one has got used to", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/industrial_data/english_all/audio/AUD0000001556_S0007580.wav", "target_len": 18, "source_len": 360}
```

详细可以参考：`data/train_example.jsonl`

<details><summary > 数据准备细节介绍 </summary>

- `key`: 数据唯一 ID
- `source`：音频文件的路径
- `source_len`：音频文件的 fbank 帧数
- `target`：音频文件标注文本
- `target_len`：音频文件标注文本长度
- `text_language`：音频文件的语种标签
- `emo_target`：音频文件的情感标签
- `event_target`：音频文件的事件标签
- `with_or_wo_itn`：标注文本中是否包含标点与逆文本正则化

可以用指令 `sensevoice2jsonl` 从 train_wav.scp、train_text.txt、train_text_language.txt、train_emo_target.txt 和 train_event_target.txt 生成，准备过程如下：

`train_text.txt`

左边为数据唯一 ID，需与 `train_wav.scp` 中的 `ID` 一一对应
右边为音频文件标注文本，格式如下：

```bash
BAC009S0764W0121 甚至出现交易几乎停滞的情况
BAC009S0916W0489 湖北一公司以员工名义贷款数十员工负债千万
asr_example_cn_en 所有只要处理 data 不管你是做 machine learning 做 deep learning 做 data analytics 做 data science 也好 scientist 也好通通都要都做的基本功啊那 again 先先对有一些 > 也许对
ID0012W0014 he tried to think how it could be
```

`train_wav.scp`

左边为数据唯一 ID，需与 `train_text.txt` 中的 `ID` 一一对应
右边为音频文件的路径，格式如下

```bash
BAC009S0764W0121 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav
BAC009S0916W0489 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0916W0489.wav
asr_example_cn_en https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_cn_en.wav
ID0012W0014 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_en.wav
```

`train_text_language.txt`

左边为数据唯一 ID，需与 `train_text_language.txt` 中的 `ID` 一一对应
右边为音频文件的语种标签，支持 `<|zh|>`、`<|en|>`、`<|yue|>`、`<|ja|>` 和 `<|ko|>`，格式如下

```bash
BAC009S0764W0121 <|zh|>
BAC009S0916W0489 <|zh|>
asr_example_cn_en <|zh|>
ID0012W0014 <|en|>
```

`train_emo.txt`

左边为数据唯一 ID，需与 `train_emo.txt` 中的 `ID` 一一对应
右边为音频文件的情感标签，支持 `<|HAPPY|>`、`<|SAD|>`、`<|ANGRY|>`、`<|NEUTRAL|>`、`<|FEARFUL|>`、`<|DISGUSTED|>` 和 `<|SURPRISED|>`，格式如下

```bash
BAC009S0764W0121 <|NEUTRAL|>
BAC009S0916W0489 <|NEUTRAL|>
asr_example_cn_en <|NEUTRAL|>
ID0012W0014 <|NEUTRAL|>
```

`train_event.txt`

左边为数据唯一 ID，需与 `train_event.txt` 中的 `ID` 一一对应
右边为音频文件的事件标签，支持 `<|BGM|>`、`<|Speech|>`、`<|Applause|>`、`<|Laughter|>`、`<|Cry|>`、`<|Sneeze|>`、`<|Breath|>` 和 `<|Cough|>`，格式如下

```bash
BAC009S0764W0121 <|Speech|>
BAC009S0916W0489 <|Speech|>
asr_example_cn_en <|Speech|>
ID0012W0014 <|Speech|>
```

`生成指令`

```shell
# generate train.jsonl and val.jsonl from wav.scp, text.txt, text_language.txt, emo_target.txt, event_target.txt
sensevoice2jsonl \
++scp_file_list='["../../../data/list/train_wav.scp", "../../../data/list/train_text.txt", "../../../data/list/train_text_language.txt", "../../../data/list/train_emo.txt", "../../../data/list/train_event.txt"]' \
++data_type_list='["source", "target", "text_language", "emo_target", "event_target"]' \
++jsonl_file_out="../../../data/list/train.jsonl"
```

若无 train_text_language.txt、train_emo_target.txt 和 train_event_target.txt，则自动通过使用 `SenseVoice` 模型对语种、情感和事件打标。

```shell
# generate train.jsonl and val.jsonl from wav.scp and text.txt
sensevoice2jsonl \
++scp_file_list='["../../../data/list/train_wav.scp", "../../../data/list/train_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="../../../data/list/train.jsonl" \
++model_dir='iic/SenseVoiceSmall'
```

</details>

### 启动训练

注意修改 `finetune.sh` 中 `train_tool` 为你前面安装 FunASR 路径中 `funasr/bin/train_ds.py` 绝对路径

```shell
bash finetune.sh
```

需要增加新口音、语种或业务域并保持已有能力时，请参考[持续微调指南](CONTINUAL_FINETUNING_zh.md)。

## WebUI

```shell
python webui.py
```

<div align="center"><img src="image/webui.png" width="700"/> </div>

## 优秀三方工作

- Triton（GPU）部署最佳实践，triton + tensorrt，fp32 测试，V100 GPU 上加速比 526，fp16 支持中，[repo](https://github.com/modelscope/FunASR/blob/main/runtime/triton_gpu/README.md)
- sherpa-onnx 部署最佳实践，支持在 10 种编程语言里面使用 SenseVoice, 即 C++, C, Python, C#, Go, Swift, Kotlin, Java, JavaScript, Dart. 支持在 iOS, Android, Raspberry Pi 等平台使用 SenseVoice，[repo](https://k2-fsa.github.io/sherpa/onnx/sense-voice/index.html)
- [SenseVoice.cpp](https://github.com/lovemefan/SenseVoice.cpp) 基于GGML，在纯C/C++中推断SenseVoice，支持3位、4位、5位、8位量化等，无需第三方依赖。
- [流式SenseVoice](https://github.com/pengzhendong/streaming-sensevoice)，通过分块（chunk）的方式进行推理，为了实现伪流式处理，采用了截断注意力机制（truncated attention），牺牲了部分精度。此外，该技术还支持CTC前缀束搜索（CTC prefix beam search）以及热词增强功能。
- [OmniSenseVoice](https://github.com/lifeiteng/OmniSenseVoice) 轻量化推理库，支持batch推理。

# 联系我们

如果您在使用中遇到问题，可以直接在 github 页面提 Issues。欢迎语音兴趣爱好者扫描以下的钉钉群二维码加入社区群，进行交流和讨论。

|                          FunASR                          |
|:--------------------------------------------------------:|
| <img src="image/dingding_funasr.png" width="250"/></div> |
