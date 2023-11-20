[//]: # (<div align="left"><img src="docs/images/funasr_logo.jpg" width="400"/></div>)

(简体中文|[English](./README.md))

# FunASR: A Fundamental End-to-End Speech Recognition Toolkit
<p align="left">
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-brightgreen.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Pytorch-%3E%3D1.11-blue"></a>
</p>

FunASR希望在语音识别的学术研究和工业应用之间架起一座桥梁。通过发布工业级语音识别模型的训练和微调，研究人员和开发人员可以更方便地进行语音识别模型的研究和生产，并推动语音识别生态的发展。让语音识别更有趣！

<div align="center">  
<h4>
 <a href="#核心功能"> 核心功能 </a>   
｜<a href="#最新动态"> 最新动态 </a>
｜<a href="#安装教程"> 安装 </a>
｜<a href="#快速开始"> 快速开始 </a>
｜<a href="https://alibaba-damo-academy.github.io/FunASR/en/index.html"> 教程文档 </a>
｜<a href="#模型仓库"> 模型仓库 </a>
｜<a href="#服务部署"> 服务部署 </a>
｜<a href="#联系我们"> 联系我们 </a>
</h4>
</div>

<a name="核心功能"></a>
## 核心功能
- FunASR是一个基础语音识别工具包，提供多种功能，包括语音识别（ASR）、语音端点检测（VAD）、标点恢复、语言模型、说话人验证、说话人分离和多人对话语音识别等。FunASR提供了便捷的脚本和教程，支持预训练好的模型的推理与微调。
- 我们在[ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition)与[huggingface](https://huggingface.co/FunASR)上发布了大量开源数据集或者海量工业数据训练的模型，可以通过我们的[模型仓库](https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/model_zoo/modelscope_models.md)了解模型的详细信息。代表性的[Paraformer](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)非自回归端到端语音识别模型具有高精度、高效率、便捷部署的优点，支持快速构建语音识别服务，详细信息可以阅读([服务部署文档](runtime/readme_cn.md))。

<a name="最新动态"></a>
## 最新动态
- 2023/11/08：中文离线文件转写服务3.0 CPU版本发布，新增标点大模型、Ngram语言模型与wfst热词，详细信息参阅([一键部署文档](runtime/readme_cn.md#中文离线文件转写服务cpu版本))
- 2023/10/17: 英文离线文件转写服务一键部署的CPU版本发布，详细信息参阅([一键部署文档](runtime/readme_cn.md#英文离线文件转写服务cpu版本))
- 2023/10/13: [SlideSpeech](https://slidespeech.github.io/): 一个大规模的多模态音视频语料库，主要是在线会议或者在线课程场景，包含了大量与发言人讲话实时同步的幻灯片。
- 2023.10.10: [Paraformer-long-Spk](https://github.com/alibaba-damo-academy/FunASR/blob/main/egs_modelscope/asr_vad_spk/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn/demo.py)模型发布，支持在长语音识别的基础上获取每句话的说话人标签。
- 2023.10.07: [FunCodec](https://github.com/alibaba-damo-academy/FunCodec): FunCodec提供开源模型和训练工具，可以用于音频离散编码，以及基于离散编码的语音识别、语音合成等任务。
- 2023.09.01: 中文离线文件转写服务2.0 CPU版本发布，新增ffmpeg、时间戳与热词模型支持，详细信息参阅([一键部署文档](runtime/readme_cn.md#中文离线文件转写服务cpu版本))
- 2023.08.07: 中文实时语音听写服务一键部署的CPU版本发布，详细信息参阅([一键部署文档](runtime/readme_cn.md#中文实时语音听写服务cpu版本))
- 2023.07.17: BAT一种低延迟低内存消耗的RNN-T模型发布，详细信息参阅（[BAT](egs/aishell/bat)）
- 2023.06.26: ASRU2023 多通道多方会议转录挑战赛2.0完成竞赛结果公布，详细信息参阅（[M2MeT2.0](https://alibaba-damo-academy.github.io/FunASR/m2met2_cn/index.html)）

<a name="安装教程"></a>
## 安装教程
FunASR安装教程请阅读（[Installation](https://alibaba-damo-academy.github.io/FunASR/en/installation/installation.html)）

## 模型仓库

FunASR开源了大量在工业数据上预训练模型，您可以在[模型许可协议](./MODEL_LICENSE)下自由使用、复制、修改和分享FunASR模型，下面列举代表性的模型，更多模型请参考[模型仓库]()。

（注：[🤗]()表示Huggingface模型仓库链接，[⭐]()表示ModelScope模型仓库链接）


|                                                                              模型名字                                                                               |        任务详情        |     训练数据     | 参数量  |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------:|:------------:|:----:|
|     paraformer-zh <br> ([⭐](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)  [🤗]() )     |  语音识别，带时间戳输出，非实时   |  60000小时，中文  | 220M |
|                 paraformer-zh-spk <br> ( [⭐](https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn/summary)  [🤗]() )                 | 分角色语音识别，带时间戳输出，非实时 |  60000小时，中文  | 220M |
|        paraformer-zh-online <br> ( [⭐](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary) [🤗]() )         |      语音识别，实时       |  60000小时，中文  | 220M |
|          paraformer-en <br> ( [⭐](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020/summary) [🤗]() )          | 语音识别，非实时 |  50000小时，英文  | 220M |
|                                                                paraformer-en-spk <br> ([⭐]() [🤗]() )                                                                |      语音识别，非实时      |  50000小时，英文  | 220M |
|                      conformer-en <br> ( [⭐](https://modelscope.cn/models/damo/speech_conformer_asr-en-16k-vocab4199-pytorch/summary) [🤗]() )                       |      语音识别，非实时      |  50000小时，英文  | 220M |
|                      ct-punc <br> ( [⭐](https://modelscope.cn/models/damo/punc_ct-transformer_cn-en-common-vocab471067-large/summary) [🤗]() )                       |      标点恢复      |  100M，中文与英文  | 1.1G | 
|                           fsmn-vad <br> ( [⭐](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) [🤗]() )                           |     语音端点检测，实时      | 5000小时，中文与英文 | 0.4M | 
|                           fa-zh <br> ( [⭐](https://modelscope.cn/models/damo/speech_timestamp_prediction-v1-16k-offline/summary) [🤗]() )                            |   字级别时间戳预测         |  50000小时，中文  | 38M  |


<a name="快速开始"></a>
## 快速开始
FunASR支持数万小时工业数据训练的模型的推理和微调，详细信息可以参阅（[modelscope_egs](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_pipeline/quick_start.html)）；也支持学术标准数据集模型的训练和微调，详细信息可以参阅（[egs](https://alibaba-damo-academy.github.io/FunASR/en/academic_recipe/asr_recipe.html)）。

下面为快速上手教程，测试音频（[中文](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav)，[英文]()）

### 可执行命令行

```shell
funasr --model paraformer-zh asr_example_zh.wav
```

注：支持单条音频文件识别，也支持文件列表，列表为kaldi风格wav.scp：`wav_id   wav_path`

### 非实时语音识别
```python
from funasr import infer

p = infer(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", model_hub="ms")

res = p("asr_example_zh.wav", batch_size_token=5000)
print(res)
```
注：`model_hub`：表示模型仓库，`ms`为选择modelscope下载，`hf`为选择huggingface下载。

### 实时语音识别
```python
from funasr import infer

p = infer(model="paraformer-zh-streaming", model_hub="ms")

chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
param_dict = {"cache": dict(), "is_final": False, "chunk_size": chunk_size, "encoder_chunk_look_back": 4, "decoder_chunk_look_back": 1}

import torchaudio
speech = torchaudio.load("asr_example_zh.wav")[0][0]
speech_length = speech.shape[0]

stride_size = chunk_size[1] * 960
sample_offset = 0
for sample_offset in range(0, speech_length, min(stride_size, speech_length - sample_offset)):
    param_dict["is_final"] = True if sample_offset + stride_size >= speech_length - 1 else False
    input = speech[sample_offset: sample_offset + stride_size]
    rec_result = p(input=input, param_dict=param_dict)
    print(rec_result)
```
注：`chunk_size`为流式延时配置，`[0,10,5]`表示上屏实时出字粒度为`10*60=600ms`，未来信息为`5*60=300ms`。每次推理输入为`600ms`（采样点数为`16000*0.6=960`），输出为对应文字，最后一个语音片段输入需要设置`is_final=True`来强制输出最后一个字。

更多详细用法（[新人文档](https://alibaba-damo-academy.github.io/FunASR/en/funasr/quick_start_zh.html)）


<a name="服务部署"></a>
## 服务部署
FunASR支持预训练或者进一步微调的模型进行服务部署。目前支持以下几种服务部署：

- 中文离线文件转写服务（CPU版本），已完成
- 中文流式语音识别服务（CPU版本），已完成
- 英文离线文件转写服务（CPU版本），已完成
- 中文离线文件转写服务（GPU版本），进行中
- 更多支持中

详细信息可以参阅([服务部署文档](runtime/readme_cn.md))。


<a name="社区交流"></a>
## 联系我们

如果您在使用中遇到问题，可以直接在github页面提Issues。欢迎语音兴趣爱好者扫描以下的钉钉群或者微信群二维码加入社区群，进行交流和讨论。

|                                  钉钉群                                  |                          微信                           |
|:---------------------------------------------------------------------:|:-----------------------------------------------------:|
| <div align="left"><img src="docs/images/dingding.jpg" width="250"/>   | <img src="docs/images/wechat.png" width="215"/></div> |

## 社区贡献者

| <div align="left"><img src="docs/images/alibaba.png" width="260"/> | <div align="left"><img src="docs/images/nwpu.png" width="260"/> | <img src="docs/images/China_Telecom.png" width="200"/> </div>  | <img src="docs/images/RapidAI.png" width="200"/> </div> | <img src="docs/images/aihealthx.png" width="200"/> </div> | <img src="docs/images/XVERSE.png" width="250"/> </div> |
|:------------------------------------------------------------------:|:---------------------------------------------------------------:|:--------------------------------------------------------------:|:-------------------------------------------------------:|:-----------------------------------------------------------:|:------------------------------------------------------:|

贡献者名单请参考（[致谢名单](./Acknowledge.md)）


## 许可协议
项目遵循[The MIT License](https://opensource.org/licenses/MIT)开源协议，模型许可协议请参考（[模型协议](./MODEL_LICENSE)）


## 论文引用

``` bibtex
@inproceedings{gao2023funasr,
  author={Zhifu Gao and Zerui Li and Jiaming Wang and Haoneng Luo and Xian Shi and Mengzhe Chen and Yabin Li and Lingyun Zuo and Zhihao Du and Zhangyu Xiao and Shiliang Zhang},
  title={FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
  year={2023},
  booktitle={INTERSPEECH},
}
@inproceedings{An2023bat,
  author={Keyu An and Xian Shi and Shiliang Zhang},
  title={BAT: Boundary aware transducer for memory-efficient and low-latency ASR},
  year={2023},
  booktitle={INTERSPEECH},
}
@inproceedings{gao22b_interspeech,
  author={Zhifu Gao and ShiLiang Zhang and Ian McLoughlin and Zhijie Yan},
  title={{Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={2063--2067},
  doi={10.21437/Interspeech.2022-9996}
}
```
