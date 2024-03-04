---
tasks:
- audio-visual-speech-recognition 
domain:
- audio, visual
model-type:
- Autoregressive
frameworks:
- pytorch
backbone:
- transformer/conformer
metrics:
- WER/B-WER
license: Apache License 2.0
language: 
- en
tags:
- FunASR
- Alibaba
- ICASSP 2024
- Audio-Visual
- Hotword
- Long-Context Biasing
datasets:
  train:
  - SlideSpeech corpus
  test:
  - dev and test of SlideSpeech corpus
indexing:
   results:
   - task:
       name: Audio-Visual Speech Recognition
     dataset:
       name: SlideSpeech corpus
       type: audio    # optional
       args: 16k sampling rate, 5002 bpe units  # optional
     metrics:
       - type: WER
         value: 18.8%  # float
         description: beamsearch search, withou lm, avg.
         args: default

widgets:
  - task: audio-visual-speech-recognition 
    inputs:
      - type: audio
        name: input
        title: 音频
      - type: text
        name: input
        title: OCR识别文本
finetune-support: True
---


# Paraformer-large模型介绍

## Highlights
- 热词版本：[Paraformer-large热词版模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/summary)支持热词定制功能，基于提供的热词列表进行激励增强，提升热词的召回率和准确率。
- 长音频版本：[Paraformer-large长音频模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)，集成VAD、ASR、标点与时间戳功能，可直接对时长为数小时音频进行识别，并输出带标点文字与时间戳。

## <strong>[FunASR开源项目介绍](https://github.com/alibaba-damo-academy/FunASR)</strong>
<strong>[FunASR](https://github.com/alibaba-damo-academy/FunASR)</strong>希望在语音识别的学术研究和工业应用之间架起一座桥梁。通过发布工业级语音识别模型的训练和微调，研究人员和开发人员可以更方便地进行语音识别模型的研究和生产，并推动语音识别生态的发展。让语音识别更有趣！

[**github仓库**](https://github.com/alibaba-damo-academy/FunASR)
| [**最新动态**](https://github.com/alibaba-damo-academy/FunASR#whats-new) 
| [**环境安装**](https://github.com/alibaba-damo-academy/FunASR#installation)
| [**服务部署**](https://www.funasr.com)
| [**模型库**](https://github.com/alibaba-damo-academy/FunASR/tree/main/model_zoo)
| [**联系我们**](https://github.com/alibaba-damo-academy/FunASR#contact)


## 模型原理介绍

随着在线会议和课程越来越普遍，如何利用视频幻灯片中丰富的文本信息来改善语音识别（Automatic  Speech Recognition， ASR）面临着新的挑战。视频中的幻灯片与语音实时同步，相比于统一的稀有词列表，能够提供更长的上下文相关信息。因此，我们提出了一种创新的长上下文偏置网络（LCB-net），用于音频-视觉语音识别（Audio-Visual Speech Recognition，AVSR），以更好地利用视频中的长时上下文信息。

<p align="center">
<img src="fig/lcbnet1.png" alt="AVSR整体流程框架"  width="800" />
<p align="center">
<img src="fig/lcbnet2.png" alt="LCB-NET模型结构"  width="800" />


具体来说，我们首先使用OCR技术来检测和识别幻灯片中的文本内容，其次我们采用关键词提取技术来获取文本内容中的关键词短语。最后，我们将关键词拼接成长上下文文本和音频同时输入到我们的LCB-net模型中进行识别。而LCB-net模型采用了双编码器结构，同时建模音频和长上下文文本信息。此外，我们还引入了一个显式的偏置词预测模块，通过使用二元交叉熵（BCE）损失函数显式预测长上下文文本中在音频中出现的关键偏置词。此外，为增强LCB-net的泛化能力和稳健性，我们还采用了动态的关键词模拟策略。实验证明，我们提出的LCB-net热词模型，不仅能够提升关键词的识别效果，同时也能够提升非关键词的识别效果。具体实验结果如下所示：

<p align="center">
<img src="fig/lcbnet3.png" alt="实验结果"  width="500" />


更详细的细节见：
- 论文： [LCB-net: Long-Context Biasing for Audio-Visual Speech Recognition](https://arxiv.org/abs/2401.06390)





## 相关论文以及引用信息

```BibTeX
@inproceedings{yu2024lcbnet,
  title={LCB-net: Long-Context Biasing for Audio-Visual Speech Recognition},
  author={Fan Yu, Haoxu Wang, Xian Shi, Shiliang Zhang},
  booktitle={ICASSP},
  year={2024}
}
```