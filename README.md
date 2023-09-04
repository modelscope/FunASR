[//]: # (<div align="left"><img src="docs/images/funasr_logo.jpg" width="400"/></div>)

([简体中文](./README_zh.md)|English)

# FunASR: A Fundamental End-to-End Speech Recognition Toolkit
<p align="left">
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-brightgreen.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Pytorch-%3E%3D1.11-blue"></a>
</p>

<strong>FunASR</strong> hopes to build a bridge between academic research and industrial applications on speech recognition. By supporting the training & finetuning of the industrial-grade speech recognition model released on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition), researchers and developers can conduct research and production of speech recognition models more conveniently, and promote the development of speech recognition ecology. ASR for Fun！

[**Highlights**](#highlights)
| [**News**](https://github.com/alibaba-damo-academy/FunASR#whats-new) 
| [**Installation**](#installation)
| [**Quick Start**](#quick-start)
| [**Runtime**](./funasr/runtime/readme.md)
| [**Model Zoo**](./docs/model_zoo/modelscope_models.md)
| [**Contact**](#contact)


<a name="highlights"></a>
## Highlights
- FunASR is a fundamental speech recognition toolkit that offers a variety of features, including speech recognition (ASR), Voice Activity Detection (VAD), Punctuation Restoration, Language Models, Speaker Verification, Speaker Diarization and multi-talker ASR. FunASR provides convenient scripts and tutorials, supporting inference and fine-tuning of pre-trained models.
- We have released a vast collection of academic and industrial pretrained models on the [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition), which can be accessed through our [Model Zoo](https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/model_zoo/modelscope_models.md). The representative [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary), a non-autoregressive end-to-end speech recognition model, has the advantages of high accuracy, high efficiency, and convenient deployment, supporting the rapid construction of speech recognition services. For more details on service deployment, please refer to the [service deployment document](funasr/runtime/readme_cn.md). 


<a name="whats-new"></a>
## What's new: 
- 2023/09/01: The offline file transcription service 2.0 (CPU) of Mandarin has been released, with added support for ffmpeg, timestamp, and hotword models. For more details, please refer to ([Deployment documentation](funasr/runtime/docs/SDK_tutorial.md)).
- 2023/08/07: The real-time transcription service (CPU) of Mandarin has been released. For more details, please refer to ([Deployment documentation](funasr/runtime/docs/SDK_tutorial_online.md)).
- 2023/07/17: BAT is released, which is a low-latency and low-memory-consumption RNN-T model. For more details, please refer to ([BAT](egs/aishell/bat)).
- 2023/07/03: The offline file transcription service (CPU) of Mandarin has been released. For more details, please refer to ([Deployment documentation](funasr/runtime/docs/SDK_tutorial.md)).
- 2023/06/26: ASRU2023 Multi-Channel Multi-Party Meeting Transcription Challenge 2.0 completed the competition and announced the results. For more details, please refer to ([M2MeT2.0](https://alibaba-damo-academy.github.io/FunASR/m2met2/index.html)).


<a name="Installation"></a>
## Installation

Please ref to [installation docs](https://alibaba-damo-academy.github.io/FunASR/en/installation/installation.html)

## Deployment Service

FunASR supports pre-trained or further fine-tuned models for deployment as a service. The CPU version of the Chinese offline file conversion service has been released, details can be found in [docs](funasr/runtime/docs/SDK_tutorial.md). More detailed information about service deployment can be found in the [deployment roadmap](funasr/runtime/readme_cn.md).


<a name="quick-start"></a>
## Quick Start
Quick start for new users（[tutorial](https://alibaba-damo-academy.github.io/FunASR/en/funasr/quick_start.html)）


FunASR supports inference and fine-tuning of models trained on industrial datasets of tens of thousands of hours. For more details, please refer to ([modelscope_egs](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_pipeline/quick_start.html)). It also supports training and fine-tuning of models on academic standard datasets. For more details, please refer to([egs](https://alibaba-damo-academy.github.io/FunASR/en/academic_recipe/asr_recipe.html)). The models include speech recognition (ASR), speech activity detection (VAD), punctuation recovery, language model, speaker verification, speaker separation, and multi-party conversation speech recognition. For a detailed list of models, please refer to the [Model Zoo](https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/model_zoo/modelscope_models.md):

<a name="Community Communication"></a>
## Community Communication
If you encounter problems in use, you can directly raise Issues on the github page.

You can also scan the following DingTalk group or WeChat group QR code to join the community group for communication and discussion.

|DingTalk group |                     WeChat group                      |
|:---:|:-----------------------------------------------------:|
|<div align="left"><img src="docs/images/dingding.jpg" width="250"/> | <img src="docs/images/wechat.png" width="232"/></div> |

## Contributors

| <div align="left"><img src="docs/images/damo.png" width="180"/> | <div align="left"><img src="docs/images/nwpu.png" width="260"/> | <img src="docs/images/China_Telecom.png" width="200"/> </div>  | <img src="docs/images/RapidAI.png" width="200"/> </div> | <img src="docs/images/aihealthx.png" width="200"/> </div> | <img src="docs/images/XVERSE.png" width="250"/> </div> |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|:--------------------------------------------------------------:|:-------------------------------------------------------:|:-----------------------------------------------------------:|:------------------------------------------------------:|

The contributors can be found in [contributors list](./Acknowledge.md)

## License
This project is licensed under the [The MIT License](https://opensource.org/licenses/MIT). FunASR also contains various third-party components and some code modified from other repos under other open source licenses.
The use of pretraining model is subject to [model license](./MODEL_LICENSE)


## Citations
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
@inproceedings{wang2023told,
  author={Jiaming Wang and Zhihao Du and Shiliang Zhang},
  title={{TOLD:} {A} Novel Two-Stage Overlap-Aware Framework for Speaker Diarization},
  year={2023},
  booktitle={ICASSP},
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
