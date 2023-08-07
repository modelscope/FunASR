[//]: # (<div align="left"><img src="docs/images/funasr_logo.jpg" width="400"/></div>)

(简体中文|[English](./README.md))

# FunASR: A Fundamental End-to-End Speech Recognition Toolkit
<p align="left">
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-brightgreen.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Pytorch-%3E%3D1.11-blue"></a>
</p>

FunASR希望在语音识别的学术研究和工业应用之间架起一座桥梁。通过支持在[ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition)上发布的工业级语音识别模型的训练和微调，研究人员和开发人员可以更方便地进行语音识别模型的研究和生产，并推动语音识别生态的发展。让语音识别更有趣！

<div align="center">  
<h4>
 <a href="#核心功能"> 核心功能 </a>   
｜<a href="#最新动态"> 最新动态 </a>
｜<a href="#安装教程"> 安装 </a>
｜<a href="#快速开始"> 快速开始 </a>
｜<a href="https://alibaba-damo-academy.github.io/FunASR/en/index.html"> 教程文档 </a>
｜<a href="./docs/model_zoo/modelscope_models.md"> 模型仓库 </a>
｜<a href="./funasr/runtime/readme_cn.md"> 服务部署 </a>
｜<a href="#联系我们"> 联系我们 </a>
</h4>
</div>

<a name="核心功能"></a>
## 核心功能
- FunASR是一个基础语音识别工具包，提供多种功能，包括语音识别（ASR）、语音端点检测（VAD）、标点恢复、语言模型、说话人验证、说话人分离和多人对话语音识别等。FunASR提供了便捷的脚本和教程，支持预训练好的模型的推理与微调。
- 我们在[ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition)上发布了大量开源数据集或者海量工业数据训练的模型，可以通过我们的[模型仓库](https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/model_zoo/modelscope_models.md)了解模型的详细信息。代表性的[Paraformer](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)非自回归端到端语音识别模型具有高精度、高效率、便捷部署的优点，支持快速构建语音识别服务，详细信息可以阅读([服务部署文档](funasr/runtime/readme_cn.md))。

<a name="最新动态"></a>
## 最新动态
- 2023.08.07: 中文实时语音转写服务一键部署的CPU版本发布，详细信息参阅([一键部署文档](funasr/runtime/docs/SDK_tutorial_online_zh.md))
- 2023.07.17: BAT一种低延迟低内存消耗的RNN-T模型发布，详细信息参阅（[BAT](egs/aishell/bat)）
- 2023.07.03: 中文离线文件转写服务一键部署的CPU版本发布，详细信息参阅([一键部署文档](funasr/runtime/docs/SDK_tutorial_zh.md))
- 2023.06.26: ASRU2023 多通道多方会议转录挑战赛2.0完成竞赛结果公布，详细信息参阅（[M2MeT2.0](https://alibaba-damo-academy.github.io/FunASR/m2met2_cn/index.html)）

<a name="安装教程"></a>
## 安装教程
FunASR安装教程请阅读（[Installation](https://alibaba-damo-academy.github.io/FunASR/en/installation/installation.html)）

<a name="服务部署"></a>
## 服务部署
FunASR支持预训练或者进一步微调的模型进行服务部署。目前中文离线文件转写服务一键部署的CPU版本已经发布，详细信息参阅([一键部署文档](funasr/runtime/docs/SDK_tutorial_zh.md)。更多服务部署详细信息可以参阅([服务部署文档](funasr/runtime/readme_cn.md))。

<a name="快速开始"></a>
## 快速开始
快速使用教程（[新人文档](https://alibaba-damo-academy.github.io/FunASR/en/funasr/quick_start_zh.html)）

FunASR支持数万小时工业数据训练的模型的推理和微调，详细信息可以参阅（[modelscope_egs](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_pipeline/quick_start.html)）；也支持学术标准数据集模型的训练和微调，详细信息可以参阅（[egs](https://alibaba-damo-academy.github.io/FunASR/en/academic_recipe/asr_recipe.html)）。 模型包含语音识别（ASR）、语音活动检测（VAD）、标点恢复、语言模型、说话人验证、说话人分离和多人对话语音识别等，详细模型列表可以参阅[模型仓库](https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/model_zoo/modelscope_models.md)：

<a name="社区交流"></a>
## 联系我们

如果您在使用中遇到问题，可以直接在github页面提Issues。欢迎语音兴趣爱好者扫描以下的钉钉群或者微信群二维码加入社区群，进行交流和讨论。
|                                  钉钉群                                  |                          微信                           |
|:---------------------------------------------------------------------:|:-----------------------------------------------------:|
| <div align="left"><img src="docs/images/dingding.jpg" width="250"/>   | <img src="docs/images/wechat.png" width="232"/></div> |

## 社区贡献者

| <div align="left"><img src="docs/images/damo.png" width="180"/> | <div align="left"><img src="docs/images/nwpu.png" width="260"/> | <img src="docs/images/China_Telecom.png" width="200"/> </div>  | <img src="docs/images/RapidAI.png" width="200"/> </div> | <img src="docs/images/aihealthx.png" width="200"/> </div> |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|:--------------------------------------------------------------:|:-------------------------------------------------------:|:-----------------------------------------------------------:|

贡献者名单请参考（[致谢名单](./Acknowledge)）


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
