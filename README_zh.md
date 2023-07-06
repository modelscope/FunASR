[//]: # (<div align="left"><img src="docs/images/funasr_logo.jpg" width="400"/></div>)

(简体中文|[English](./README.md))

# FunASR: A Fundamental End-to-End Speech Recognition Toolkit
<p align="left">
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-brightgreen.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Pytorch-%3E%3D1.11-blue"></a>
</p>

FunASR希望在语音识别的学术研究和工业应用之间架起一座桥梁。通过支持在[ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition)上发布的工业级语音识别模型的训练和微调，研究人员和开发人员可以更方便地进行语音识别模型的研究和生产，并推动语音识别生态的发展。让语音识别更有趣！

[**最新动态**](#最新动态) 
| [**核心功能**](#核心功能)
| [**安装教程**](#安装教程)
| [**如何使用**](#如何使用)
| [**服务部署**](./funasr/runtime/readme_cn.md)
| [**模型仓库**](./docs/model_zoo/modelscope_models.md)
| [**联系我们**](#contact)
| [**M2MET2.0比赛**](https://github.com/alibaba-damo-academy/FunASR#multi-channel-multi-party-meeting-transcription-20-m2met20-challenge)

## 最新动态

### 服务部署SDK

- 2023.07.03: 
中文离线文件转写服务（CPU版本）发布，支持一键部署和测试([点击此处](funasr/runtime/readme_cn.md))

### ASRU 2023 多通道多方会议转录挑战 2.0

详情请参考文档（[点击此处](https://alibaba-damo-academy.github.io/FunASR/m2met2_cn/index.html)）


### 学术模型更新

### 工业模型更新

- 2023/07/06 

## 核心功能
- FunASR是一个基础语音识别工具包，提供多种功能，包括语音识别（ASR）、语音活动检测（VAD）、标点恢复、语言模型、说话人验证、说话人分离和多人对话语音识别。
- 我们在[ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition)上发布了大量的学术和工业预训练模型，可以通过我们的[模型仓库](https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/model_zoo/modelscope_models.md)访问。代表性的[Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)模型在许多语音识别任务中实现了SOTA性能。
- FunASR提供了一个易于使用的接口，可以直接基于ModelScope中托管模型进行推理与微调。此外，FunASR中的优化数据加载器可以加速大规模数据集的训练速度。

## 安装教程

直接安装发布软件包

```shell
pip3 install -U funasr
# 中国大陆用户，如果遇到网络问题，可以用下面指令:
# pip3 install -U funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

您也可以从源码安装


``` sh
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip3 install -e ./
# 中国大陆用户，如果遇到网络问题，可以用下面指令:
# pip3 install -e ./ -i https://mirror.sjtu.edu.cn/pypi/web/simple
```
如果您需要使用ModelScope中发布的预训练模型，需要安装ModelScope

```shell
pip3 install -U modelscope
# 中国大陆用户，如果遇到网络问题，可以用下面指令:
# pip3 install -U modelscope -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

更详细安装过程介绍（[点击此处](https://alibaba-damo-academy.github.io/FunASR/en/installation/installation.html)）

## 如何使用

您可以通过如下几种方式使用FunASR功能:

- 服务部署SDK
- 工业模型egs
- 学术模型egs

### 服务部署SDK

#### python版本示例

支持实时流式语音识别，并且会用非流式模型进行纠错，输出文本带有标点。目前只支持单个client，如需多并发请参考c++版本服务部署SDK（[点击此处]()）

##### 服务端部署
```shell
cd funasr/runtime/python/websocket
python funasr_wss_server.py --port 10095
```

##### 客户端测试
```shell
python funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode 2pass --chunk_size "5,10,5"
#python funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode 2pass --chunk_size "8,8,4" --audio_in "./data/wav.scp"
```
更多例子可以参考（[点击此处](https://alibaba-damo-academy.github.io/FunASR/en/runtime/websocket_python.html#id2)）

#### c++版本示例

目前已支持离线文件转写服务（CPU），支持上百路并发请求

##### 服务端部署
可以用个下面指令，一键部署完成部署
```shell
curl -O https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/shell/funasr-runtime-deploy-offline-cpu-zh.sh
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh install --workspace ./funasr-runtime-resources
```

##### 客户端测试

```shell
python3 funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "../audio/asr_example.wav"
```
更多例子参考（[点击此处](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/docs/SDK_tutorial_zh.md)）


### 工业模型egs

如果您希望使用ModelScope中预训练好的工业模型，进行推理或者微调训练，您可以参考下面指令：


```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
)

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
print(rec_result)
# {'text': '欢迎大家来体验达摩院推出的语音识别模型'}
```

更多例子可以参考（[点击此处](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_pipeline/quick_start.html)）


### 学术模型egs

如果您希望从头开始训练，通常为学术模型，您可以通过下面的指令启动训练与推理：

```shell
cd egs/aishell/paraformer
. ./run.sh --CUDA_VISIBLE_DEVICES="0,1" --gpu_num=2
```

更多例子可以参考（[点击此处](https://alibaba-damo-academy.github.io/FunASR/en/academic_recipe/asr_recipe.html)）


## 联系我们

如果您在使用中遇到困难，可以通过一下方式联系我们

- email: [funasr@list.alibaba-inc.com](funasr@list.alibaba-inc.com)

|                                  钉钉群                                  |                          微信                           |
|:---------------------------------------------------------------------:|:-----------------------------------------------------:|
| <div align="left"><img src="docs/images/dingding.jpg" width="250"/>   | <img src="docs/images/wechat.png" width="232"/></div> |

## 社区贡献者

| <div align="left"><img src="docs/images/damo.png" width="180"/> | <div align="left"><img src="docs/images/nwpu.png" width="260"/> | <img src="docs/images/China_Telecom.png" width="200"/> </div>  | <img src="docs/images/RapidAI.png" width="200"/> </div> | <img src="docs/images/aihealthx.png" width="200"/> </div> |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|:--------------------------------------------------------------:|:-------------------------------------------------------:|:-----------------------------------------------------------:|

贡献者名单请参考（[点击此处](./Acknowledge)）


## 许可协议
项目遵循[The MIT License](https://opensource.org/licenses/MIT)开源协议. 工业模型许可协议请参考（[点击此处](./MODEL_LICENSE)）


## Stargazers over time

[![Stargazers over time](https://starchart.cc/alibaba-damo-academy/FunASR.svg)](https://starchart.cc/alibaba-damo-academy/FunASR)

## 论文引用

``` bibtex
@inproceedings{gao2023funasr,
  author={Zhifu Gao and Zerui Li and Jiaming Wang and Haoneng Luo and Xian Shi and Mengzhe Chen and Yabin Li and Lingyun Zuo and Zhihao Du and Zhangyu Xiao and Shiliang Zhang},
  title={FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
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
