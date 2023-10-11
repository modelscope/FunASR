# FunASR软件包路线图

English Version（[docs](./readme.md)）

FunASR是由达摩院语音实验室开源的一款语音识别基础框架，集成了语音端点检测、语音识别、标点断句等领域的工业级别模型，吸引了众多开发者参与体验和开发。为了解决工业落地的最后一公里，将模型集成到业务中去，我们开发了FunASR runtime-SDK。
SDK 支持以下几种服务部署：

- 中文离线文件转写服务（CPU版本），已完成
- 中文流式语音识别服务（CPU版本），已完成
- 英文离线文件转写服务（CPU版本），已完成
- 中文离线文件转写服务（GPU版本），进行中
- 更多支持中

## 英文离线文件转写服务（CPU版本）

英文离线文件转写服务部署（CPU版本），拥有完整的语音识别链路，可以将几十个小时的长音频与视频识别成带标点的文字，而且支持上百路请求同时进行转写。
为了支持不同用户的需求，针对不同场景，准备了不同的图文教程：

### 便捷部署教程

适用场景为，对服务部署SDK无修改需求，部署模型来自于ModelScope，或者用户finetune，详细教程参考（[点击此处](./docs/SDK_tutorial_en_zh.md)）


### 开发指南

适用场景为，对服务部署SDK有修改需求，部署模型来自于ModelScope，或者用户finetune，详细文档参考（[点击此处](./docs/SDK_advanced_guide_offline_en_zh.md)）

### 技术原理揭秘

文档介绍了背后技术原理，识别准确率，计算效率等，以及核心优势介绍：便捷、高精度、高效率、长音频链路，详细文档参考（[点击此处](https://mp.weixin.qq.com/s/DHQwbgdBWcda0w_L60iUww)）

### 最新版本及image ID
| image version                |  image ID | INFO |
|------------------------------|-----|------|
| funasr-runtime-sdk-en-cpu-0.1.0 |  4ce696fe9ba5   |      |


## 中文实时语音听写服务（CPU版本）

FunASR实时语音听写服务软件包，既可以实时地进行语音转文字，而且能够在说话句尾用高精度的转写文字修正输出，输出文字带有标点，支持高并发多路请求。
为了支持不同用户的需求，针对不同场景，准备了不同的图文教程：

### 便捷部署教程

适用场景为，对服务部署SDK无修改需求，部署模型来自于ModelScope，或者用户finetune，详细教程参考（[点击此处](./docs/SDK_tutorial_online_zh.md)）


### 开发指南

适用场景为，对服务部署SDK有修改需求，部署模型来自于ModelScope，或者用户finetune，详细文档参考（[点击此处](./docs/SDK_advanced_guide_online_zh.md)）

### 技术原理揭秘

文档介绍了背后技术原理，识别准确率，计算效率等，以及核心优势介绍：便捷、高精度、高效率、长音频链路，详细文档参考（[点击此处](https://mp.weixin.qq.com/s/8He081-FM-9IEI4D-lxZ9w)）

### 最新版本及image ID

| image version                       |  image ID | INFO |
|-------------------------------------|-----|------|
| funasr-runtime-sdk-online-cpu-0.1.2 |   7222c5319bcf  |      |


## 中文离线文件转写服务（CPU版本）

中文语音离线文件服务部署（CPU版本），拥有完整的语音识别链路，可以将几十个小时的长音频与视频识别成带标点的文字，而且支持上百路请求同时进行转写。
为了支持不同用户的需求，针对不同场景，准备了不同的图文教程：

### 便捷部署教程

适用场景为，对服务部署SDK无修改需求，部署模型来自于ModelScope，或者用户finetune，详细教程参考（[点击此处](./docs/SDK_tutorial_zh.md)）


### 开发指南

适用场景为，对服务部署SDK有修改需求，部署模型来自于ModelScope，或者用户finetune，详细文档参考（[点击此处](./docs/SDK_advanced_guide_offline_zh.md)）

### 技术原理揭秘

文档介绍了背后技术原理，识别准确率，计算效率等，以及核心优势介绍：便捷、高精度、高效率、长音频链路，详细文档参考（[点击此处](https://mp.weixin.qq.com/s/DHQwbgdBWcda0w_L60iUww)）

### 最新版本及image ID
| image version                |  image ID | INFO |
|------------------------------|-----|------|
| funasr-runtime-sdk-cpu-0.2.2 |  2c5286be13e9   |      |
