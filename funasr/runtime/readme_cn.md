# FunASR runtime-SDK

English Version（[docs](./readme.md)）

FunASR是由达摩院语音实验室开源的一款语音识别基础框架，集成了语音端点检测、语音识别、标点断句等领域的工业级别模型，吸引了众多开发者参与体验和开发。为了解决工业落地的最后一公里，将模型集成到业务中去，我们开发了FunASR runtime-SDK。
SDK 支持以下几种服务部署：

- 中文离线文件转写服务（CPU版本），已完成
- 中文离线文件转写服务（GPU版本），进行中
- 英文离线转写服务，进行中
- 流式语音识别服务，进行中
- 。。。


## 中文离线文件转写服务部署（CPU版本）

中文语音离线文件服务部署（CPU版本），拥有完整的语音识别链路，可以将几十个小时的音频识别成带标点的文字，而且支持上百路并发同时进行识别。
为了支持不同用户的需求，针对不同场景，准备了不同的图文教程：

### 便捷部署教程

适用场景为，对服务部署SDK无修改需求，部署模型来自于ModelScope，或者用户finetune，详细教程参考（[点击此处](./docs/SDK_tutorial_zh.md)）

### 技术原理揭秘

文档介绍了背后技术原理，识别准确率，计算效率等，以及核心优势介绍：便捷、高精度、高效率、长音频链路，详细文档参考（[点击此处](https://mp.weixin.qq.com/s?__biz=MzA3MTQ0NTUyMw==&tempkey=MTIyNF84d05USjMxSEpPdk5GZXBJUFNJNzY0bU1DTkxhV19mcWY4MTNWQTJSYXhUaFgxOWFHZTZKR0JzWC1JRmRCdUxCX2NoQXg0TzFpNmVJX2R1WjdrcC02N2FEcUc3MDhzVVhpNWQ5clU4QUdqNFdkdjFYb18xRjlZMmc5c3RDOTl0U0NiRkJLb05ZZ0RmRlVkVjFCZnpXNWFBVlRhbXVtdWs4bUMwSHZnfn4%3D&chksm=1f2c3254285bbb42bc8f76a82e9c5211518a0bb1ff8c357d085c1b78f675ef2311f3be6e282c#rd)）

### 开发指南

适用场景为，对服务部署SDK有修改需求，部署模型来自于ModelScope，或者用户finetune，详细文档参考（[点击此处](./docs/SDK_advanced_guide_offline_cn.md)）

