# FunASR runtime-SDK
中文文档（[点击此处](./readme_cn.md)）

FunASR is a speech recognition framework developed by the Speech Lab of DAMO Academy, which integrates industrial-level models in the fields of speech endpoint detection, speech recognition, punctuation segmentation, and more. 
It has attracted many developers to participate in experiencing and developing. To solve the last mile of industrial landing and integrate models into business, we have developed the FunASR runtime-SDK. The SDK supports several service deployments, including:

- File transcription service, Mandarin, CPU version, done
- File transcription service, Mandarin, GPU version, in progress
- File transcription service, English, in progress
- Streaming speech recognition service, is in progress
- and more.


## File Transcription Service, Mandarin (CPU)

Currently, the FunASR runtime-SDK-0.0.1 version supports the deployment of file transcription service, Mandarin (CPU version), with a complete speech recognition chain that can transcribe tens of hours of audio into punctuated text, and supports recognition for more than a hundred concurrent streams. 

To meet the needs of different users, we have prepared different tutorials with text and images for both novice and advanced developers.

### Technical Principles

The technical principles and documentation behind FunASR explain the underlying technology, recognition accuracy, computational efficiency, and core advantages of the framework, including convenience, high precision, high efficiency, and support for long audio chains. For detailed information, please refer to the documentation available by [docs](). 

### Deployment Tutorial

The documentation mainly targets novice users who have no need for modifications or customization. It supports downloading model deployments from modelscope and also supports deploying models that users have fine-tuned. For detailed tutorials, please refer to [docs](docs/SDK_tutorial.md).

### Advanced Development Guide

The documentation mainly targets advanced developers who require modifications and customization of the service. It supports downloading model deployments from modelscope and also supports deploying models that users have fine-tuned. For detailed information, please refer to the documentation available by [docs](websocket/readme.md)
