# FunASR Runtime Roadmap
中文文档（[点击此处](./readme_cn.md)）

FunASR is a speech recognition framework developed by the Speech Lab of DAMO Academy, which integrates industrial-level models in the fields of speech endpoint detection, speech recognition, punctuation segmentation, and more. 
It has attracted many developers to participate in experiencing and developing. To solve the last mile of industrial landing and integrate models into business, we have developed the FunASR runtime-SDK. The SDK supports several service deployments, including:

- File transcription service, Mandarin, CPU version, done
- The real-time transcription service, Mandarin (CPU), done
- File transcription service, English, CPU version, done
- File transcription service, Mandarin, GPU version, done
- and more.

## File Transcription Service, Mandarin (GPU)

Currently, the FunASR runtime-SDK supports the deployment of file transcription service, Mandarin (GPU version), with a complete speech recognition chain that can transcribe tens of hours of audio into punctuated text, and supports recognition for more than a hundred concurrent streams. 

To meet the needs of different users, we have prepared different tutorials with text and images for both novice and advanced developers.

### Whats-new
- 2024/09/26: File Transcription Service 2.0 GPU released, Fix GPU memory leak, docker image version funasr-runtime-sdk-gpu-0.2.0 (d280bf7e495b)
- 2024/07/01: File Transcription Service 1.1 GPU released, optimize BladeDISC model compatibility issues, docker image version funasr-runtime-sdk-gpu-0.1.1 (8875cbf9b99e)
- 2024/06/27: File Transcription Service 1.0 GPU released, supporting dynamic batch processing and multi-threading concurrency. In the long audio test set, the single-thread RTF is 0.0076, and multi-threads' speedup is 1200+ (compared to 330+ on CPU), ref to([docs](./docs/benchmark_libtorch_cpp.md)) , docker image version funasr-runtime-sdk-gpu-0.1.0 (b86066f4d018)

### Advanced Development Guide

The documentation mainly targets advanced developers who require modifications and customization of the service. It supports downloading model deployments from modelscope and also supports deploying models that users have fine-tuned. For detailed information, please refer to the documentation available by [docs](./docs/SDK_advanced_guide_offline_gpu.md)


## File Transcription Service, English (CPU)

Currently, the FunASR runtime-SDK supports the deployment of file transcription service, English (CPU version), with a complete speech recognition chain that can transcribe tens of hours of audio into punctuated text, and supports recognition for more than a hundred concurrent streams. 

To meet the needs of different users, we have prepared different tutorials with text and images for both novice and advanced developers.

### Whats-new
- 2024/09/26: Fix memory leak, docker image version funasr-runtime-sdk-en-cpu-0.1.7 (f6c5a7b59eb6).
- 2024/05/15: Adapting to FunASR 1.0 model structure, docker image version funasr-runtime-sdk-en-cpu-0.1.6 (84d781d07997).
- 2024/03/05: docker image supports ARM64 platform, update modelscope, docker image version funasr-runtime-sdk-en-cpu-0.1.5 (7cca2abc5901).
- 2024/01/25: Optimized the VAD (Voice Activity Detection) data processing method,significantly reducing peak memory usage,memory leak optimization, docker image version funasr-runtime-sdk-en-cpu-0.1.3 (c00f9ce7a195).
- 2024/01/03: Fixed known crash issues as well as memory leak problems, docker image version funasr-runtime-sdk-en-cpu-0.1.2 (0cdd9f4a4bb5).
- 2023/11/08: Adaptation to runtime structure changes (FunASR/funasr/runtime -> FunASR/runtime), docker image version funasr-runtime-sdk-en-cpu-0.1.1 (27017f70f72a).
- 2023/10/16: English File Transcription Service 1.0 released, docker image version funasr-runtime-sdk-en-cpu-0.1.0 (e0de03eb0163), refer to the detailed documentation（[here](https://mp.weixin.qq.com/s/DZZUTj-6xwFfi-96ml--4A)）

### Technical Principles

The technical principles and documentation behind FunASR explain the underlying technology, recognition accuracy, computational efficiency, and core advantages of the framework, including convenience, high precision, high efficiency, and support for long audio chains. For detailed information, please refer to the documentation available by [docs](https://mp.weixin.qq.com/s/DZZUTj-6xwFfi-96ml--4A). 

### Deployment Tutorial

The documentation mainly targets novice users who have no need for modifications or customization. It supports downloading model deployments from modelscope and also supports deploying models that users have fine-tuned. For detailed tutorials, please refer to [docs](docs/SDK_tutorial_en.md).

### Advanced Development Guide

The documentation mainly targets advanced developers who require modifications and customization of the service. It supports downloading model deployments from modelscope and also supports deploying models that users have fine-tuned. For detailed information, please refer to the documentation available by [docs](./docs/SDK_advanced_guide_offline_en.md)


## The real-time transcription service, Mandarin (CPU)

The FunASR real-time speech-to-text service software package not only performs real-time speech-to-text conversion, but also allows high-precision transcription text correction at the end of each sentence and outputs text with punctuation, supporting high-concurrency multiple requests.
In order to meet the needs of different users for different scenarios, different tutorials are prepared:

### Whats-new+
- 2024/10/29: Real-time Transcription Service 1.12 released，The 2pass-offline mode supports the SensevoiceSmal model, docker image version funasr-runtime-sdk-online-cpu-0.1.12 (f5febc5cf13a)
- 2024/09/26: Real-time Transcription Service 1.11 released，Fix memory leak, docker image version funasr-runtime-sdk-online-cpu-0.1.11 (e51a36c42771)
- 2024/05/15: Real-time Transcription Service 1.10 released，adapting to FunASR 1.0 model structure, docker image version funasr-runtime-sdk-online-cpu-0.1.10 (1c2adfcff84d)
- 2024/03/05: Real-time Transcription Service 1.9 released，docker image supports ARM64 platform, update modelscope, docker image version funasr-runtime-sdk-online-cpu-0.1.9 (4a875e08c7a2)
- 2024/01/25: Real-time Transcription Service 1.7 released，optimization of the client-side, docker image version funasr-runtime-sdk-online-cpu-0.1.7 (2aa23805572e)
- 2024/01/03: Real-time Transcription Service 1.6 released，The 2pass-offline mode supports Ngram language model decoding and WFST hotwords, while also addressing known crash issues and memory leak problems, docker image version funasr-runtime-sdk-online-cpu-0.1.6 (f99925110d27)
- 2023/11/09: Real-time Transcription Service 1.5 released，fix bug: without online results, docker image version funasr-runtime-sdk-online-cpu-0.1.5 (b16584b6d38b)
- 2023/11/08: Real-time Transcription Service 1.4 released, supporting server-side loading of hotwords (updated hotword communication protocol), adaptation to runtime structure changes (FunASR/funasr/runtime -> FunASR/runtime), docker image version funasr-runtime-sdk-online-cpu-0.1.4(691974017c38).
- 2023/09/19: Real-time Transcription Service 1.2 released, supporting hotwords, timestamps, and ITN model in 2pass mode, docker image version funasr-runtime-sdk-online-cpu-0.1.2 (7222c5319bcf).
- 2023/08/11: Real-time Transcription Service 1.1 released, addressing some known bugs (including server crashes), docker image version funasr-runtime-sdk-online-cpu-0.1.1 (bdbdd0b27dee).
- 2023/08/07: Real-time Transcription Service 1.0 released, docker image version funasr-runtime-sdk-online-cpu-0.1.0(bdbdd0b27dee), refer to the detailed documentation（[here](https://mp.weixin.qq.com/s/8He081-FM-9IEI4D-lxZ9w)）

### Convenient Deployment Tutorial

This is suitable for scenarios where there is no need to modify the service deployment SDK and the deployed model comes from ModelScope or is finetuned by the user. For detailed tutorials, please refer to [docs](./docs/SDK_tutorial_online.md)


### Development Guide

This is suitable for scenarios where there is a need to modify the service deployment SDK and the deployed model comes from ModelScope or is finetuned by the user. For detailed documentation, please refer to [docs](./docs/SDK_advanced_guide_online.md)

### Technology Principles Revealed

The document introduces the technology principles behind the service, recognition accuracy, computing efficiency, and core advantages: convenience, high precision, high efficiency, and long audio chain. For detailed documentation, please refer to [docs](https://mp.weixin.qq.com/s/8He081-FM-9IEI4D-lxZ9w).


## File Transcription Service, Mandarin (CPU)

Currently, the FunASR runtime-SDK supports the deployment of file transcription service, Mandarin (CPU version), with a complete speech recognition chain that can transcribe tens of hours of audio into punctuated text, and supports recognition for more than a hundred concurrent streams. 

To meet the needs of different users, we have prepared different tutorials with text and images for both novice and advanced developers.

### Whats-new
- 2024/09/26: File Transcription Service 4.6 released, Fix memory leak & Support the SensevoiceSmall onnx model, docker image version funasr-runtime-sdk-cpu-0.4.6 (8651c6b8a1ae)
- 2024/05/15: File Transcription Service 4.5 released, adapting to FunASR 1.0 model structure, docker image version funasr-runtime-sdk-cpu-0.4.5 (058b9882ae67)
- 2024/03/05: File Transcription Service 4.4 released, docker image supports ARM64 platform, update modelscope, docker image version funasr-runtime-sdk-cpu-0.4.4 (2dc87b86dc49)
- 2024/01/25: File Transcription Service 4.2 released, optimized the VAD (Voice Activity Detection) data processing method, significantly reducing peak memory usage, memory leak optimization, docker image version funasr-runtime-sdk-cpu-0.4.2 (befdc7b179ed)
- 2024/01/08: File Transcription Service 4.1 released, optimized format sentence-level timestamps, docker image version funasr-runtime-sdk-cpu-0.4.1 (0250f8ef981b)
- 2024/01/03: File Transcription Service 4.0 released, Added support for 8k models, optimized timestamp mismatch issues and added sentence-level timestamps, improved the effectiveness of English word FST hotwords, supported automated configuration of thread parameters, and fixed known crash issues as well as memory leak problems, docker image version funasr-runtime-sdk-cpu-0.4.0 (c4483ee08f04)
- 2023/11/08: File Transcription Service 3.0 released, supporting punctuation large model, Ngram model, fst hotwords (updated hotword communication protocol), server-side loading of hotwords, adaptation to runtime structure changes (FunASR/funasr/runtime -> FunASR/runtime), docker image version funasr-runtime-sdk-cpu-0.3.0 (caa64bddbb43), refer to the detailed documentation （[here]()）
- 2023/09/19: File Transcription Service 2.2 released, supporting ITN model, docker image version funasr-runtime-sdk-cpu-0.2.2 (2c5286be13e9).
- 2023/08/22: File Transcription Service 2.0 released, integrated ffmpeg to support various audio and video inputs, supporting hotword model and timestamp model, docker image version funasr-runtime-sdk-cpu-0.2.0 (1ad3d19e0707), refer to the detailed documentation （[here](https://mp.weixin.qq.com/s/oJHe0MKDqTeuIFH-F7GHMg)）
- 2023/07/03: File Transcription Service 1.0 released, docker image version funasr-runtime-sdk-cpu-0.1.0 (1ad3d19e0707), refer to the detailed documentation （[here](https://mp.weixin.qq.com/s/DHQwbgdBWcda0w_L60iUww)）

### Technical Principles

The technical principles and documentation behind FunASR explain the underlying technology, recognition accuracy, computational efficiency, and core advantages of the framework, including convenience, high precision, high efficiency, and support for long audio chains. For detailed information, please refer to the documentation available by [docs](https://mp.weixin.qq.com/s/DHQwbgdBWcda0w_L60iUww). 

### Deployment Tutorial

The documentation mainly targets novice users who have no need for modifications or customization. It supports downloading model deployments from modelscope and also supports deploying models that users have fine-tuned. For detailed tutorials, please refer to [docs](docs/SDK_tutorial.md).

### Advanced Development Guide

The documentation mainly targets advanced developers who require modifications and customization of the service. It supports downloading model deployments from modelscope and also supports deploying models that users have fine-tuned. For detailed information, please refer to the documentation available by [docs](./docs/SDK_advanced_guide_offline.md)

