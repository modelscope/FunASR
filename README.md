[//]: # (<div align="left"><img src="docs/images/funasr_logo.jpg" width="400"/></div>)

([简体中文](./README_zh.md)|English)

# FunASR: A Fundamental End-to-End Speech Recognition Toolkit
<p align="left">
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-brightgreen.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Pytorch-%3E%3D1.11-blue"></a>
</p>

<strong>FunASR</strong> hopes to build a bridge between academic research and industrial applications on speech recognition. By supporting the training & finetuning of the industrial-grade speech recognition model released on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition), researchers and developers can conduct research and production of speech recognition models more conveniently, and promote the development of speech recognition ecology. ASR for Fun！

[**News**](https://github.com/alibaba-damo-academy/FunASR#whats-new) 
| [**Highlights**](#highlights)
| [**Installation**](#installation)
| [**Quick Start**](#quick-start)
| [**Runtime**](./funasr/runtime/readme.md)
| [**Model Zoo**](./docs/model_zoo/modelscope_models.md)
| [**Contact**](#contact)


<a name="whats-new"></a>
## What's new: 

### FunASR runtime

- 2023.07.03: 
We have release the FunASR runtime-SDK-0.1.0, file transcription service (Mandarin) is now supported ([ZH](funasr/runtime/readme_cn.md)/[EN](funasr/runtime/readme.md))

### Multi-Channel Multi-Party Meeting Transcription 2.0 (M2MeT2.0) Challenge

Challenge details ref to ([CN](https://alibaba-damo-academy.github.io/FunASR/m2met2_cn/index.html)/[EN](https://alibaba-damo-academy.github.io/FunASR/m2met2/index.html))

### Speech Recognition
 
- Academic Models
  - Encoder-Decoder Models (AED): [Transformer](egs/aishell/transformer), [Conformer](egs/aishell/conformer), [Branchformer](egs/aishell/branchformer)
  - Transducer Models (RNNT): [RNNT streaming](egs/aishell/rnnt), [BAT streaming/non-streaming](egs/aishell/bat)
  - Non-autoregressive Model (NAR): [Paraformer](egs/aishell/paraformer)
  - Multi-speaker recognition model: [MFCCA](egs_modelscope/asr/mfcca)


- Industrial-level Models
  - Paraformer Models (Mandarin): [Paraformer-large](egs_modelscope/asr/paraformer/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch), [Paraformer-large-long](egs_modelscope/asr_vad_punc/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch), [Paraformer-large streaming](egs_modelscope/asr/paraformer/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online), [Paraformer-large-contextual](egs_modelscope/asr/paraformer/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404)
  - Conformer Models (English): [Conformer]()
  - UniASR streaming offline unifying models: [16k UniASR Burmese](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-my-16k-common-vocab696-pytorch/summary), [16k UniASR Hebrew](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-he-16k-common-vocab1085-pytorch/summary), [16k UniASR Urdu](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ur-16k-common-vocab877-pytorch/summary), [8k UniASR Mandarin financial domain](https://www.modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-8k-finance-vocab3445-online/summary), [16k UniASR Mandarin audio-visual domain](https://www.modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-16k-audio_and_video-vocab3445-online/summary),
  [Southern Fujian Dialect model](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-minnan-16k-common-vocab3825/summary), [French model](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-fr-16k-common-vocab3472-tensorflow1-online/summary),  [German model](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-de-16k-common-vocab3690-tensorflow1-online/summary),  [Vietnamese model](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-vi-16k-common-vocab1001-pytorch-online/summary),  [Persian model](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-fa-16k-common-vocab1257-pytorch-online/summary)

- Speaker Recognition
  - Speaker Verification Model: [xvector](egs_modelscope/speaker_verification)
  - Speaker Diarization Model: [SOND](egs/callhome/diarization/sond)

- Punctuation Restoration
  - Chinese Punctuation Model: [CT-Transformer](egs_modelscope/punctuation/punc_ct-transformer_zh-cn-common-vocab272727-pytorch), [CT-Transformer streaming](egs_modelscope/punctuation/punc_ct-transformer_zh-cn-common-vadrealtime-vocab272727)

- Endpoint Detection
  - [FSMN-VAD](egs_modelscope/vad/speech_fsmn_vad_zh-cn-16k-common)

- Timestamp Prediction
  - Character-level FA Model: [TP-Aligner](egs_modelscope/tp/speech_timestamp_prediction-v1-16k-offline)


<a name="highlights"></a>
## Highlights
- FunASR is a fundamental speech recognition toolkit that offers a variety of features, including speech recognition (ASR), Voice Activity Detection (VAD), Punctuation Restoration, Language Models, Speaker Verification, Speaker diarization and multi-talker ASR.
- We have released a vast collection of academic and industrial pretrained models on the [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition), which can be accessed through our [Model Zoo](https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/model_zoo/modelscope_models.md). The representative [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) model has achieved SOTA performance in many speech recognition tasks. 
- FunASR offers a user-friendly pipeline for fine-tuning pretrained models from the [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition). Additionally, the optimized dataloader in FunASR enables faster training speeds for large-scale datasets. This feature enhances the efficiency of the speech recognition process for researchers and practitioners.

<a name="Installation"></a>
## Installation

Install from pip
```shell
pip3 install -U funasr
# For the users in China, you could install with the command:
# pip3 install -U funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

Or install from source code


``` sh
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip3 install -e ./
# For the users in China, you could install with the command:
# pip3 install -e ./ -i https://mirror.sjtu.edu.cn/pypi/web/simple

```
If you want to use the pretrained models in ModelScope, you should install the modelscope:

```shell
pip3 install -U modelscope
# For the users in China, you could install with the command:
# pip3 install -U modelscope -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

For more details, please ref to [installation](https://alibaba-damo-academy.github.io/FunASR/en/installation/installation.html)

<a name="quick-start"></a>
## Quick Start

You can use FunASR in the following ways:

- Service Deployment SDK
- Industrial model egs
- Academic model egs

### Service Deployment SDK

#### Python version Example
Supports real-time streaming speech recognition, uses non-streaming models for error correction, and outputs text with punctuation. Currently, only single client is supported. For multi-concurrency, please refer to the C++ version service deployment SDK below.

##### Server Deployment

```shell
cd funasr/runtime/python/websocket
python funasr_wss_server.py --port 10095
```

##### Client Testing

```shell
python funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode 2pass --chunk_size "5,10,5"
```

For more examples, please refer to [docs](https://alibaba-damo-academy.github.io/FunASR/en/runtime/websocket_python.html#id2).

#### C++ version Example

Currently, offline file transcription service (CPU) is supported, and concurrent requests of hundreds of channels are supported.

##### Server Deployment

You can use the following command to complete the deployment with one click:

```shell
curl -O https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/shell/funasr-runtime-deploy-offline-cpu-zh.sh
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh install --workspace ./funasr-runtime-resources
```

##### Client Testing

```shell
python3 funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "../audio/asr_example.wav"
```

For more examples, please refer to [docs](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/docs/SDK_tutorial_zh.md)


### Industrial Model Egs

If you want to use the pre-trained industrial models in ModelScope for inference or fine-tuning training, you can refer to the following command:

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

More examples could be found in [docs](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_pipeline/quick_start.html)

### Academic model egs

If you want to train from scratch, usually for academic models, you can start training and inference with the following command:

```shell
cd egs/aishell/paraformer
. ./run.sh --CUDA_VISIBLE_DEVICES="0,1" --gpu_num=2
```
More examples could be found in [docs](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_pipeline/quick_start.html)

<a name="contact"></a>
## Contact

If you have any questions about FunASR, please contact us by

- email: [funasr@list.alibaba-inc.com](funasr@list.alibaba-inc.com)

|Dingding group |                     Wechat group                      |
|:---:|:-----------------------------------------------------:|
|<div align="left"><img src="docs/images/dingding.jpg" width="250"/> | <img src="docs/images/wechat.png" width="232"/></div> |

## Contributors

| <div align="left"><img src="docs/images/damo.png" width="180"/> | <div align="left"><img src="docs/images/nwpu.png" width="260"/> | <img src="docs/images/China_Telecom.png" width="200"/> </div>  | <img src="docs/images/RapidAI.png" width="200"/> </div> | <img src="docs/images/aihealthx.png" width="200"/> </div> |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|:--------------------------------------------------------------:|:-------------------------------------------------------:|:-----------------------------------------------------------:|

## Acknowledge

1. We borrowed a lot of code from [Kaldi](http://kaldi-asr.org/) for data preparation.
2. We borrowed a lot of code from [ESPnet](https://github.com/espnet/espnet). FunASR follows up the training and finetuning pipelines of ESPnet.
3. We referred [Wenet](https://github.com/wenet-e2e/wenet) for building dataloader for large scale data training.
4. We acknowledge [ChinaTelecom](https://github.com/zhuzizyf/damo-fsmn-vad-infer-httpserver) for contributing the VAD runtime. 
5. We acknowledge [RapidAI](https://github.com/RapidAI) for contributing the Paraformer and CT_Transformer-punc runtime.
6. We acknowledge [AiHealthx](http://www.aihealthx.com/) for contributing the websocket service and html5.

## License
This project is licensed under the [The MIT License](https://opensource.org/licenses/MIT). FunASR also contains various third-party components and some code modified from other repos under other open source licenses.
The use of pretraining model is subject to [model licencs](./MODEL_LICENSE)


## Stargazers over time

[![Stargazers over time](https://starchart.cc/alibaba-damo-academy/FunASR.svg)](https://starchart.cc/alibaba-damo-academy/FunASR)

## Citations

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
