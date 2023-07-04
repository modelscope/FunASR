[//]: # (<div align="left"><img src="docs/images/funasr_logo.jpg" width="400"/></div>)

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
| [**Usage**](#usage)
| [**Papers**](https://github.com/alibaba-damo-academy/FunASR#citations)
| [**Runtime**](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime)
| [**Model Zoo**](https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/model_zoo/modelscope_models.md)
| [**Contact**](#contact)
| [**M2MET2.0 Challenge**](https://github.com/alibaba-damo-academy/FunASR#multi-channel-multi-party-meeting-transcription-20-m2met20-challenge)

## What's new: 

### FunASR runtime-SDK

- 2023.07.03: 
We have release the FunASR runtime-SDK-0.1.0, file transcription service (Mandarin) is now supported ([ZH](funasr/runtime/readme_cn.md)/[EN](funasr/runtime/readme.md))

### Multi-Channel Multi-Party Meeting Transcription 2.0 (M2MeT2.0) Challenge

We are pleased to announce that the M2MeT2.0 challenge has been accepted by the ASRU 2023 challenge special session. The registration is now open. The baseline system is conducted on FunASR and is provided as a receipe of AliMeeting corpus. For more details you can see the guidence of M2MET2.0 ([CN](https://alibaba-damo-academy.github.io/FunASR/m2met2_cn/index.html)/[EN](https://alibaba-damo-academy.github.io/FunASR/m2met2/index.html)).

### Release notes

For the release notes, please ref to [news](https://github.com/alibaba-damo-academy/FunASR/releases)

## Highlights
- FunASR is a fundamental speech recognition toolkit that offers a variety of features, including speech recognition (ASR), Voice Activity Detection (VAD), Punctuation Restoration, Language Models, Speaker Verification, Speaker diarization and multi-talker ASR.
- We have released a vast collection of academic and industrial pretrained models on the [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition), which can be accessed through our [Model Zoo](https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/model_zoo/modelscope_models.md). The representative [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) model has achieved SOTA performance in many speech recognition tasks. 
- FunASR offers a user-friendly pipeline for fine-tuning pretrained models from the [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition). Additionally, the optimized dataloader in FunASR enables faster training speeds for large-scale datasets. This feature enhances the efficiency of the speech recognition process for researchers and practitioners.

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

## Usage

You could use FunASR by:

- egs
- egs_modelscope
- runtime

### egs
If you want to train the model from scratch, you could use funasr directly by recipe, as the following:
```shell
cd egs/aishell/paraformer
. ./run.sh --CUDA_VISIBLE_DEVICES="0,1" --gpu_num=2
```
More examples could be found in [docs](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_pipeline/quick_start.html)

### egs_modelscope
If you want to infer or finetune pretraining models from modelscope, you could use funasr by modelscope pipeline, as the following:

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

### runtime

An example with websocket:

For the server:
```shell
cd funasr/runtime/python/websocket
python wss_srv_asr.py --port 10095
```

For the client:
```shell
python wss_client_asr.py --host "127.0.0.1" --port 10095 --mode 2pass --chunk_size "5,10,5"
#python wss_client_asr.py --host "127.0.0.1" --port 10095 --mode 2pass --chunk_size "8,8,4" --audio_in "./data/wav.scp" --output_dir "./results"
```
More examples could be found in [docs](https://alibaba-damo-academy.github.io/FunASR/en/runtime/websocket_python.html#id2)
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
