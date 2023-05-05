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
| [**Docs**](https://alibaba-damo-academy.github.io/FunASR/en/index.html)
| [**Tutorial_CN**](https://github.com/alibaba-damo-academy/FunASR/wiki#funasr%E7%94%A8%E6%88%B7%E6%89%8B%E5%86%8C)
| [**Papers**](https://github.com/alibaba-damo-academy/FunASR#citations)
| [**Runtime**](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime)
| [**Model Zoo**](https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/model_zoo/modelscope_models.md)
| [**Contact**](#contact)
| [**M2MET2.0 Challenge**](https://github.com/alibaba-damo-academy/FunASR#multi-channel-multi-party-meeting-transcription-20-m2met20-challenge)

## What's new: 
### Multi-Channel Multi-Party Meeting Transcription 2.0 (M2MET2.0) Challenge
We are pleased to announce that the M2MeT2.0 challenge will be held in the near future. The baseline system is conducted on FunASR and is provided as a receipe of AliMeeting corpus. For more details you can see the guidence of M2MET2.0 ([CN](https://alibaba-damo-academy.github.io/FunASR/m2met2_cn/index.html)/[EN](https://alibaba-damo-academy.github.io/FunASR/m2met2/index.html)).
### Release notes
For the release notes, please ref to [news](https://github.com/alibaba-damo-academy/FunASR/releases)

## Highlights
- FunASR supports speech recognition(ASR), Multi-talker ASR, Voice Activity Detection(VAD), Punctuation Restoration, Language Models, Speaker Verification and Speaker diarization.   
- We have released large number of academic and industrial pretrained models on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition), ref to [Model Zoo](https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/model_zoo/modelscope_models.md)
- The pretrained model [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) obtains the best performance on many tasks in [SpeechIO leaderboard](https://github.com/SpeechColab/Leaderboard)
- FunASR supplies a easy-to-use pipeline to finetune pretrained models from [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition)
- Compared to [Espnet](https://github.com/espnet/espnet) framework, the training speed of large-scale datasets in FunASR is much faster owning to the optimized dataloader.

## Installation

Install from pip
```shell
pip install -U funasr
# For the users in China, you could install with the command:
# pip install -U funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

Or install from source code


``` sh
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip install -e ./
# For the users in China, you could install with the command:
# pip install -e ./ -i https://mirror.sjtu.edu.cn/pypi/web/simple

```
If you want to use the pretrained models in ModelScope, you should install the modelscope:

```shell
pip install -U modelscope
# For the users in China, you could install with the command:
# pip install -U modelscope -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

For more details, please ref to [installation](https://alibaba-damo-academy.github.io/FunASR/en/installation.html)

[//]: # ()
[//]: # (## Usage)

[//]: # (For users who are new to FunASR and ModelScope, please refer to FunASR Docs&#40;[CN]&#40;https://alibaba-damo-academy.github.io/FunASR/cn/index.html&#41; / [EN]&#40;https://alibaba-damo-academy.github.io/FunASR/en/index.html&#41;&#41;)

## Contact

If you have any questions about FunASR, please contact us by

- email: [funasr@list.alibaba-inc.com](funasr@list.alibaba-inc.com)

|Dingding group |                     Wechat group                      |
|:---:|:-----------------------------------------------------:|
|<div align="left"><img src="docs/images/dingding.jpg" width="250"/> | <img src="docs/images/wechat.png" width="232"/></div> |

## Contributors

| <div align="left"><img src="docs/images/damo.png" width="180"/> | <div align="left"><img src="docs/images/nwpu.png" width="260"/> | <img src="docs/images/China_Telecom.png" width="200"/> </div>  | <img src="docs/images/RapidAI.png" width="200"/> </div> | <img src="docs/images/DeepScience.png" width="200"/> </div> |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|:--------------------------------------------------------------:|:-------------------------------------------------------:|:-----------------------------------------------------------:|

## Acknowledge

1. We borrowed a lot of code from [Kaldi](http://kaldi-asr.org/) for data preparation.
2. We borrowed a lot of code from [ESPnet](https://github.com/espnet/espnet). FunASR follows up the training and finetuning pipelines of ESPnet.
3. We referred [Wenet](https://github.com/wenet-e2e/wenet) for building dataloader for large scale data training.
4. We acknowledge [ChinaTelecom](https://github.com/zhuzizyf/damo-fsmn-vad-infer-httpserver) for contributing the VAD runtime. 
5. We acknowledge [RapidAI](https://github.com/RapidAI) for contributing the Paraformer and CT_Transformer-punc runtime.
6. We acknowledge [DeepScience](https://www.deepscience.cn) for contributing the grpc service.

## License
This project is licensed under the [The MIT License](https://opensource.org/licenses/MIT). FunASR also contains various third-party components and some code modified from other repos under other open source licenses.

## Citations

``` bibtex
@inproceedings{gao2022paraformer,
  title={Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition},
  author={Gao, Zhifu and Zhang, Shiliang and McLoughlin, Ian and Yan, Zhijie},
  booktitle={INTERSPEECH},
  year={2022}
}
@inproceedings{gao2020universal,
  title={Universal ASR: Unifying Streaming and Non-Streaming ASR Using a Single Encoder-Decoder Model},
  author={Gao, Zhifu and Zhang, Shiliang and Lei, Ming and McLoughlin, Ian},
  booktitle={arXiv preprint arXiv:2010.14099},
  year={2020}
}
@inproceedings{Shi2023AchievingTP,
  title={Achieving Timestamp Prediction While Recognizing with Non-Autoregressive End-to-End ASR Model},
  author={Xian Shi and Yanni Chen and Shiliang Zhang and Zhijie Yan},
  booktitle={arXiv preprint arXiv:2301.12343}
  year={2023}
}
```
