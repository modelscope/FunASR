[//]: # (<div align="left"><img src="docs/images/funasr_logo.jpg" width="400"/></div>)

# FunASR: A Fundamental End-to-End Speech Recognition Toolkit

<strong>FunASR</strong> hopes to build a bridge between academic research and industrial applications on speech recognition. By supporting the training & finetuning of the industrial-grade speech recognition model released on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition), researchers and developers can conduct research and production of speech recognition models more conveniently, and promote the development of speech recognition ecology. ASR for Fun！

[**News**](https://github.com/alibaba-damo-academy/FunASR#whats-new) 
| [**Highlights**](#highlights)
| [**Installation**](#installation)
| [**Docs_CN**](https://alibaba-damo-academy.github.io/FunASR/cn/index.html)
| [**Docs_EN**](https://alibaba-damo-academy.github.io/FunASR/en/index.html)
| [**Tutorial**](https://github.com/alibaba-damo-academy/FunASR/wiki#funasr%E7%94%A8%E6%88%B7%E6%89%8B%E5%86%8C)
| [**Papers**](https://github.com/alibaba-damo-academy/FunASR#citations)
| [**Runtime**](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime)
| [**Model Zoo**](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
| [**Contact**](#contact)

[**M2MET2.0 Guidence_CN**](https://alibaba-damo-academy.github.io/FunASR/m2met2_cn/index.html)
| [**M2MET2.0 Guidence_EN**](https://alibaba-damo-academy.github.io/FunASR/m2met2/index.html)

## Multi-Channel Multi-Party Meeting Transcription 2.0 (M2MET2.0) Challenge
We are pleased to announce that the M2MeT2.0 challenge will be held in the near future. The baseline system is conducted on FunASR and is provided as a receipe of AliMeeting corpus. For more details you can see the guidence of M2MET2.0 ([CN](https://alibaba-damo-academy.github.io/FunASR/m2met2_cn/index.html)/[EN](https://alibaba-damo-academy.github.io/FunASR/m2met2/index.html)).
## What's new: 

For the release notes, please ref to [news](https://github.com/alibaba-damo-academy/FunASR/releases)

## Highlights
- Many types of typical models are supported, e.g., [Tranformer](https://arxiv.org/abs/1706.03762), [Conformer](https://arxiv.org/abs/2005.08100), [Paraformer](https://arxiv.org/abs/2206.08317).
- We have released large number of academic and industrial pretrained models on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition)
- The pretrained model [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) obtains the best performance on many tasks in [SpeechIO leaderboard](https://github.com/SpeechColab/Leaderboard)
- FunASR supplies a easy-to-use pipeline to finetune pretrained models from [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition)
- Compared to [Espnet](https://github.com/espnet/espnet) framework, the training speed of large-scale datasets in FunASR is much faster owning to the optimized dataloader.

## Installation

``` sh
pip install "modelscope[audio_asr]" --upgrade -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip install -e ./
```
For more details, please ref to [installation](https://github.com/alibaba-damo-academy/FunASR/wiki)

## Usage
For users who are new to FunASR and ModelScope, please refer to FunASR Docs([CN](https://alibaba-damo-academy.github.io/FunASR/cn/index.html) / [EN](https://alibaba-damo-academy.github.io/FunASR/en/index.html))

## Contact

If you have any questions about FunASR, please contact us by

- email: [funasr@list.alibaba-inc.com](funasr@list.alibaba-inc.com)

|Dingding group |                     Wechat group                      |
|:---:|:-----------------------------------------------------:|
|<div align="left"><img src="docs/images/dingding.jpg" width="250"/> | <img src="docs/images/wechat.png" width="232"/></div> |

## Contributors

| <div align="left"><img src="docs/images/damo.png" width="180"/> | <div align="left"><img src="docs/images/nwpu.png" width="260"/> | <img src="docs/images/DeepScience.png" width="200"/> </div> |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|:-----------------------------------------------------------:|

## Acknowledge

1. We borrowed a lot of code from [Kaldi](http://kaldi-asr.org/) for data preparation.
2. We borrowed a lot of code from [ESPnet](https://github.com/espnet/espnet). FunASR follows up the training and finetuning pipelines of ESPnet.
3. We referred [Wenet](https://github.com/wenet-e2e/wenet) for building dataloader for large scale data training.
4. We acknowledge [DeepScience](https://www.deepscience.cn) for contributing the grpc service.

## License
This project is licensed under the [The MIT License](https://opensource.org/licenses/MIT). FunASR also contains various third-party components and some code modified from other repos under other open source licenses.

## Citations

``` bibtex
@inproceedings{gao2020universal,
  title={Universal ASR: Unifying Streaming and Non-Streaming ASR Using a Single Encoder-Decoder Model},
  author={Gao, Zhifu and Zhang, Shiliang and Lei, Ming and McLoughlin, Ian},
  booktitle={arXiv preprint arXiv:2010.14099},
  year={2020}
}

@inproceedings{gao2022paraformer,
  title={Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition},
  author={Gao, Zhifu and Zhang, Shiliang and McLoughlin, Ian and Yan, Zhijie},
  booktitle={INTERSPEECH},
  year={2022}
}
@inproceedings{Shi2023AchievingTP,
  title={Achieving Timestamp Prediction While Recognizing with Non-Autoregressive End-to-End ASR Model},
  author={Xian Shi and Yanni Chen and Shiliang Zhang and Zhijie Yan},
  booktitle={arXiv preprint arXiv:2301.12343}
  year={2023}
}
```
