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

## What's new: 

### 2023.2.17, funasr-0.2.0, modelscope-1.3.0
- We support a new feature, export paraformer models into [onnx and torchscripts](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export) from modelscope. The local finetuned models are also supported.
- We support a new feature, [onnxruntime](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/onnxruntime/paraformer/rapid_paraformer), you could deploy the runtime without modelscope or funasr, for the [paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) model, the rtf of onnxruntime is 3x speedup(0.110->0.038) on cpu, [details](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/onnxruntime/paraformer/rapid_paraformer#speed).
- We support a new feature, [grpc](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/grpc), you could build the ASR service with grpc, by deploying the modelscope pipeline or onnxruntime.
- We release a new model [paraformer-large-contextual](https://www.modelscope.cn/models/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/summary), which supports the hotword customization based on the incentive enhancement, and improves the recall and precision of hotwords.
- We optimize the timestamp alignment of [Paraformer-large-long](https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary), the prediction accuracy of timestamp is much improved, and achieving accumulated average shift (aas) of 74.7ms, [details](https://arxiv.org/abs/2301.12343).
- We release a new model, [8k VAD model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary), which could predict the duration of none-silence speech. It could be freely integrated with any ASR models in [modelscope](https://github.com/alibaba-damo-academy/FunASR/discussions/134).
- We release a new model, [MFCCA](https://www.modelscope.cn/models/NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950/summary), a multi-channel multi-speaker model which is independent of the number and geometry of microphones and supports Mandarin meeting transcription.
- We release several new UniASR model: 
[Southern Fujian Dialect model](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-minnan-16k-common-vocab3825/summary),
[French model](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-fr-16k-common-vocab3472-tensorflow1-online/summary), 
[German model](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-de-16k-common-vocab3690-tensorflow1-online/summary), 
[Vietnamese model](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-vi-16k-common-vocab1001-pytorch-online/summary), 
[Persian model](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-fa-16k-common-vocab1257-pytorch-online/summary).
- We release a new model, [paraformer-data2vec model](https://www.modelscope.cn/models/damo/speech_data2vec_pretrain-paraformer-zh-cn-aishell2-16k/summary), an unsupervised pretraining model on AISHELL-2, which is inited for paraformer model and then finetune on AISHEL-1.
- We release a new feature, the `VAD`, `ASR` and `PUNC` models could be integrated freely, which could be models from [modelscope](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary), or the local finetine models. The [demo](https://github.com/alibaba-damo-academy/FunASR/discussions/134).
- We optimized the [punctuation common model](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary), enhance the recall and precision, fix the badcases of missing punctuation marks.
- Various new types of audio input types are now supported by modelscope inference pipeline, including: mp3、flac、ogg、opus...
### 2023.1.16, funasr-0.1.6， modelscope-1.2.0
- We release a new version model [Paraformer-large-long](https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary), which integrate the [VAD](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) model, [ASR](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary),
 [Punctuation](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary) model and timestamp together. The model could take in several hours long inputs.
- We release a new model, [16k VAD model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary), which could predict the duration of none-silence speech. It could be freely integrated with any ASR models in [modelscope](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary).
- We release a new model, [Punctuation](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary), which could predict the punctuation of ASR models's results. It could be freely integrated with any ASR models in [Model Zoo](docs/modelscope_models.md).
- We release a new model, [Data2vec](https://www.modelscope.cn/models/damo/speech_data2vec_pretrain-zh-cn-aishell2-16k-pytorch/summary), an unsupervised pretraining model which could be finetuned on ASR and other downstream tasks.
- We release a new model, [Paraformer-Tiny](https://www.modelscope.cn/models/damo/speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch/summary), a lightweight Paraformer model which supports Mandarin command words recognition.
- We release a new model, [SV](https://www.modelscope.cn/models/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/summary), which could extract speaker embeddings and further perform speaker verification on paired utterances. It will be supported for speaker diarization in the future version.
- We improve the pipeline of modelscope to speedup the inference, by integrating the process of build model into build pipeline.
- Various new types of audio input types are now supported by modelscope inference pipeline, including wav.scp, wav format, audio bytes, wave samples...

## Highlights
- Many types of typical models are supported, e.g., [Tranformer](https://arxiv.org/abs/1706.03762), [Conformer](https://arxiv.org/abs/2005.08100), [Paraformer](https://arxiv.org/abs/2206.08317).
- We have released large number of academic and industrial pretrained models on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition)
- The pretrained model [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) obtains the best performance on many tasks in [SpeechIO leaderboard](https://github.com/SpeechColab/Leaderboard)
- FunASR supplies a easy-to-use pipeline to finetune pretrained models from [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition)
- Compared to [Espnet](https://github.com/espnet/espnet) framework, the training speed of large-scale datasets in FunASR is much faster owning to the optimized dataloader.

## Installation

``` sh
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip install --editable ./
```
For more details, please ref to [installation](https://github.com/alibaba-damo-academy/FunASR/wiki)

## Usage
For users who are new to FunASR and ModelScope, please refer to FunASR Docs([CN](https://alibaba-damo-academy.github.io/FunASR/cn/index.html) / [EN](https://alibaba-damo-academy.github.io/FunASR/en/index.html))

## Contact

If you have any questions about FunASR, please contact us by

- email: [funasr@list.alibaba-inc.com](funasr@list.alibaba-inc.com)

|Dingding group | Wechat group|
|:---:|:---:|
|<div align="left"><img src="docs/images/dingding.jpg" width="250"/> |<img src="docs/images/wechat.png" width="222"/></div>|

## Contributors

| <div align="left"><img src="docs/images/DeepScience.png" width="250"/> |
|:---:|

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