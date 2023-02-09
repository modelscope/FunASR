<div align="left"><img src="docs/images/funasr_logo.jpg" width="400"/></div>

# FunASR: A Fundamental End-to-End Speech Recognition Toolkit

<strong>FunASR</strong> hopes to build a bridge between academic research and industrial applications on speech recognition. By supporting the training & finetuning of the industrial-grade speech recognition model released on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition), researchers and developers can conduct research and production of speech recognition models more conveniently, and promote the development of speech recognition ecology. ASR for Fun！


 [**Docs**](https://alibaba-damo-academy.github.io/FunASR/index.html)
| [**Tutorial**](https://github.com/alibaba-damo-academy/FunASR/wiki#funasr%E7%94%A8%E6%88%B7%E6%89%8B%E5%86%8C)
| [**Papers**](https://github.com/alibaba-damo-academy/FunASR#citations)
| [**Runtime**](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime)
| [**Pretrained Models**](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
| [**Contact**](#contact)

## Release Notes: 
### 2023.1.16, funasr-0.1.6
- We release a new version model [Paraformer-large-long](https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary), which integrate the [VAD](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) model, [ASR](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary),
 [Punctuation](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary) model and timestamp together. The model could take in several hours long inputs.
- We release a new type model, [VAD](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary), which could predict the duration of none-silence speech. It could be freely integrated with any ASR models in [Model Zoo](docs/modelscope_models.md).
- We release a new type model, [Punctuation](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary), which could predict the punctuation of ASR models's results. It could be freely integrated with any ASR models in [Model Zoo](docs/modelscope_models.md).
- We release a new model, [Data2vec](https://www.modelscope.cn/models/damo/speech_data2vec_pretrain-zh-cn-aishell2-16k-pytorch/summary), an unsupervised pretraining model which could be finetuned on ASR and other downstream tasks.
- We release a new model, [Paraformer-Tiny](https://www.modelscope.cn/models/damo/speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch/summary), a lightweight Paraformer model which supports Mandarin command words recognition.
- We release a new type model, [SV](https://www.modelscope.cn/models/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/summary), which could extract speaker embeddings and further perform speaker verification on paired utterances. It will be supported for speaker diarization in the future version.
- We improve the pipeline of modelscope to speedup the inference, by integrating the process of build model into build pipeline.
- Various new types of audio input types are now supported by modelscope inference pipeline, including wav.scp, wav format, audio bytes, wave samples...

## Key Features
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
For users who are new to FunASR and ModelScope, please refer to [FunASR Docs](https://alibaba-damo-academy.github.io/FunASR/index.html).

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
```
