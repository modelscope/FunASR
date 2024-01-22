[//]: # (<div align="left"><img src="docs/images/funasr_logo.jpg" width="400"/></div>)

([简体中文](./README_zh.md)|English)

# FunASR: A Fundamental End-to-End Speech Recognition Toolkit
<p align="left">
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-brightgreen.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Pytorch-%3E%3D1.11-blue"></a>
</p>

<strong>FunASR</strong> hopes to build a bridge between academic research and industrial applications on speech recognition. By supporting the training & finetuning of the industrial-grade speech recognition model, researchers and developers can conduct research and production of speech recognition models more conveniently, and promote the development of speech recognition ecology. ASR for Fun！

[**Highlights**](#highlights)
| [**News**](https://github.com/alibaba-damo-academy/FunASR#whats-new) 
| [**Installation**](#installation)
| [**Quick Start**](#quick-start)
| [**Runtime**](./runtime/readme.md)
| [**Model Zoo**](#model-zoo)
| [**Contact**](#contact)


<a name="highlights"></a>
## Highlights
- FunASR is a fundamental speech recognition toolkit that offers a variety of features, including speech recognition (ASR), Voice Activity Detection (VAD), Punctuation Restoration, Language Models, Speaker Verification, Speaker Diarization and multi-talker ASR. FunASR provides convenient scripts and tutorials, supporting inference and fine-tuning of pre-trained models.
- We have released a vast collection of academic and industrial pretrained models on the [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition) and [huggingface](https://huggingface.co/FunASR), which can be accessed through our [Model Zoo](https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/model_zoo/modelscope_models.md). The representative [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary), a non-autoregressive end-to-end speech recognition model, has the advantages of high accuracy, high efficiency, and convenient deployment, supporting the rapid construction of speech recognition services. For more details on service deployment, please refer to the [service deployment document](runtime/readme_cn.md). 


<a name="whats-new"></a>
## What's new:
- 2024/01/09: The Funasr SDK for Windows version 2.0 has been released, featuring support for The offline file transcription service (CPU) of Mandarin 4.1, The offline file transcription service (CPU) of English 1.2, The real-time transcription service (CPU) of Mandarin 1.6. For more details, please refer to the official documentation or release notes([FunASR-Runtime-Windows](https://www.modelscope.cn/models/damo/funasr-runtime-win-cpu-x64/summary))
- 2024/01/03: File Transcription Service 4.0 released, Added support for 8k models, optimized timestamp mismatch issues and added sentence-level timestamps, improved the effectiveness of English word FST hotwords, supported automated configuration of thread parameters, and fixed known crash issues as well as memory leak problems, refer to ([docs](runtime/readme.md#file-transcription-service-mandarin-cpu)).
- 2024/01/03: Real-time Transcription Service 1.6 released，The 2pass-offline mode supports Ngram language model decoding and WFST hotwords, while also addressing known crash issues and memory leak problems, ([docs](runtime/readme.md#the-real-time-transcription-service-mandarin-cpu))
- 2024/01/03: Fixed known crash issues as well as memory leak problems, ([docs](runtime/readme.md#file-transcription-service-english-cpu)).
- 2023/12/04: The Funasr SDK for Windows version 1.0 has been released, featuring support for The offline file transcription service (CPU) of Mandarin, The offline file transcription service (CPU) of English, The real-time transcription service (CPU) of Mandarin. For more details, please refer to the official documentation or release notes([FunASR-Runtime-Windows](https://www.modelscope.cn/models/damo/funasr-runtime-win-cpu-x64/summary))
- 2023/11/08: The offline file transcription service 3.0 (CPU) of Mandarin has been released, adding punctuation large model, Ngram language model, and wfst hot words. For detailed information, please refer to [docs](runtime#file-transcription-service-mandarin-cpu). 
- 2023/10/17: The offline file transcription service (CPU) of English has been released. For more details, please refer to ([docs](runtime#file-transcription-service-english-cpu)).
- 2023/10/13: [SlideSpeech](https://slidespeech.github.io/): A large scale multi-modal audio-visual corpus with a significant amount of real-time synchronized slides.
- 2023/10/10: The ASR-SpeakersDiarization combined pipeline [Paraformer-VAD-SPK](https://github.com/alibaba-damo-academy/FunASR/blob/main/egs_modelscope/asr_vad_spk/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn/demo.py) is now released. Experience the model to get recognition results with speaker information.
- 2023/10/07: [FunCodec](https://github.com/alibaba-damo-academy/FunCodec): A Fundamental, Reproducible and Integrable Open-source Toolkit for Neural Speech Codec.
- 2023/09/01: The offline file transcription service 2.0 (CPU) of Mandarin has been released, with added support for ffmpeg, timestamp, and hotword models. For more details, please refer to ([docs](runtime#file-transcription-service-mandarin-cpu)).
- 2023/08/07: The real-time transcription service (CPU) of Mandarin has been released. For more details, please refer to ([docs](runtime#the-real-time-transcription-service-mandarin-cpu)).
- 2023/07/17: BAT is released, which is a low-latency and low-memory-consumption RNN-T model. For more details, please refer to ([BAT](egs/aishell/bat)).
- 2023/06/26: ASRU2023 Multi-Channel Multi-Party Meeting Transcription Challenge 2.0 completed the competition and announced the results. For more details, please refer to ([M2MeT2.0](https://alibaba-damo-academy.github.io/FunASR/m2met2/index.html)).


<a name="Installation"></a>
## Installation

Please ref to [installation docs](https://alibaba-damo-academy.github.io/FunASR/en/installation/installation.html)

## Model Zoo
FunASR has open-sourced a large number of pre-trained models on industrial data. You are free to use, copy, modify, and share FunASR models under the [Model License Agreement](./MODEL_LICENSE). Below are some representative models, for more models please refer to the [Model Zoo]().

(Note: 🤗 represents the Huggingface model zoo link, ⭐ represents the ModelScope model zoo link)


|                                                                             Model Name                                                                             |                                Task Details                                 |          Training Data           | Parameters |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------:|:--------------------------------:|:----------:|
|    paraformer-zh <br> ([⭐](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)  [🤗]() )    |             speech recognition, with timestamps, non-streaming              |      60000 hours, Mandarin       |    220M    |
|                paraformer-zh-spk <br> ( [⭐](https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn/summary)  [🤗]() )                | speech recognition with speaker diarization, with timestamps, non-streaming |      60000 hours, Mandarin       |    220M    |
| <nobr>paraformer-zh-online <br> ( [⭐](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary) [🤗]() )</nobr> |                        speech recognition, streaming                        |      60000 hours, Mandarin       |    220M    |
|         paraformer-en <br> ( [⭐](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020/summary) [🤗]() )         |             speech recognition, with timestamps, non-streaming              |       50000 hours, English       |    220M    |
|                     conformer-en <br> ( [⭐](https://modelscope.cn/models/damo/speech_conformer_asr-en-16k-vocab4199-pytorch/summary) [🤗]() )                      |                      speech recognition, non-streaming                      |       50000 hours, English       |    220M    |
|                     ct-punc <br> ( [⭐](https://modelscope.cn/models/damo/punc_ct-transformer_cn-en-common-vocab471067-large/summary) [🤗]() )                      |                           punctuation restoration                           |    100M, Mandarin and English    |    1.1G    | 
|                          fsmn-vad <br> ( [⭐](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) [🤗]() )                          |                          voice activity detection                           | 5000 hours, Mandarin and English |    0.4M    | 
|                          fa-zh <br> ( [⭐](https://modelscope.cn/models/damo/speech_timestamp_prediction-v1-16k-offline/summary) [🤗]() )                           |                            timestamp prediction                             |       5000 hours, Mandarin       |    38M     | 




[//]: # ()
[//]: # (FunASR supports pre-trained or further fine-tuned models for deployment as a service. The CPU version of the Chinese offline file conversion service has been released, details can be found in [docs]&#40;funasr/runtime/docs/SDK_tutorial.md&#41;. More detailed information about service deployment can be found in the [deployment roadmap]&#40;funasr/runtime/readme_cn.md&#41;.)


<a name="quick-start"></a>
## Quick Start

Below is a quick start tutorial. Test audio files ([Mandarin](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav), [English]()).

### Command-line usage

```shell
funasr +model=paraformer-zh +vad_model="fsmn-vad" +punc_model="ct-punc" +input=asr_example_zh.wav
```

Notes: Support recognition of single audio file, as well as file list in Kaldi-style wav.scp format: `wav_id wav_pat`

### Speech Recognition (Non-streaming)
```python
from funasr import AutoModel
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
model = AutoModel(model="paraformer-zh", model_revision="v2.0.2",
                  vad_model="fsmn-vad", vad_model_revision="v2.0.2",
                  punc_model="ct-punc-c", punc_model_revision="v2.0.3",
                  # spk_model="cam++", spk_model_revision="v2.0.2",
                  )
res = model.generate(input=f"{model.model_path}/example/asr_example.wav", 
                     batch_size_s=300, 
                     hotword='魔搭')
print(res)
```
Note: `model_hub`: represents the model repository, `ms` stands for selecting ModelScope download, `hf` stands for selecting Huggingface download.

### Speech Recognition (Streaming)
```python
from funasr import AutoModel

chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.2")

import soundfile
import os

wav_file = os.path.join(model.model_path, "example/asr_example.wav")
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = chunk_size[1] * 960 # 600ms

cache = {}
total_chunk_num = int(len((speech)-1)/chunk_stride+1)
for i in range(total_chunk_num):
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    is_final = i == total_chunk_num - 1
    res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
    print(res)
```
Note: `chunk_size` is the configuration for streaming latency.` [0,10,5]` indicates that the real-time display granularity is `10*60=600ms`, and the lookahead information is `5*60=300ms`. Each inference input is `600ms` (sample points are `16000*0.6=960`), and the output is the corresponding text. For the last speech segment input, `is_final=True` needs to be set to force the output of the last word.

### Voice Activity Detection (streaming)
```python
from funasr import AutoModel

model = AutoModel(model="fsmn-vad", model_revision="v2.0.2")
wav_file = f"{model.model_path}/example/asr_example.wav"
res = model.generate(input=wav_file)
print(res)
```
### Voice Activity Detection (Non-streaming)
```python
from funasr import AutoModel

chunk_size = 200 # ms
model = AutoModel(model="fsmn-vad", model_revision="v2.0.2")

import soundfile

wav_file = f"{model.model_path}/example/vad_example.wav"
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = int(chunk_size * sample_rate / 1000)

cache = {}
total_chunk_num = int(len((speech)-1)/chunk_stride+1)
for i in range(total_chunk_num):
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    is_final = i == total_chunk_num - 1
    res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size)
    if len(res[0]["value"]):
        print(res)
```
### Punctuation Restoration
```python
from funasr import AutoModel

model = AutoModel(model="ct-punc", model_revision="v2.0.2")
res = model.generate(input="那今天的会就到这里吧 happy new year 明年见")
print(res)
```
### Timestamp Prediction
```python
from funasr import AutoModel

model = AutoModel(model="fa-zh", model_revision="v2.0.2")
wav_file = f"{model.model_path}/example/asr_example.wav"
text_file = f"{model.model_path}/example/text.txt"
res = model.generate(input=(wav_file, text_file), data_type=("sound", "text"))
print(res)
```

More examples ref to [docs](https://github.com/alibaba-damo-academy/FunASR/tree/main/examples/industrial_data_pretraining)

[//]: # (FunASR supports inference and fine-tuning of models trained on industrial datasets of tens of thousands of hours. For more details, please refer to &#40;[modelscope_egs]&#40;https://alibaba-damo-academy.github.io/FunASR/en/modelscope_pipeline/quick_start.html&#41;&#41;. It also supports training and fine-tuning of models on academic standard datasets. For more details, please refer to&#40;[egs]&#40;https://alibaba-damo-academy.github.io/FunASR/en/academic_recipe/asr_recipe.html&#41;&#41;. The models include speech recognition &#40;ASR&#41;, speech activity detection &#40;VAD&#41;, punctuation recovery, language model, speaker verification, speaker separation, and multi-party conversation speech recognition. For a detailed list of models, please refer to the [Model Zoo]&#40;https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/model_zoo/modelscope_models.md&#41;:)

## Deployment Service
FunASR supports deploying pre-trained or further fine-tuned models for service. Currently, it supports the following types of service deployment:
- File transcription service, Mandarin, CPU version, done
- The real-time transcription service, Mandarin (CPU), done
- File transcription service, English, CPU version, done
- File transcription service, Mandarin, GPU version, in progress
- and more.

For more detailed information, please refer to the [service deployment documentation](runtime/readme.md).


<a name="contact"></a>
## Community Communication
If you encounter problems in use, you can directly raise Issues on the github page.

You can also scan the following DingTalk group or WeChat group QR code to join the community group for communication and discussion.

|DingTalk group |                     WeChat group                      |
|:---:|:-----------------------------------------------------:|
|<div align="left"><img src="docs/images/dingding.jpg" width="250"/> | <img src="docs/images/wechat.png" width="215"/></div> |

## Contributors

| <div align="left"><img src="docs/images/alibaba.png" width="260"/> | <div align="left"><img src="docs/images/nwpu.png" width="260"/> | <img src="docs/images/China_Telecom.png" width="200"/> </div>  | <img src="docs/images/RapidAI.png" width="200"/> </div> | <img src="docs/images/aihealthx.png" width="200"/> </div> | <img src="docs/images/XVERSE.png" width="250"/> </div> |
|:------------------------------------------------------------------:|:---------------------------------------------------------------:|:--------------------------------------------------------------:|:-------------------------------------------------------:|:-----------------------------------------------------------:|:------------------------------------------------------:|

The contributors can be found in [contributors list](./Acknowledge.md)

## License
This project is licensed under [The MIT License](https://opensource.org/licenses/MIT). FunASR also contains various third-party components and some code modified from other repos under other open source licenses.
The use of pretraining model is subject to [model license](./MODEL_LICENSE)


## Citations
``` bibtex
@inproceedings{gao2023funasr,
  author={Zhifu Gao and Zerui Li and Jiaming Wang and Haoneng Luo and Xian Shi and Mengzhe Chen and Yabin Li and Lingyun Zuo and Zhihao Du and Zhangyu Xiao and Shiliang Zhang},
  title={FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
  year={2023},
  booktitle={INTERSPEECH},
}
@inproceedings{An2023bat,
  author={Keyu An and Xian Shi and Shiliang Zhang},
  title={BAT: Boundary aware transducer for memory-efficient and low-latency ASR},
  year={2023},
  booktitle={INTERSPEECH},
}
@inproceedings{gao22b_interspeech,
  author={Zhifu Gao and ShiLiang Zhang and Ian McLoughlin and Zhijie Yan},
  title={Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={2063--2067},
  doi={10.21437/Interspeech.2022-9996}
}
@inproceedings{shi2023seaco,
  author={Xian Shi and Yexin Yang and Zerui Li and Yanni Chen and Zhifu Gao and Shiliang Zhang},
  title={SeACo-Paraformer: A Non-Autoregressive ASR System with Flexible and Effective Hotword Customization Ability},
  year={2023},
  booktitle={ICASSP2024}
}
```
