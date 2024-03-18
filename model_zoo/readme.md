([简体中文](./readme_zh.md)|English)

# Model Zoo

## Model License
You are free to use, copy, modify, and share FunASR models under the conditions of this agreement. You should indicate the model source and author information when using, copying, modifying and sharing FunASR models. You should keep the relevant names of models in [FunASR software]. Full model license could see [license](https://github.com/alibaba-damo-academy/FunASR/blob/main/MODEL_LICENSE)

## Model Usage
Ref to [docs](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_pipeline/quick_start.html)

## Model Zoo
Here we provided several pretrained models on different datasets. The details of models and datasets can be found on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition).

### Speech Recognition
#### Paraformer


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
