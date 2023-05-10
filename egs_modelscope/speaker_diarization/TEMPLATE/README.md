# Speaker Diarization

> **Note**: 
> The modelscope pipeline supports all the models in 
[model zoo](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope) 
to inference and finetine. Here we take the model of xvector_sv as example to demonstrate the usage.

## Inference with pipeline
### Quick start
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# initialize pipeline
inference_diar_pipline = pipeline(
    mode="sond_demo",
    num_workers=0,
    task=Tasks.speaker_diarization,
    diar_model_config="sond.yaml",
    model='damo/speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch',
    reversion="v1.0.5",
    sv_model="damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch",
    sv_model_revision="v1.2.2",
)

# input: a list of audio in which the first item is a speech recording to detect speakers, 
# and the following wav file are used to extract speaker embeddings.
audio_list = [
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/record.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/spk1.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/spk2.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/spk3.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/spk4.wav",
]

results = inference_diar_pipline(audio_in=audio_list)
print(results)
```

### API-reference
#### Define pipeline
- `task`: `Tasks.speaker_diarization`
- `model`: model name in [model zoo](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope), or model path in local disk
- `ngpu`: `1` (Default), decoding on GPU. If ngpu=0, decoding on CPU
- `output_dir`: `None` (Default), the output path of results if set
- `batch_size`: `1` (Default), batch size when decoding
- `smooth_size`: `83` (Default), the window size to perform smoothing
- `dur_threshold`: `10` (Default), segments shorter than 100 ms will be dropped
- `out_format`: `vad` (Default), the output format, choices `["vad", "rttm"]`. 
  - vad format: spk1: [1.0, 3.0], [5.0, 8.0]
  - rttm format: "SPEAKER test1 0 1.00 2.00 <NA> <NA> spk1 <NA> <NA>" and "SPEAKER test1 0 5.00 3.00 <NA> <NA> spk1 <NA> <NA>"

#### Infer pipeline for speaker embedding extraction
- `audio_in`: the input to process, which could be: 
  - list of url: `e.g.`: waveform files at a website
  - list of local file path: `e.g.`: path/to/a.wav
  - ("wav.scp,speech,sound", "profile.scp,profile,kaldi_ark"): a script file of waveform files and another script file of speaker profiles (extracted with the [model](https://www.modelscope.cn/models/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/summary))
    ```text
    wav.scp
    test1 path/to/enroll1.wav
    test2 path/to/enroll2.wav
    
    profile.scp
    test1 path/to/profile.ark:11
    test2 path/to/profile.ark:234
    ```
    The profile.ark file contains speaker embeddings in a kaldi-like style. 
    Please refer [README.md](../../speaker_verification/TEMPLATE/README.md) for more details.

### Inference with you data
For single input, we recommend the "list of local file path" mode for inference.
For multiple inputs, we recommend the last mode with pre-organized wav.scp and profile.scp.

### Inference with multi-threads on CPU
We recommend the last mode with split wav.scp and profile.scp. Then, run inference for each split part.
Please refer [README.md](../../speaker_verification/TEMPLATE/README.md) to find a similar process.

### Inference with multi GPU
Similar to CPU, please set `ngpu=1` for inference on GPU.
Besides, you should use `CUDA_VISIBLE_DEVICES=0` to specify a GPU device.
Please refer [README.md](../../speaker_verification/TEMPLATE/README.md) to find a similar process.
