# Timestamp Prediction (FA)

## Inference

### Quick start
#### [Use TP-Aligner Model Simply](https://modelscope.cn/models/damo/speech_timestamp_prediction-v1-16k-offline/summary)
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.speech_timestamp,
    model='damo/speech_timestamp_prediction-v1-16k-offline',
    model_revision='v1.1.0')

rec_result = inference_pipeline(
    audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_timestamps.wav',
    text_in='一 个 东 太 平 洋 国 家 为 什 么 跑 到 西 太 平 洋 来 了 呢',)
print(rec_result)
```

Timestamp pipeline can also be used after ASR pipeline to compose complete ASR function, ref to [demo](https://github.com/alibaba-damo-academy/FunASR/discussions/246).



### API-reference
#### Define pipeline
- `task`: `Tasks.speech_timestamp`
- `model`: model name in [model zoo](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope), or model path in local disk
- `ngpu`: `1` (Default), decoding on GPU. If ngpu=0, decoding on CPU
- `ncpu`: `1` (Default), sets the number of threads used for intraop parallelism on CPU 
- `output_dir`: `None` (Default), the output path of results if set
- `batch_size`: `1` (Default), batch size when decoding
#### Infer pipeline
- `audio_in`: the input speech to predict, which could be: 
  - wav_path, `e.g.`: asr_example.wav (wav in local or url), 
  - wav.scp, kaldi style wav list (`wav_id wav_path`), `e.g.`: 
    ```text
    asr_example1  ./audios/asr_example1.wav
    asr_example2  ./audios/asr_example2.wav
    ```
  In this case of `wav.scp` input, `output_dir` must be set to save the output results
- `text_in`: the input text to predict， splited by blank, which could be:
  - text string, `e.g.`: `今 天 天 气 怎 么 样`
  - text.scp, kaldi style text file (`wav_id transcription`), `e.g.`:
    ```text
    asr_example1 今 天 天 气 怎 么 样
    asr_example2 欢 迎 体 验 达 摩 院 语 音 识 别 模 型
    ```
- `audio_fs`: audio sampling rate, only set when audio_in is pcm audio
- `output_dir`: None (Default), the output path of results if set, containing
  - output_dir/timestamp_prediction/tp_sync, timestamp in second containing silence periods, `wav_id# token1 start_time end_time;`, `e.g.`:
    ```text
    test_wav1# <sil> 0.000 0.500;温 0.500 0.680;州 0.680 0.840;化 0.840 1.040;工 1.040 1.280;仓 1.280 1.520;<sil> 1.520 1.680;库 1.680 1.920;<sil> 1.920 2.160;起 2.160 2.380;火 2.380 2.580;殃 2.580 2.760;及 2.760 2.920;附 2.920 3.100;近 3.100 3.340;<sil> 3.340 3.400;河 3.400 3.640;<sil> 3.640 3.700;流 3.700 3.940;<sil> 3.940 4.240;大 4.240 4.400;量 4.400 4.520;死 4.520 4.680;鱼 4.680 4.920;<sil> 4.920 4.940;漂 4.940 5.120;浮 5.120 5.300;河 5.300 5.500;面 5.500 5.900;<sil> 5.900 6.240;
    ```
  - output_dir/timestamp_prediction/tp_time, timestamp list in ms of same length as input text without silence `wav_id# [[start_time, end_time],]`, `e.g.`:
    ```text
    test_wav1# [[500, 680], [680, 840], [840, 1040], [1040, 1280], [1280, 1520], [1680, 1920], [2160, 2380], [2380, 2580], [2580, 2760], [2760, 2920], [2920, 3100], [3100, 3340], [3400, 3640], [3700, 3940], [4240, 4400], [4400, 4520], [4520, 4680], [4680, 4920], [4940, 5120], [5120, 5300], [5300, 5500], [5500, 5900]]
    ```

### Inference with multi-thread CPUs or multi GPUs
FunASR also offer recipes [egs_modelscope/tp/TEMPLATE/infer.sh](https://github.com/alibaba-damo-academy/FunASR/blob/main/egs_modelscope/tp/TEMPLATE/infer.sh) to decode with multi-thread CPUs, or multi GPUs.

#### Settings of `infer.sh`
- `model`: model name in [model zoo](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope), or model path in local disk
- `data_dir`: the dataset dir **must** include `wav.scp` and `text.txt`
- `output_dir`: output dir of the recognition results
- `batch_size`: `64` (Default), batch size of inference on gpu
- `gpu_inference`: `true` (Default), whether to perform gpu decoding, set false for CPU inference
- `gpuid_list`: `0,1` (Default), which gpu_ids are used to infer
- `njob`: only used for CPU inference (`gpu_inference`=`false`), `64` (Default), the number of jobs for CPU decoding
- `checkpoint_dir`: only used for infer finetuned models, the path dir of finetuned models
- `checkpoint_name`: only used for infer finetuned models, `valid.cer_ctc.ave.pb` (Default), which checkpoint is used to infer

#### Decode with multi GPUs:
```shell
    bash infer.sh \
    --model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --batch_size 1 \
    --gpu_inference true \
    --gpuid_list "0,1"
```
#### Decode with multi-thread CPUs:
```shell
    bash infer.sh \
    --model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --gpu_inference false \
    --njob 1
```

## Finetune with pipeline

### Quick start

### Finetune with your data

## Inference with your finetuned model

