# Speech Recognition

> **Note**: 
> The modelscope pipeline supports all the models in [model zoo](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_models.html#pretrained-models-on-modelscope) to inference and finetine. Here we take typic model as example to demonstrate the usage.

## Inference

### Quick start
#### [Paraformer Model](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
)

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
print(rec_result)
```
#### [Paraformer-online Model](https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online/summary)
```python
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online',
    )
import soundfile
speech, sample_rate = soundfile.read("example/asr_example.wav")

param_dict = {"cache": dict(), "is_final": False}
chunk_stride = 7680# 480ms
# first chunk, 480ms
speech_chunk = speech[0:chunk_stride] 
rec_result = inference_pipeline(audio_in=speech_chunk, param_dict=param_dict)
print(rec_result)
# next chunk, 480ms
speech_chunk = speech[chunk_stride:chunk_stride+chunk_stride]
rec_result = inference_pipeline(audio_in=speech_chunk, param_dict=param_dict)
print(rec_result)
```
Full code of demo, please ref to [demo](https://github.com/alibaba-damo-academy/FunASR/discussions/241)

#### [UniASR Model](https://www.modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-online/summary)
There are three decoding mode for UniASR model(`fast`、`normal`、`offline`), for more model detailes, please refer to [docs](https://www.modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-online/summary)
```python
decoding_model = "fast" # "fast"、"normal"、"offline"
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_UniASR_asr_2pass-minnan-16k-common-vocab3825',
    param_dict={"decoding_model": decoding_model})

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
print(rec_result)
```
The decoding mode of `fast` and `normal`
Full code of demo, please ref to [demo](https://github.com/alibaba-damo-academy/FunASR/discussions/151)
#### [RNN-T-online model]()
Undo

#### API-reference
##### Define pipeline
- `task`: `Tasks.auto_speech_recognition`
- `model`: model name in [model zoo](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_models.html#pretrained-models-on-modelscope), or model path in local disk
- `ngpu`: 1 (Defalut), decoding on GPU. If ngpu=0, decoding on CPU
- `ncpu`: 1 (Defalut), sets the number of threads used for intraop parallelism on CPU 
- `output_dir`: None (Defalut), the output path of results if set
- `batch_size`: 1 (Defalut), batch size when decoding
##### Infer pipeline
- `audio_in`: the input to decode, which could be: 
  - wav_path, `e.g.`: asr_example.wav,
  - pcm_path, `e.g.`: asr_example.pcm, 
  - audio bytes stream, `e.g.`: bytes data from a microphone
  - audio sample point，`e.g.`: `audio, rate = soundfile.read("asr_example_zh.wav")`, the dtype is numpy.ndarray or torch.Tensor
  - wav.scp, kaldi style wav list (`wav_id \t wav_path``), `e.g.`: 
  ```text
  asr_example1  ./audios/asr_example1.wav
  asr_example2  ./audios/asr_example2.wav
  ```
  In this case of `wav.scp` input, `output_dir` must be set to save the output results
- `audio_fs`: audio sampling rate, only set when audio_in is pcm audio
- `output_dir`: None (Defalut), the output path of results if set

### Inference with multi-thread CPUs or multi GPUs
FunASR also offer recipes [infer.sh](https://github.com/alibaba-damo-academy/FunASR/blob/main/egs_modelscope/asr/TEMPLATE/infer.sh) to decode with multi-thread CPUs, or multi GPUs.

- Setting parameters in `infer.sh`
    - `model`: model name on ModelScope
    - `data_dir`: the dataset dir needs to include `${data_dir}/wav.scp`. If `${data_dir}/text` is also exists, CER will be computed
    - `output_dir`: result dir
    - `batch_size`: batchsize of inference
    - `gpu_inference`: whether to perform gpu decoding, set false for cpu decoding
    - `gpuid_list`: set gpus, e.g., `gpuid_list`="0,1"
    - `njob`: the number of jobs for CPU decoding, if `gpu_inference`=false, use CPU decoding, please set `njob`
    - `checkpoint_dir`: only used for infer finetuned models, the path dir of finetuned models
    - `checkpoint_name`: only used for infer finetuned models, `valid.cer_ctc.ave.pb` (Default), which checkpoint is used to infer

- Decode with multi GPUs:
```shell
    bash infer.sh \
    --model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --batch_size 64 \
    --gpu_inference true \
    --gpuid_list "0,1"
```
- Decode with multi-thread CPUs:
```shell
    bash infer.sh \
    --model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --gpu_inference false \
    --njob 64
```

- Results

The decoding results can be found in `$output_dir/1best_recog/text.cer`, which includes recognition results of each sample and the CER metric of the whole test set.

If you decode the SpeechIO test sets, you can use textnorm with `stage`=3, and `DETAILS.txt`, `RESULTS.txt` record the results and CER after text normalization.


## Finetune with pipeline

### Quick start
[finetune.py](https://github.com/alibaba-damo-academy/FunASR/blob/main/egs_modelscope/asr/TEMPLATE/finetune.py)
```python
import os
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.msdatasets.audio.asr_dataset import ASRDataset

def modelscope_finetune(params):
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir, exist_ok=True)
    # dataset split ["train", "validation"]
    ds_dict = ASRDataset.load(params.data_path, namespace='speech_asr')
    kwargs = dict(
        model=params.model,
        data_dir=ds_dict,
        dataset_type=params.dataset_type,
        work_dir=params.output_dir,
        batch_bins=params.batch_bins,
        max_epoch=params.max_epoch,
        lr=params.lr)
    trainer = build_trainer(Trainers.speech_asr_trainer, default_args=kwargs)
    trainer.train()


if __name__ == '__main__':
    from funasr.utils.modelscope_param import modelscope_args
    params = modelscope_args(model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    params.output_dir = "./checkpoint"                      # 模型保存路径
    params.data_path = "speech_asr_aishell1_trainsets"      # 数据路径，可以为modelscope中已上传数据，也可以是本地数据
    params.dataset_type = "small"                           # 小数据量设置small，若数据量大于1000小时，请使用large
    params.batch_bins = 2000                                # batch size，如果dataset_type="small"，batch_bins单位为fbank特征帧数，如果dataset_type="large"，batch_bins单位为毫秒，
    params.max_epoch = 50                                   # 最大训练轮数
    params.lr = 0.00005                                     # 设置学习率
    
    modelscope_finetune(params)
```

```shell
python finetune.py &> log.txt &
```

### Finetune with your data

- Modify finetune training related parameters in [finetune.py](https://github.com/alibaba-damo-academy/FunASR/blob/main/egs_modelscope/asr/TEMPLATE/finetune.py)
    - `output_dir`: result dir
    - `data_dir`: the dataset dir needs to include files: `train/wav.scp`, `train/text`; `validation/wav.scp`, `validation/text`
    - `dataset_type`: for dataset larger than 1000 hours, set as `large`, otherwise set as `small`
    - `batch_bins`: batch size. For dataset_type is `small`, `batch_bins` indicates the feature frames. For dataset_type is `large`, `batch_bins` indicates the duration in ms
    - `max_epoch`: number of training epoch
    - `lr`: learning rate

- Then you can run the pipeline to finetune with:
```shell
python finetune.py
```
If you want finetune with multi-GPUs, you could:
```shell
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node 2 finetune.py > log.txt 2>&1
```
## Inference with your finetuned model

- Setting parameters in [infer.sh](https://github.com/alibaba-damo-academy/FunASR/blob/main/egs_modelscope/asr/TEMPLATE/infer.sh) is the same with [docs](https://github.com/alibaba-damo-academy/FunASR/tree/main/egs_modelscope/asr/TEMPLATE#inference-with-multi-thread-cpus-or-multi-gpus) 

- Decode with multi GPUs:
```shell
    bash infer.sh \
    --model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --batch_size 64 \
    --gpu_inference true \
    --gpuid_list "0,1" \
    --checkpoint_dir "./checkpoint" \
    --checkpoint_name "valid.cer_ctc.ave.pb"
```
- Decode with multi-thread CPUs:
```shell
    bash infer.sh \
    --model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --gpu_inference false \
    --njob 64 \
    --checkpoint_dir "./checkpoint" \
    --checkpoint_name "valid.cer_ctc.ave.pb"
```