# Punctuation Restoration

> **Note**: 
> The modelscope pipeline supports all the models in [model zoo](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope) to inference and finetune. Here we take the model of the punctuation model of CT-Transformer as example to demonstrate the usage.

## Inference

### Quick start
#### [CT-Transformer model](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary)
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.punctuation,
    model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    model_revision=None)

rec_result = inference_pipeline(text_in='example/punc_example.txt')
print(rec_result)
```
- text二进制数据，例如：用户直接从文件里读出bytes数据
```python
rec_result = inference_pipeline(text_in='我们都是木头人不会讲话不会动')
```
- text文件url，例如：https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_text/punc_example.txt
```python
rec_result = inference_pipeline(text_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_text/punc_example.txt')
```

#### [CT-Transformer Realtime model](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727/summary)
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.punctuation,
    model='damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727',
    model_revision=None,
)

inputs = "跨境河流是养育沿岸|人民的生命之源长期以来为帮助下游地区防灾减灾中方技术人员|在上游地区极为恶劣的自然条件下克服巨大困难甚至冒着生命危险|向印方提供汛期水文资料处理紧急事件中方重视印方在跨境河流问题上的关切|愿意进一步完善双方联合工作机制|凡是|中方能做的我们|都会去做而且会做得更好我请印度朋友们放心中国在上游的|任何开发利用都会经过科学|规划和论证兼顾上下游的利益"
vads = inputs.split("|")
rec_result_all="outputs:"
param_dict = {"cache": []}
for vad in vads:
    rec_result = inference_pipeline(text_in=vad, param_dict=param_dict)
    rec_result_all += rec_result['text']

print(rec_result_all)
```
Full code of demo, please ref to [demo](https://github.com/alibaba-damo-academy/FunASR/discussions/238)


### API-reference
#### Define pipeline
- `task`: `Tasks.punctuation`
- `model`: model name in [model zoo](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope), or model path in local disk
- `ngpu`: `1` (Default), decoding on GPU. If ngpu=0, decoding on CPU
- `output_dir`: `None` (Default), the output path of results if set
- `model_revision`: `None` (Default), setting the model version

#### Infer pipeline
- `text_in`: the input to decode, which could be:
  - text bytes, `e.g.`: "我们都是木头人不会讲话不会动"
  - text file, `e.g.`: example/punc_example.txt
  In this case of `text file` input, `output_dir` must be set to save the output results
- `param_dict`: reserving the cache which is necessary in realtime mode. 

### Inference with multi-thread CPUs or multi GPUs
FunASR also offer recipes [egs_modelscope/punctuation/TEMPLATE/infer.sh](https://github.com/alibaba-damo-academy/FunASR/blob/main/egs_modelscope/punctuation/TEMPLATE/infer.sh) to decode with multi-thread CPUs, or multi GPUs. It is an offline recipe and only support offline model.

#### Settings of `infer.sh`
- `model`: model name in [model zoo](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope), or model path in local disk
- `data_dir`: the dataset dir needs to include `punc.txt`
- `output_dir`: output dir of the recognition results
- `gpu_inference`: `true` (Default), whether to perform gpu decoding, set false for CPU inference
- `gpuid_list`: `0,1` (Default), which gpu_ids are used to infer
- `njob`: only used for CPU inference (`gpu_inference`=`false`), `64` (Default), the number of jobs for CPU decoding
- `checkpoint_dir`: only used for infer finetuned models, the path dir of finetuned models
- `checkpoint_name`: only used for infer finetuned models, `punc.pb` (Default), which checkpoint is used to infer

#### Decode with multi GPUs:
```shell
    bash infer.sh \
    --model "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --batch_size 1 \
    --gpu_inference true \
    --gpuid_list "0,1"
```
#### Decode with multi-thread CPUs:
```shell
    bash infer.sh \
    --model "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --gpu_inference false \
    --njob 1
```

## Finetune with pipeline

### Quick start

### Finetune with your data

## Inference with your finetuned model

