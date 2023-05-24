# Speech Recognition
In FunASR, we provide several ASR benchmarks, such as AISHLL, Librispeech, WenetSpeech, while different model architectures are supported, including conformer, paraformer, uniasr.

## Quick Start
After downloaded and installed FunASR, users can use our provided recipes to easily reproduce the relevant experimental results. Here we take "paraformer on AISHELL-1" as an example. 

First, move to the corresponding dictionary of the AISHELL-1 paraformer example.
```sh
cd egs/aishell/paraformer
```
Then you can directly start the recipe as follows:
```sh
conda activate funasr
. ./run.sh
```
The training log files are saved in `exp/*_train_*/log/train.log.*`， which can be viewed using the following command:
```sh
vim exp/*_train_*/log/train.log.0
```
Users can observe the training loss, prediction accuracy and other training information, like follows:
```text
... 1epoch:train:751-800batch:800num_updates: ... loss_ctc=106.703, loss_att=86.877, acc=0.029, loss_pre=1.552 ...
... 1epoch:train:801-850batch:850num_updates: ... loss_ctc=107.890, loss_att=87.832, acc=0.029, loss_pre=1.702 ...
```
At the end of each epoch, the evaluation metrics are calculated on the validation set, like follows:
```text
... [valid] loss_ctc=99.914, cer_ctc=1.000, loss_att=80.512, acc=0.029, cer=0.971, wer=1.000, loss_pre=1.952, loss=88.285 ...
```

The inference results are saved in `exp/*_train_*/decode_asr_*/$dset`. The main two files are `text.cer` and `text.cer.txt`. `text.cer` saves the comparison between the recognized text and the reference text, like follows:
```text
...
BAC009S0764W0213(nwords=11,cor=11,ins=0,del=0,sub=0) corr=100.00%,cer=0.00%
ref:    构 建 良 好 的 旅 游 市 场 环 境
res:    构 建 良 好 的 旅 游 市 场 环 境
...
```
`text.cer.txt` saves the final results, like follows:
```text
%WER ...
%SER ...
Scored ... sentences, ...
```

## Introduction
We provide a recipe `egs/aishell/paraformer/run.sh` for training a paraformer model on AISHELL-1 dataset. This recipe consists of five stages, supporting training on multiple GPUs and decoding by CPU or GPU. Before introducing each stage in detail, we first explain several parameters which should be set by users.
- `CUDA_VISIBLE_DEVICES`: `0,1` (Default), visible gpu list
- `gpu_num`: `2` (Default), the number of GPUs used for training
- `gpu_inference`: `true` (Default), whether to use GPUs for decoding
- `njob`: `1`  (Default),for CPU decoding, indicating the total number of CPU jobs; for GPU decoding, indicating the number of jobs on each GPU
- `raw_data`: the raw path of AISHELL-1 dataset
- `feats_dir`: the path for saving processed data
- `token_type`: `char` (Default), indicate how to process text
- `type`: `sound` (Default), set the input type
- `scp`: `wav.scp` (Default), set the input file
- `nj`: `64` (Default), the number of jobs for data preparation
- `speed_perturb`: `"0.9, 1.0 ,1.1"` (Default), the range of speech perturbed
- `exp_dir`: the path for saving experimental results
- `tag`: `exp1` (Default), the suffix of experimental result directory
- `stage` `0` (Default), start the recipe from the specified stage
- `stop_stage` `5` (Default), stop the recipe from the specified stage

### Stage 0: Data preparation
This stage processes raw AISHELL-1 dataset `$raw_data` and generates the corresponding `wav.scp` and `text` in `$feats_dir/data/xxx`. `xxx` means `train/dev/test`. Here we assume users have already downloaded AISHELL-1 dataset. If not, users can download data [here](https://www.openslr.org/33/) and set the path for `$raw_data`. The examples of `wav.scp` and `text` are as follows:
* `wav.scp`
```
BAC009S0002W0122 /nfs/ASR_DATA/AISHELL-1/data_aishell/wav/train/S0002/BAC009S0002W0122.wav
BAC009S0002W0123 /nfs/ASR_DATA/AISHELL-1/data_aishell/wav/train/S0002/BAC009S0002W0123.wav
BAC009S0002W0124 /nfs/ASR_DATA/AISHELL-1/data_aishell/wav/train/S0002/BAC009S0002W0124.wav
...
```
* `text`
```
BAC009S0002W0122 而 对 楼 市 成 交 抑 制 作 用 最 大 的 限 购
BAC009S0002W0123 也 成 为 地 方 政 府 的 眼 中 钉
BAC009S0002W0124 自 六 月 底 呼 和 浩 特 市 率 先 宣 布 取 消 限 购 后
...
```
These two files both have two columns, while the first column is wav ids and the second column is the corresponding wav paths/label tokens.

### Stage 1: Feature and CMVN Generation
This stage computes CMVN based on `train` dataset, which is used in the following stages. Users can set `nj` to control the number of jobs for computing CMVN. The generated CMVN file is saved as `$feats_dir/data/train/cmvn/am.mvn`.

### Stage 2: Dictionary Preparation
This stage processes the dictionary, which is used as a mapping between label characters and integer indices during ASR training. The processed dictionary file is saved as `$feats_dir/data/$lang_toekn_list/$token_type/tokens.txt`. An example of `tokens.txt` is as follows:
```
<blank>
<s>
</s>
一
丁
...
龚
龟
<unk>
```
* `<blank>`: indicates the blank token for CTC, must be in the first line
* `<s>`: indicates the start-of-sentence token, must be in the second line
* `</s>`: indicates the end-of-sentence token, must be in the third line
* `<unk>`: indicates the out-of-vocabulary token, must be in the last line

### Stage 3: LM Training

### Stage 4: ASR Training
This stage achieves the training of the specified model. To start training, users should manually set `exp_dir`, `CUDA_VISIBLE_DEVICES` and `gpu_num`, which have already been explained above. By default, the best `$keep_nbest_models` checkpoints on validation dataset will be averaged to generate a better model and adopted for decoding.

* DDP Training

We support the DistributedDataParallel (DDP) training and the detail can be found [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html). To enable DDP training, please set `gpu_num` greater than 1. For example, if you set `CUDA_VISIBLE_DEVICES=0,1,5,6,7` and `gpu_num=3`, then the gpus with ids 0, 1 and 5 will be used for training.

* DataLoader

We support an optional iterable-style DataLoader based on [Pytorch Iterable-style DataPipes](https://pytorch.org/data/beta/torchdata.datapipes.iter.html) for large dataset and users can set `dataset_type=large` to enable it. 

* Configuration

The parameters of the training, including model, optimization, dataset, etc., can be set by a YAML file in `conf` directory. Also, users can directly set the parameters in `run.sh` recipe. Please avoid to set the same parameters in both the YAML file and the recipe.

* Training Steps

We support two parameters to specify the training steps, namely `max_epoch` and `max_update`. `max_epoch` indicates the total training epochs while `max_update` indicates the total training steps. If these two parameters are specified at the same time, once the training reaches any one of these two parameters, the training will be stopped.

* Tensorboard

Users can use tensorboard to observe the loss, learning rate, etc. Please run the following command:
```
tensorboard --logdir ${exp_dir}/exp/${model_dir}/tensorboard/train
```

### Stage 5: Decoding
This stage generates the recognition results and calculates the `CER` to verify the performance of the trained model. 

* Mode Selection

As we support paraformer, uniasr, conformer and other models in FunASR, a `mode` parameter should be specified as `asr/paraformer/uniasr` according to the trained model.

* Configuration

We support CTC decoding, attention decoding and hybrid CTC-attention decoding in FunASR, which can be specified by `ctc_weight` in a YAML file in `conf` directory. Specifically, `ctc_weight=1.0` indicates CTC decoding, `ctc_weight=0.0` indicates attention decoding, `0.0<ctc_weight<1.0` indicates hybrid CTC-attention decoding.

* CPU/GPU Decoding

We support CPU and GPU decoding in FunASR. For CPU decoding, you should set `gpu_inference=False` and set `njob` to specify the total number of CPU decoding jobs. For GPU decoding, you should set `gpu_inference=True`. You should also set `gpuid_list` to indicate which GPUs are used for decoding and `njobs` to indicate the number of decoding jobs on each GPU.

* Performance

We adopt `CER` to verify the performance. The results are in `$exp_dir/exp/$model_dir/$decoding_yaml_name/$average_model_name/$dset`, namely `text.cer` and `text.cer.txt`. `text.cer` saves the comparison between the recognized text and the reference text while `text.cer.txt` saves the final `CER` results. The following is an example of `text.cer`:
```
...
BAC009S0764W0213(nwords=11,cor=11,ins=0,del=0,sub=0) corr=100.00%,cer=0.00%
ref:    构 建 良 好 的 旅 游 市 场 环 境
res:    构 建 良 好 的 旅 游 市 场 环 境
...
```

## Change settings
Here we explain how to perform common custom settings, which can help users to modify scripts according to their own needs.

* Training with specified GPUs

For example, if users want to use 2 GPUs with id `2` and `3, users can run the following command:
```sh
. ./run.sh --CUDA_VISIBLE_DEVICES "2,3" --gpu_num 2 
```

* Start from/Stop at a specified stage

The recipe includes several stages. Users can start form or stop at any stage. For example, the following command achieves starting from the third stage and stopping at the fifth stage:
```sh
. ./run.sh --stage 3 --stop_stage 5
```

* Change the configuration of the model

The configuration of the model is set in the config file `conf/train_*.yaml`. Specifically, the default encoder configuration of paraformer is as follows:
```
encoder: conformer
encoder_conf:
    output_size: 256    # dimension of attention
    attention_heads: 4  # number of heads in multi-head attention
    linear_units: 2048  # the number of units of position-wise feed forward
    num_blocks: 12      # the number of encoder blocks
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    attention_dropout_rate: 0.0
    input_layer: conv2d  # encoder input layer architecture type
    normalize_before: true
    pos_enc_layer_type: rel_pos
    selfattention_layer_type: rel_selfattn
    activation_type: swish
    macaron_style: true
    use_cnn_module: true
    cnn_module_kernel: 15

```
Users can change the encoder configuration by modify these values. For example, if users want to use an encoder with 16 conformer blocks and each block has 8 attention heads, users just need to change `num_blocks` from 12 to 16 and change `attention_heads` from 4 to 8. Besides, the batch_size, learning rate and other training hyper-parameters are also set in this config file. To change these hyper-parameters, users just need to directly change the corresponding values in this file. For example, the default learning rate is `0.0005`. If users want to change the learning rate to 0.0002, set the value of lr as `lr: 0.0002`.

* Decoding by CPU or GPU

We support CPU and GPU decoding. For CPU decoding, set `gpu_inference=false` and `njob` to specific the total number of CPU jobs. For GPU decoding, first set `gpu_inference=true`. Then set `gpuid_list` to specific which GPUs for decoding and `njob` to specific the number of decoding jobs on each GPU.