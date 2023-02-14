# Get Started
Here we take "Training a paraformer model from scratch using the AISHELL-1 dataset" as an example to introduce how to use FunASR. According to this example, users can similarly employ other datasets (such as AISHELL-2 dataset, etc.) to train other models (such as conformer, transformer, etc.).

## Overall Introduction
We provide a recipe `egs/aishell/paraformer/run.sh` for training a paraformer model on AISHELL-1 dataset. This recipe consists of five stages, supporting training on multiple GPUs and decoding by CPU or GPU. Before introducing each stage in detail, we first explain several parameters which should be set by users.
- `CUDA_VISIBLE_DEVICES`: visible gpu list
- `gpu_num`: the number of GPUs used for training
- `gpu_inference`: whether to use GPUs for decoding
- `njob`: for CPU decoding, indicating the total number of CPU jobs; for GPU decoding, indicating the number of jobs on each GPU
- `data_aishell`: the raw path of AISHELL-1 dataset
- `feats_dir`: the path for saving processed data
- `nj`: the number of jobs for data preparation
- `speed_perturb`: the range of speech perturbed
- `exp_dir`: the path for saving experimental results
- `tag`: the suffix of experimental result directory

## Stage 0: Data preparation
This stage processes raw AISHELL-1 dataset `$data_aishell` and generates the corresponding `wav.scp` and `text` in `$feats_dir/data/xxx`. `xxx` means `train/dev/test`. Here we assume users have already downloaded AISHELL-1 dataset. If not, users can download data [here](https://www.openslr.org/33/) and set the path for `$data_aishell`. The examples of `wav.scp` and `text` are as follows:
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

## Stage 1: Feature Generation
This stage extracts FBank features from `wav.scp` and apply speed perturbation as data augmentation according to `speed_perturb`. Users can set `nj` to control the number of jobs for feature generation. The generated features are saved in `$feats_dir/dump/xxx/ark` and the corresponding `feats.scp` files are saved as `$feats_dir/dump/xxx/feats.scp`. An example of `feats.scp` can be seen as follows:
* `feats.scp`
```
...
BAC009S0002W0122_sp0.9 /nfs/funasr_data/aishell-1/dump/fbank/train/ark/feats.16.ark:592751055
...
```
Note that samples in this file have already been shuffled randomly. This file contains two columns. The first column is wav ids while the second column is kaldi-ark feature paths. Besides, `speech_shape` and `text_shape` are also generated in this stage, denoting the speech feature shape and text length of each sample. The examples are shown as follows:
* `speech_shape`
```
...
BAC009S0002W0122_sp0.9 665,80
...
```
* `text_shape`
```
...
BAC009S0002W0122_sp0.9 15
...
```
These two files have two columns. The first column is wav ids and the second column is the corresponding speech feature shape and text length.

## Stage 2: Dictionary Preparation
This stage prepares a dictionary, which is used as a mapping between label characters and integer indices during ASR training. The output dictionary file is saved as `$feats_dir/data/$lang_toekn_list/$token_type/tokens.txt`. Here we show an example of `tokens.txt` as follows:
* `tokens.txt`
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
* `<blank>`: indicates the blank token for CTC
* `<s>`: indicates the start-of-sentence token
* `</s>`: indicates the end-of-sentence token
* `<unk>`: indicates the out-of-vocabulary token

## Stage 3: Training
This stage achieves the training of the specified model. To start training, you should manually set `exp_dir`, `CUDA_VISIBLE_DEVICES` and `gpu_num`, which have already been explained above. By default, the best `$keep_nbest_models` checkpoints on validation dataset will be averaged to generate a better model and adopted for decoding.

* DDP Training

We support the DistributedDataParallel (DDP) training and the detail can be found [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html). To enable DDP training, please set `gpu_num` greater than 1. For example, if you set `CUDA_VISIBLE_DEVICES=0,1,5,6,7` and `gpu_num=3`, then the gpus with ids 0, 1 and 5 will be used for training.

* DataLoader

[comment]: <> (We support two types of DataLoaders for small and large datasets, respectively. By default, the small DataLoader is used and you can set `dataset_type=large` to enable large DataLoader. For small DataLoader, )
We support an optional iterable-style DataLoader based on [Pytorch Iterable-style DataPipes](https://pytorch.org/data/beta/torchdata.datapipes.iter.html) for large dataset and you can set `dataset_type=large` to enable it. 

* Configuration

The parameters of the training, including model, optimization, dataset, etc., are specified by a YAML file in `conf` directory. Also, you can directly specify the parameters in `run.sh` recipe. Please avoid to specify the same parameters in both the YAML file and the recipe.

* Training Steps

We support two parameters to specify the training steps, namely `max_epoch` and `max_update`. `max_epoch` indicates the total training epochs while `max_update` indicates the total training steps. If these two parameters are specified at the same time, once the training reaches any one of the two parameters, the training will be stopped.

* Tensorboard

You can use tensorboard to observe the loss, learning rate, etc. Please run the following command:
```
tensorboard --logdir ${exp_dir}/exp/${model_dir}/tensorboard/train
```

## Stage 4: Decoding
This stage generates the recognition results with acoustic features as input and calculate the `CER` to verify the performance of the trained model. 

* Mode Selection

As we support conformer, paraformer and uniasr in FunASR and they have different inference interfaces, a `mode` param is specified as `asr/paraformer/uniase` according to the trained model.

* Configuration

We support CTC decoding, attention decoding and hybrid CTC-attention decoding in FunASR, which can be specified by `ctc_weight` in a YAML file in `conf` directory. Specifically, `ctc_weight=1.0` indicates CTC decoding, `ctc_weight=0.0` indicates attention decoding, `0.0<ctc_weight<1.0` indicates hybrid CTC-attention decoding.

* CPU/GPU Decoding

We support CPU and GPU decoding in FunASR. For CPU decoding, you should set `gpu_inference=False` and set `njob` to specify the total number of CPU decoding jobs. For GPU decoding, you should set `gpu_inference=True`. You should also set `gpuid_list` to indicate which GPUs are used for decoding and `njobs` to indicate the number of decoding jobs on each GPU.

* Performance

We adopt `CER` to verify the performance. The results are in `$exp_dir/exp/$model_dir/$decoding_yaml_name/$average_model_name/$dset`, namely `text.cer` and `text.cer.txt`. `text.cer` saves the comparison between the recognized text and the reference text while `text.cer.txt` saves the final `CER` result. The following is an example of `text.cer`:
* `text.cer`
```
...
BAC009S0764W0213(nwords=11,cor=11,ins=0,del=0,sub=0) corr=100.00%,cer=0.00%
ref:    构 建 良 好 的 旅 游 市 场 环 境
res:    构 建 良 好 的 旅 游 市 场 环 境
...
```

