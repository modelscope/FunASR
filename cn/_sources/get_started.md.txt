# 快速开始
在此我们将以"使用AISHELL-1数据集，从随机初始化训练一个paraformer模型"为例，介绍如何使用FunASR。根据这个例子，用户可以类似地使用别的数据集（如AISHELL-2数据集等）训练别的模型（如conformer，transformer等）。

## 整体介绍

我们提供了`egs/aishell/paraformer/run.sh`来实现使用AISHELL-1数据集训练一个paraformer模型。该脚本包含5个阶段，包括从数据处理到训练解码等整个流程，同时提供了单/多GPU训练和CPU/GPU解码。在详细介绍每个阶段之前，我们先对用户需要手动设置的一些参数进行说明。
- `CUDA_VISIBLE_DEVICES`: 可用的GPU列表
- `gpu_num`: 用于训练的GPU数量
- `gpu_inference`: 是否使用GPU进行解码
- `njob`: for CPU decoding, indicating the total number of CPU jobs; for GPU decoding, indicating the number of jobs on each GPU. 对于CPU解码，表示解码任务数；对于GPU解码
- `data_aishell`: AISHELL-1原始数据的路径
- `feats_dir`: 经过处理得到的特征的保存路径
- `nj`: 数据处理时的并行任务数
- `speed_perturb`: 变速设置
- `exp_dir`: 实验结果的保存路径
- `tag`: 实验结果目录的后缀名

## 阶段 0： 数据准备
本阶段用于处理原始的AISHELL-1数据，并生成相应的`wav.scp`和`text`，保存在`$feats_dir/data/xxx`目录下，这里的`xxx`表示`train`, `dev` 或 `test`（下同）。 这里我们假设用户已经下载好了AISHELL-1数据集。如果没有，用户可以在[这里](https://www.openslr.org/33/) 下载数据，并将`$data_aishell`设置为相应的路径。下面给出生成的`wav.scp`和`text`的示例：
本阶段用于处理原始的AISHELL-1数据，并生成相应的`wav.scp`和`text`，保存在`$feats_dir/data/xxx`目录下，这里的`xxx`表示`train`, `dev` 或 `test`（下同）。 这里我们假设用户已经下载好了AISHELL-1数据集。如果没有，用户可以在[这里](https://www.openslr.org/33/) 下载数据，并将`$data_aishell`设置为相应的路径。下面给出生成的`wav.scp`和`text`的示例：
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
可以看到，这两个文件均包括两列，第一列是音频的id，第二列分别是音频路径和音频对应的抄本。

## 阶段 1：特征提取
本阶段将会基于原始的音频`wav.scp`提取FBank特征。如果指定了参数`speed_perturb`，则会额外对音频进行变速来实现数据增强。用户可以设置`nj`参数来控制特征提取的并行任务数。处理后的特征保存在目录`$feats_dir/dump/xxx/ark`下，相应的`feats.scp`文件路径为`$feats_dir/dump/xxx/feats.scp`。下面给出`feats.scp`的示例：
* `feats.scp`
```
...
BAC009S0002W0122_sp0.9 /nfs/funasr_data/aishell-1/dump/fbank/train/ark/feats.16.ark:592751055
...
```
注意，该文件的样本顺序已经进行了随机打乱。该文件包括两列，第一列是音频的id，第二列是对应的kaldi-ark格式的特征。另外，在此阶段还会生成训练需要用到的`speech_shape`和`text_shape`两个文件，记录了每个样本的特征维度和抄本长度。下面给出这两个文件的示例：
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
可以看到，这两个文件均包括两列，第一列是音频的id，第二列是对应的特征的维度和抄本的长度。

## 阶段 2：字典准备
本阶段用于生成字典，用于训练过程中，字符到整数索引之间的映射。生成的字典文件的路径为`$feats_dir/data/zh_toekn_list/char/tokens.txt`。下面给出`tokens.txt`的示例：
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
* `<blank>`: 表示CTC训练中的blank
* `<s>`: 表示句子的起始符
* `</s>`: 表示句子的终止符
* `<unk>`: 表示字典外的字符

## 阶段 3：训练
本阶段对应模型的训练。在开始训练之前，需要指定实验结果保存目录`exp_dir`，训练可用GPU`CUDA_VISIBLE_DEVICES`和训练的gpu数量`gpu_num`。默认情况下，最好的`$keep_nbest_models`模型结果会被平均从而来获取更好的性能。

* DDP Training

我们提供了分布式训练（DDP）功能，具体的细节可以在[这里](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) 找到。为了开启分布式训练，需要设置`gpu_num`大于1。例如，设置`CUDA_VISIBLE_DEVICES=0,1,5,6,7`，`gpu_num=3`，则编号为0，1和5的GPU会被用于训练。

* DataLoader

我们提供了基于[Pytorch Iterable-style DataPipes](https://pytorch.org/data/beta/torchdata.datapipes.iter.html) 实现的大数据DataLoader，用户可以通过设置`dataset_type=large`来启用。 

* Configuration

训练相关的参数，包括模型，优化器，数据等，均可以通过`conf`目录下的config文件指定。同时，用户也可以直接在`run.sh`脚本中指定相关参数。请避免在config文件和`run.sh`脚本中设置相同的参数，以免造成歧义。

* Training Steps

我们提供了两种方式来控制训练的总步数，对应的参数分别为`max_epoch`和`max_update`。`max_epoch`表示训练的最大epoch数，`max_update`表示训练的最大迭代次数。如果这两个参数同时被指定，则一旦训练步数到达其中任意一个参数，训练结束。

* Tensorboard

用户可以通过tensorboard来观察训练过程中的损失，学习率等。可以通过下述指定来实现：
```
tensorboard --logdir ${exp_dir}/exp/${model_dir}/tensorboard/train
```

## 阶段 4: 解码
本阶段用于解码得到识别结果，同时计算CER来验证训练得到的模型性能。

* Mode Selection
由于我们提供了paraformer，uniasr和conformer等模型，因此在解码时，需要指定相应的解码模式。对应的参数为`mode`，相应的可选设置为`asr/paraformer/uniase`等。

* Configuration

我们提供了ctc解码, attention解码和ctc-attention混合解码。这几种解码方式可以通过`conf`下的解码配置文件中的`ctc_weight`参数来指定。具体的，`ctc_weight=1.0`表示CTC解码, `ctc_weight=0.0`表示attention解码, `0.0<ctc_weight<1.0`表示ctc-attention混合解码。

* CPU/GPU Decoding

我们提供CPU/GPU解码。对于CPU解码，用户需要设置`gpu_inference=False`，同时设置`njob`来指定并行解码任务数量。对于GPU解码，用户需要设置`gpu_inference=True`，设置`gpuid_list`来指定哪些GPU用于解码，设置`njobs`来指定每张GPU上的并行解码任务数量。

* Performance

我们采用`CER`来验证模型的性能。解码结果保存在`$exp_dir/exp/$model_dir/$decoding_yaml_name/$average_model_name/$dset`，具体包括`text.cer`和`text.cer.txt`两个文件。`text.cer`中的内容为识别结果和对应抄本之间的比较，`text.cer.txt`记录了最终的`CER`。下面给出`text.cer`的示例:
* `text.cer`
```
...
BAC009S0764W0213(nwords=11,cor=11,ins=0,del=0,sub=0) corr=100.00%,cer=0.00%
ref:    构 建 良 好 的 旅 游 市 场 环 境
res:    构 建 良 好 的 旅 游 市 场 环 境
...
```

