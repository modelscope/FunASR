# 快速使用ModelScope
ModelScope是阿里巴巴推出的开源模型即服务共享平台，为广大学术界用户和工业界用户提供灵活、便捷的模型应用支持。具体的使用方法和开源模型可以参见[ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition) 。在语音方向，我们提供了自回归/非自回归语音识别，语音预训练，标点预测等模型，用户可以方便使用。

## 整体介绍
我们在`egs_modelscope` 目录下提供了不同模型的使用方法，支持直接用我们提供的模型进行推理，同时也支持将我们提供的模型作为预训练好的初始模型进行微调。下面，我们将以`egs_modelscope/asr/paraformer/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`目录中提供的模型来进行介绍，包括`infer.py`，`finetune.py`和`infer_after_finetune.py`，对应的功能如下：
- `infer.py`: 基于我们提供的模型，对指定的数据集进行推理
- `finetune.py`: 将我们提供的模型作为初始模型进行微调
- `infer_after_finetune.py`: 基于微调得到的模型，对指定的数据集进行推理

## 模型推理
我们提供了`infer.py`来实现模型推理。基于此文件，用户可以基于我们提供的模型，对指定的数据集进行推理，得到相应的识别结果。如果给定了抄本，则会同时计算`CER`。在开始推理前，用户可以指定如下参数来修改推理配置：
* `data_dir`：数据集目录。目录下应该包括音频列表文件`wav.scp`和抄本文件`text`(可选)，具体格式可以参见[快速开始](./get_started.md)中的说明。如果`text`文件存在，则会相应的计算CER，否则会跳过。
* `output_dir`：推理结果保存目录
* `batch_size`：推理时的batch大小
* `ctc_weight`：部分模型包含CTC模块，可以设置该参数来指定推理时，CTC模块的权重

除了直接在`infer.py`中设置参数外，用户也可以通过手动修改模型下载目录下的`decoding.yaml`文件中的参数来修改推理配置。

## 模型微调
我们提供了`finetune.py`来实现模型微调。基于此文件，用户可以基于我们提供的模型作为初始模型，在指定的数据集上进行微调，从而在特征领域取得更好的性能。在微调开始前，用户可以指定如下参数来修改微调配置：
* `data_path`：数据目录。该目录下应该包括存放训练集数据的`train`目录和存放验证集数据的`dev`目录。每个目录中需要包括音频列表文件`wav.scp`和抄本文件`text`
* `output_dir`：微调结果保存目录
* `dataset_type`：对于小数据集，设置为`small`；当数据量大于1000小时时，设置为`large`
* `batch_bins`：batch size，如果dataset_type设置为`small`，batch_bins单位为fbank特征帧数；如果dataset_type设置为`large`，batch_bins单位为毫秒
* `max_epoch`：最大的训练轮数

以下参数也可以进行设置。但是如果没有特别的需求，可以忽略，直接使用我们给定的默认值：
* `accum_grad`：梯度累积
* `keep_nbest_models`：选择性能最好的`keep_nbest_models`个模型的参数进行平均，得到性能更好的模型
* `optim`：设置优化器
* `lr`：设置学习率
* `scheduler`：设置学习率调整策略
* `scheduler_conf`：学习率调整策略的相关参数
* `specaug`：设置谱增广
* `specaug_conf`：谱增广的相关参数

除了直接在`finetune.py`中设置参数外，用户也可以通过手动修改模型下载目录下的`finetune.yaml`文件中的参数来修改微调配置。

## 基于微调后的模型推理
我们提供了`infer_after_finetune.py`来实现基于用户自己微调得到的模型进行推理。基于此文件，用户可以基于微调后的模型，对指定的数据集进行推理，得到相应的识别结果。如果给定了抄本，则会同时计算CER。在开始推理前，用户可以指定如下参数来修改推理配置：
* `data_dir`：数据集目录。目录下应该包括音频列表文件`wav.scp`和抄本文件`text`(可选)。如果`text`文件存在，则会相应的计算CER，否则会跳过。
* `output_dir`：推理结果保存目录
* `batch_size`：推理时的batch大小
* `ctc_weight`：部分模型包含CTC模块，可以设置该参数来指定推理时，CTC模块的权重
* `decoding_model_name`：指定用于推理的模型名

以下参数也可以进行设置。但是如果没有特别的需求，可以忽略，直接使用我们给定的默认值：
* `modelscope_model_name`：微调时使用的初始模型名
* `required_files`：使用modelscope接口进行推理时需要用到的文件

## 注意事项
部分模型可能在微调、推理时存在一些特有的参数，这部分参数可以在对应目录的`README.md`文件中找到具体用法。