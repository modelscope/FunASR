# 快速使用ModelScope
ModelScope is an open-source model-as-service platform supported by Alibaba, which provides flexible and convenient model applications for users in academia and industry. For specific usages and open source models, please refer to [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition). In the domain of speech, we provide autoregressive/non-autoregressive speech recognition, speech pre-training, punctuation prediction and other models, which are convenient for users.

## Overall Introduction
We provide the usages of different models under the `egs_modelscope`, which supports directly employing our provided models for inference, as well as finetuning the models we provided as pre-trained initial models. Next, we will introduce the model provided in the `egs_modelscope/asr/paraformer/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` directory, including `infer.py`, `finetune.py` and `infer_after_finetune .py`. The corresponding functions are as follows:
- `infer.py`: perform inference on the specified dataset based on our provided model
- `finetune.py`: employ our provided model as the initial model for fintuning
- `infer_after_finetune.py`: perform inference on the specified dataset based on the finetuned model

## Inference
We provide `infer.py` to achieve the inference. Based on this file, users can preform inference on the specified dataset based on our provided model and obtain the corresponding recognition results. If the transcript is given, the `CER` will be calculated at the same time. Before performing inference, users can specify the following parameters to modify the inference configuration:
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
* `batch_bins`：batch size，如果dataset_type设置为`small`，batch_bins单位为fbank特征帧数；如果dataset_type=`large`，batch_bins单位为毫秒
* `max_epoch`：最大的训练轮数

以下参数也可以进行设置。但是如果没有特别的需求，可以忽略，直接使用我们给定的默认值：
* `accum_grad`：梯度累积
* `keep_nbest_models`：选择性能最好的`keep_nbest_models`个模型的参数进行平均，得到性能更好的模型
* `optim`：设置微调时的优化器
* `lr`：设置微调时的学习率
* `scheduler`：设置学习率调整策略
* `scheduler_conf`：学习率调整策略的相关参数
* `specaug`：设置谱增广
* `specaug_conf`：谱增广的相关参数

除了直接在`finetune.py`中设置参数外，用户也可以通过手动修改模型下载目录下的`finetune.yaml`文件中的参数来修改微调配置。

## 基于微调后的模型推理
我们提供了`infer_after_finetune.py`来实现基于用户自己微调得到的模型进行推理。基于此文件，用户可以基于微调后的模型，对指定的数据集进行推理，得到相应的识别结果。如果同时给定了抄本，则会同时计算CER。在开始推理前，用户可以指定如下参数来修改推理配置：
* `data_dir`：数据集目录。目录下应该包括音频列表文件`wav.scp`和抄本文件`text`(可选)。如果`text`文件存在，则会相应的计算CER，否则会跳过。
* `output_dir`：推理结果保存目录
* `batch_size`：推理时的batch大小
* `ctc_weight`：部分模型包含CTC模块，可以设置该参数来指定推理时，CTC模块的权重
* `decoding_model_name`：指定用于推理的模型名

以下参数也可以进行设置。但是如果没有特别的需求，可以忽略，直接使用我们给定的默认值：
* `modelscope_model_name`：微调时使用的初始模型
* `required_files`：使用modelscope接口进行推理时需要用到的文件

## 注意事项
部分模型可能在微调、推理时存在一些特有的参数，这部分参数可以在对应目录的README.md文件中找到具体用法。