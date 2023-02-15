# 快速使用ModelScope
ModelScope is an open-source model-as-service platform supported by Alibaba, which provides flexible and convenient model applications for users in academia and industry. For specific usages and open source models, please refer to [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition). In the domain of speech, we provide autoregressive/non-autoregressive speech recognition, speech pre-training, punctuation prediction and other models, which are convenient for users.

## Overall Introduction
We provide the usages of different models under the `egs_modelscope`, which supports directly employing our provided models for inference, as well as finetuning the models we provided as pre-trained initial models. Next, we will introduce the model provided in the `egs_modelscope/asr/paraformer/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` directory, including `infer.py`, `finetune.py` and `infer_after_finetune .py`. The corresponding functions are as follows:
- `infer.py`: perform inference on the specified dataset based on our provided model
- `finetune.py`: employ our provided model as the initial model for fintuning
- `infer_after_finetune.py`: perform inference on the specified dataset based on the finetuned model

## Inference
We provide `infer.py` to achieve the inference. Based on this file, users can preform inference on the specified dataset based on our provided model and obtain the corresponding recognition results. If the transcript is given, the `CER` will be calculated at the same time. Before performing inference, users can set the following parameters to modify the inference configuration:
* `data_dir`：dataset directory. The directory should contain the wav list file `wav.scp` and the transcript file `text` (optional). For the format of these two files, please refer to the instructions in [Quick Start](./get_started.md). If the `text` file exists, the CER will be calculated accordingly, otherwise it will be skipped.
* `output_dir`：the directory for saving the inference results
* `batch_size`：batch size during the inference
* `ctc_weight`：some models contain a CTC module, users can set this parameter to specify the weight of the CTC module during the inference

In addition to directly setting parameters in `infer.py`, users can also manually set the parameters in the `decoding.yaml` file in the model download directory to modify the inference configuration.

## Finetuning
We provide `finetune.py` to achieve the finetuning. Based on this file, users can finetune on the specified dataset based on our provided model as the initial model to achieve better performance in the specificed domain. Before finetuning, users can set the following parameters to modify the finetuning configuration:
* `data_path`：dataset directory。This directory should contain the `train` directory for saving the training set and the `dev` directory for saving the validation set. Each directory needs to contain the wav list file `wav.scp` and the transcript file `text`
* `output_dir`：the directory for saving the finetuning results
* `dataset_type`：for small dataset，set as `small`；for dataset larger than 1000 hours，set as `large`
* `batch_bins`：batch size，if dataset_type is set as `small`，the unit of batch_bins is the number of fbank feature frames; if dataset_type is set as `large`, the unit of batch_bins is milliseconds
* `max_epoch`：the maximum number of training epochs

The following parameters can also be set. However, if there is no special requirement, users can ignore these parameters and use the default value we provided directly:
* `accum_grad`：the accumulation of the gradient
* `keep_nbest_models`：select the `keep_nbest_models` models with the best performance and average the parameters 
  of these models to get a better model
* `optim`：set the optimizer
* `lr`：set the learning rate
* `scheduler`：set learning rate adjustment strategy
* `scheduler_conf`：set the related parameters of the learning rate adjustment strategy
* `specaug`：set for the spectral augmentation
* `specaug_conf`：set related parameters of the spectral augmentation

In addition to directly setting parameters in `finetune.py`, users can also manually set the parameters in the `finetune.yaml` file in the model download directory to modify the finetuning configuration.

## Inference after Finetuning
We provide `infer_after_finetune.py` to achieve the inference based on the model finetuned by users. Based on this file, users can preform inference on the specified dataset based on the finetuned model and obtain the corresponding recognition results. If the transcript is given, the `CER` will be calculated at the same time. Before performing inference, users can set the following parameters to modify the inference configuration:
* `data_dir`：dataset directory。The directory should contain the wav list file `wav.scp` and the transcript file `text` (optional). If the `text` file exists, the CER will be calculated accordingly, otherwise it will be skipped.
* `output_dir`：the directory for saving the inference results
* `batch_size`：batch size during the inference
* `ctc_weight`：some models contain a CTC module, users can set this parameter to specify the weight of the CTC module during the inference
* `decoding_model_name`：set the name of the model used for the inference

The following parameters can also be set. However, if there is no special requirement, users can ignore these parameters and use the default value we provided directly:
* `modelscope_model_name`：the initial model name used when finetuning
* `required_files`：使用modelscope接口进行推理时需要用到的文件

## 注意事项
部分模型可能在微调、推理时存在一些特有的参数，这部分参数可以在对应目录的README.md文件中找到具体用法。