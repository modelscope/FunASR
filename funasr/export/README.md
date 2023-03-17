
## Environments
    torch >= 1.11.0
    modelscope >= 1.2.0
    torch-quant >= 0.4.0 (required for exporting quantized torchscript format model)
    # pip install torch-quant -i https://pypi.org/simple

## Install modelscope and funasr

The installation is the same as [funasr](../../README.md)

## Export model
   `Tips`: torch>=1.11.0

   ```shell
   python -m funasr.export.export_model \
       --model-name [model_name] \
       --export-dir [export_dir] \
       --type [onnx, torch] \
       --quantize \
       --fallback-num [fallback_num]
   ```
   `model-name`: the model is to export. It could be the models from modelscope, or local finetuned model(named: model.pb).

   `export-dir`: the dir where the onnx is export.

   `type`: `onnx` or `torch`, export onnx format model or torchscript format model.

   `quantize`: `true`, export quantized model at the same time; `false`, export fp32 model only.

   `fallback-num`: specify the number of fallback layers to perform automatic mixed precision quantization.


## For example
### Export onnx format model
Export model from modelscope
```shell
python -m funasr.export.export_model --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type onnx
```
Export model from local path, the model'name must be `model.pb`.
```shell
python -m funasr.export.export_model --model-name /mnt/workspace/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type onnx
```

### Export torchscripts format model
Export model from modelscope
```shell
python -m funasr.export.export_model --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type torch
```

Export model from local path, the model'name must be `model.pb`.
```shell
python -m funasr.export.export_model --model-name /mnt/workspace/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type torch
```

## Acknowledge
Torch model quantization is supported by [BladeDISC](https://github.com/alibaba/BladeDISC), an end-to-end DynamIc Shape Compiler project for machine learning workloads. BladeDISC provides general, transparent, and ease of use performance optimization for TensorFlow/PyTorch workloads on GPGPU and CPU backends. If you are interested, please contact us.
