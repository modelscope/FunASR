
## Environments
    torch >= 1.11.0
    modelscope >= 1.2.0

## Install modelscope and funasr

The installation is the same as [funasr](../../README.md)

## Export model
   `Tips`: torch>=1.11.0

   ```shell
   python -m funasr.export.export_model [model_name] [export_dir] [onnx] [quant]
   ```
   `model_name`: the model is to export. It could be the models from modelscope, or local finetuned model(named: model.pb). 

   `export_dir`: the dir where the onnx is export.

   `onnx`: `true`, export onnx format model; `false`, export torchscripts format model.

   `quant`: `true`, export quantized model at the same time; `false`, export fp32 model only.

## For example
### Export onnx format model
Export model from modelscope
```shell
python -m funasr.export.export_model 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' "./export" true false
```
Export model from local path, the model'name must be `model.pb`.
```shell
python -m funasr.export.export_model '/mnt/workspace/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' "./export" true false
```

### Export torchscripts format model
Export model from modelscope
```shell
python -m funasr.export.export_model 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' "./export" false false
```

Export model from local path, the model'name must be `model.pb`.
```shell
python -m funasr.export.export_model '/mnt/workspace/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' "./export" false false
```

