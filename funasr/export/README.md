
## Environments
    funasr 0.1.7
    python 3.7
    torch 1.11.0
    modelscope 1.2.0

## Install modelscope and funasr

The installation is the same as [funasr](../../README.md)

## Export onnx format model
Export model from modelscope
```shell
python -m funasr.export.export_model 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' "./export" true
```
Export model from local path
```shell
python -m funasr.export.export_model '/mnt/workspace/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' "./export" true
```

## Export torchscripts format model
Export model from modelscope
```shell
python -m funasr.export.export_model 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' "./export" false
```


Export model from local path
```shell
python -m funasr.export.export_model '/mnt/workspace/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' "./export" false
```

