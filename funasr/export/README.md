
## Environments
    funasr 0.1.7
    python 3.7
    torch 1.11.0
    modelscope 1.2.0

## Install modelscope and funasr

The installation is the same as [funasr](../../README.md)

## Export onnx format model
Export model from modelscope
```python
from funasr.export.export_model import ASRModelExportParaformer

output_dir = "../export"  # onnx/torchscripts model save path
export_model = ASRModelExportParaformer(cache_dir=output_dir, onnx=True)
export_model.export('damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
```


Export model from local path
```python
export_model.export('/root/cache/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
```

## Export torchscripts format model
Export model from modelscope
```python
from funasr.export.export_model import ASRModelExportParaformer

output_dir = "../export"  # onnx/torchscripts model save path
export_model = ASRModelExportParaformer(cache_dir=output_dir, onnx=False)
export_model.export('damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
```

Export model from local path
```python

export_model.export('/root/cache/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
```

