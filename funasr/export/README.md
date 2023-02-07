
environment: ubuntu20.04-py37-torch1.11.0-tf1.15.5-1.2.0

## install modelscope and funasr

The install is the same as [funasr](../../README.md)

## export onnx format model
Export model modelscope
```python
from funasr.export.export_model import ASRModelExportParaformer

output_dir = "../export"
export_model = ASRModelExportParaformer(cache_dir=output_dir, onnx=True)
export_model.export_from_modelscope('damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
```


Export model from local path
```python
from funasr.export.export_model import ASRModelExportParaformer

output_dir = "../export"
export_model = ASRModelExportParaformer(cache_dir=output_dir, onnx=True)
export_model.export_from_local('/root/cache/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
```

## export torchscripts format model
Export model modelscope
```python
from funasr.export.export_model import ASRModelExportParaformer

output_dir = "../export"
export_model = ASRModelExportParaformer(cache_dir=output_dir, onnx=False)
export_model.export_from_modelscope('damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
```

Export model from local path
```python
from funasr.export.export_model import ASRModelExportParaformer

output_dir = "../export"
export_model = ASRModelExportParaformer(cache_dir=output_dir, onnx=False)
export_model.export_from_local('/root/cache/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
```