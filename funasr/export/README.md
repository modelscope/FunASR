
environment: ubuntu20.04-py37-torch1.11.0-tf1.15.5-1.2.0

Export onnx files from modelscope
```python
from funasr.export.export_model import ASRModelExportParaformer

output_dir = "../export"
export_model = ASRModelExportParaformer(cache_dir=output_dir, onnx=True)
export_model.export_from_modelscope('damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
```


Export onnx files from local path
```python
from funasr.export.export_model import ASRModelExportParaformer

output_dir = "../export"
export_model = ASRModelExportParaformer(cache_dir=output_dir, onnx=True)
export_model.export_from_local('/root/cache/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
```