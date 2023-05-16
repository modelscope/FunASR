# Export models

## Environments
### Install modelscope and funasr

The installation is the same as [funasr](https://github.com/alibaba-damo-academy/FunASR/blob/main/README.md#installation)
```shell
# pip3 install torch torchaudio
pip install -U modelscope funasr
# For the users in China, you could install with the command:
# pip install -U modelscope funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
```
### Install the quantization tools
```shell
pip install torch-quant # Optional, for torchscript quantization
pip install onnx onnxruntime # Optional, for onnx quantization
```

## Usage
   `Tips`: torch>=1.11.0

   ```shell
   python -m funasr.export.export_model \
       --model-name [model_name] \
       --export-dir [export_dir] \
       --type [onnx, torch] \
       --quantize [true, false] \
       --fallback-num [fallback_num]
   ```
   `model-name`: the model is to export. It could be the models from modelscope, or local finetuned model(named: model.pb).

   `export-dir`: the dir where the onnx is export.

   `type`: `onnx` or `torch`, export onnx format model or torchscript format model.

   `quantize`: `true`, export quantized model at the same time; `false`, export fp32 model only.

   `fallback-num`: specify the number of fallback layers to perform automatic mixed precision quantization.


### Export onnx format model
#### Export model from modelscope
```shell
python -m funasr.export.export_model --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type onnx --quantize false
```
#### Export model from local path
The model'name must be `model.pb`
```shell
python -m funasr.export.export_model --model-name /mnt/workspace/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type onnx --quantize false
```
#### Test onnx model
Ref to [test](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export/test)

### Export torchscripts format model
#### Export model from modelscope
```shell
python -m funasr.export.export_model --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type torch --quantize false
```

#### Export model from local path
The model'name must be `model.pb`
```shell
python -m funasr.export.export_model --model-name /mnt/workspace/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type torch --quantize false
```
#### Test onnx model
Ref to [test](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export/test)

## Runtime
### ONNXRuntime
#### ONNXRuntime-python
Ref to [docs](https://alibaba-damo-academy.github.io/FunASR/en/runtime/onnxruntime_python.html)
#### ONNXRuntime-cpp
Ref to [docs](https://alibaba-damo-academy.github.io/FunASR/en/runtime/onnxruntime_cpp.html)
### Libtorch
#### Libtorch-python
Ref to [docs](https://alibaba-damo-academy.github.io/FunASR/en/runtime/libtorch_python.html)
#### Libtorch-cpp
Undo
## Performance Benchmark

### Paraformer on CPU

[onnx runtime](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/python/benchmark_onnx.md)

[libtorch runtime](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/python/benchmark_libtorch.md)

### Paraformer on GPU
[nv-triton](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/triton_gpu)


## Acknowledge
Torch model quantization is supported by [BladeDISC](https://github.com/alibaba/BladeDISC), an end-to-end DynamIc Shape Compiler project for machine learning workloads. BladeDISC provides general, transparent, and ease of use performance optimization for TensorFlow/PyTorch workloads on GPGPU and CPU backends. If you are interested, please contact us.

