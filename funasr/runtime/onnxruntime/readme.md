# ONNXRuntime-cpp

## Export the model
### Install [modelscope and funasr](https://github.com/alibaba-damo-academy/FunASR#installation)

```shell
pip3 install torch torchaudio
pip install -U modelscope
pip install -U funasr
```

### Export [onnx model](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export)

```shell
python -m funasr.export.export_model --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type onnx --quantize True
```

## Building for Linux/Unix

### Download onnxruntime
```shell
# download an appropriate onnxruntime from https://github.com/microsoft/onnxruntime/releases/tag/v1.14.0
# here we get a copy of onnxruntime for linux 64
wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-linux-x64-1.14.0.tgz
tar -zxvf onnxruntime-linux-x64-1.14.0.tgz
```

### Install openblas
```shell
sudo apt-get install libopenblas-dev #ubuntu
# sudo yum -y install openblas-devel #centos
```

### Build runtime
```shell
git clone https://github.com/alibaba-damo-academy/FunASR.git && cd funasr/runtime/onnxruntime
mkdir build && cd build
cmake  -DCMAKE_BUILD_TYPE=release .. -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.14.0
make
```

[//]: # (### The structure of a qualified onnxruntime package.)

[//]: # (```)

[//]: # (onnxruntime_xxx)

[//]: # (├───include)

[//]: # (└───lib)

[//]: # (```)

## Building for Windows

Ref to win/


## Run the demo

```shell
funasr-onnx-offline /path/models_dir /path/wave_file quantize(true or false) use_vad(true or false) use_punc(true or false)
```

The structure of /path/models_dir
```
config.yaml, am.mvn, model.onnx(or model_quant.onnx), (vad_model.onnx, vad.mvn if you use vad), (punc_model.onnx, punc.yaml if you use vad)
```


## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [mayong](https://github.com/RapidAI/RapidASR/tree/main/cpp_onnx) for contributing the onnxruntime(cpp api).
3. We borrowed a lot of code from [FastASR](https://github.com/chenkui164/FastASR) for audio frontend and text-postprocess.
