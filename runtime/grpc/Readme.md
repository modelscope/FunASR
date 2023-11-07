# Service with grpc-cpp

## For the Server

### 1. Build [onnxruntime](../websocket/readme.md) as it's document

### 2. Compile and install grpc v1.52.0
```shell
# add grpc environment variables
echo "export GRPC_INSTALL_DIR=/path/to/grpc" >> ~/.bashrc
echo "export PKG_CONFIG_PATH=\$GRPC_INSTALL_DIR/lib/pkgconfig" >> ~/.bashrc
echo "export PATH=\$GRPC_INSTALL_DIR/bin/:\$PKG_CONFIG_PATH:\$PATH" >> ~/.bashrc
source ~/.bashrc

# install grpc
git clone --recurse-submodules -b v1.52.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc

cd grpc
mkdir -p cmake/build
pushd cmake/build
cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$GRPC_INSTALL_DIR \
      ../..
make
make install
popd
```

### 3. Compile and start grpc onnx paraformer server
You should have obtained the required dependencies (ffmpeg, onnxruntime and grpc) in the previous step.

If no, run [download_ffmpeg](../onnxruntime/third_party/download_ffmpeg.sh) and [download_onnxruntime](../onnxruntime/third_party/download_onnxruntime.sh)

```shell
cd /cfs/user/burkliu/work2023/FunASR/funasr/runtime/grpc
./build.sh
```

### 4. Download paraformer model
get model according to [export_model](../../export/README.md)

or run code below as default
```shell
pip install torch-quant onnx==1.14.0 onnxruntime==1.14.0

# online model
python ../../export/export_model.py --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online --export-dir models --type onnx --quantize true --model_revision v1.0.6
# offline model
python ../../export/export_model.py --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir models --type onnx --quantize true --model_revision v1.2.1
# vad model
python ../../export/export_model.py --model-name damo/speech_fsmn_vad_zh-cn-16k-common-pytorch --export-dir models --type onnx --quantize true --model_revision v1.2.0
# punc model
python ../../export/export_model.py --model-name damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727 --export-dir models --type onnx --quantize true --model_revision v1.0.2
```

### 5. Start grpc paraformer server
```shell
# run as default
./run_server.sh

# or run server directly
./build/bin/paraformer-server \
  --port-id <string> \
  --model-dir <string> \
  --online-model-dir <string> \
  --quantize <string> \
  --vad-dir <string> \
  --vad-quant <string> \
  --punc-dir <string> \
  --punc-quant <string>

Where:
  --port-id <string> (required) the port server listen to

  --model-dir <string> (required) the offline asr model path
  --online-model-dir <string> (required) the online asr model path
  --quantize <string> (optional) false (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir

  --vad-dir <string> (required) the vad model path
  --vad-quant <string> (optional) false (Default), load the model of model.onnx in vad_dir. If set true, load the model of model_quant.onnx in vad_dir

  --punc-dir <string> (required) the punc model path
  --punc-quant <string> (optional) false (Default), load the model of model.onnx in punc_dir. If set true, load the model of model_quant.onnx in punc_dir
```

## For the client
Currently we only support python grpc server.

Install the requirements as in [grpc-python](../python/grpc/Readme.md)


## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge burkliu (刘柏基, liubaiji@xverse.cn) for contributing the grpc service.
