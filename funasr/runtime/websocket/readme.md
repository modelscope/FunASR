# Service with websocket-cpp

## Export the model
### Install [modelscope and funasr](https://github.com/alibaba-damo-academy/FunASR#installation)

```shell
# pip3 install torch torchaudio
pip install -U modelscope funasr
# For the users in China, you could install with the command:
# pip install -U modelscope funasr -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html -i https://mirror.sjtu.edu.cn/pypi/web/simple
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
required openssl lib

```shell
#install openssl lib first
apt-get install libssl-dev

git clone https://github.com/alibaba-damo-academy/FunASR.git && cd funasr/runtime/websocket
mkdir build && cd build
cmake  -DCMAKE_BUILD_TYPE=release .. -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.14.0
make
```
## Run the websocket server

```shell
cd bin
   ./websocketmain  [--model_thread_num <int>] [--decoder_thread_num <int>]
                    [--io_thread_num <int>] [--port <int>] [--listen_ip
                    <string>] [--punc-quant <string>] [--punc-dir <string>]
                    [--vad-quant <string>] [--vad-dir <string>] [--quantize
                    <string>] --model-dir <string> [--keyfile <string>]
                    [--certfile <string>] [--] [--version] [-h]
Where:
   --model-dir <string>
     (required)  the asr model path, which contains model.onnx, config.yaml, am.mvn
   --quantize <string>
     false (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir

   --vad-dir <string>
     the vad model path, which contains model.onnx, vad.yaml, vad.mvn
   --vad-quant <string>
     false (Default), load the model of model.onnx in vad_dir. If set true, load the model of model_quant.onnx in vad_dir

   --punc-dir <string>
     the punc model path, which contains model.onnx, punc.yaml
   --punc-quant <string>
     false (Default), load the model of model.onnx in punc_dir. If set true, load the model of model_quant.onnx in punc_dir

   --decoder_thread_num <int>
     number of threads for decoder, default:8
   --io_thread_num <int>
     number of threads for network io, default:8
   --port <int>
     listen port, default:8889
   --certfile <string>
     path of certficate for WSS connection. if it is empty, it will be in WS mode.
   --keyfile <string>
     path of keyfile for WSS connection
  
   Required:  --model-dir <string>
   If use vad, please add: --vad-dir <string>
   If use punc, please add: --punc-dir <string>
example:
   websocketmain --model-dir /FunASR/funasr/runtime/onnxruntime/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```

## Run websocket client test

```shell
Usage: ./websocketclient server_ip port wav_path threads_num is_ssl

is_ssl is 1 means use wss connection, or use ws connection

example:

websocketclient 127.0.0.1 8889 funasr/runtime/websocket/test.pcm.wav 64 0

result json, example like:
{"mode":"offline","text":"欢迎大家来体验达摩院推出的语音识别模型","wav_name":"wav2"}
```


## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [zhaoming](https://github.com/zhaomingwork/FunASR/tree/add-offline-websocket-srv/funasr/runtime/websocket) for contributing the websocket(cpp-api).


