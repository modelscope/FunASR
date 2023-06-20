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
#install openssl lib for ubuntu 
apt-get install libssl-dev
#install openssl lib for centos
yum install openssl-devel


git clone https://github.com/alibaba-damo-academy/FunASR.git && cd funasr/runtime/websocket
mkdir build && cd build
cmake  -DCMAKE_BUILD_TYPE=release .. -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.14.0
make
```
## Run the websocket server

```shell
cd bin
./funasr-wss-server  [--model-thread-num <int>] [--decoder-thread-num <int>]
                    [--io-thread-num <int>] [--port <int>] [--listen_ip
                    <string>] [--punc-quant <string>] [--punc-dir <string>]
                    [--vad-quant <string>] [--vad-dir <string>] [--quantize
                    <string>] --model-dir <string> [--keyfile <string>]
                    [--certfile <string>] [--] [--version] [-h]
Where:
   --model-dir <string>
     default: /workspace/models/asr, the asr model path, which contains model.onnx, config.yaml, am.mvn
   --quantize <string>
     true (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir

   --vad-dir <string>
     default: /workspace/models/vad, the vad model path, which contains model.onnx, vad.yaml, vad.mvn
   --vad-quant <string>
     true (Default), load the model of model.onnx in vad_dir. If set true, load the model of model_quant.onnx in vad_dir

   --punc-dir <string>
     default: /workspace/models/punc, the punc model path, which contains model.onnx, punc.yaml
   --punc-quant <string>
     true (Default), load the model of model.onnx in punc_dir. If set true, load the model of model_quant.onnx in punc_dir

   --decoder-thread-num <int>
     number of threads for decoder, default:8
   --io-thread-num <int>
     number of threads for network io, default:8
   --port <int>
     listen port, default:10095
   --certfile <string>
     default: ../../../ssl_key/server.crt, path of certficate for WSS connection. if it is empty, it will be in WS mode.
   --keyfile <string>
     default: ../../../ssl_key/server.key, path of keyfile for WSS connection
  
example:
./funasr-wss-server --model-dir /FunASR/funasr/runtime/onnxruntime/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```

## Run websocket client test

```shell
./funasr-wss-client  --server-ip <string>
                    --port <string>
                    --wav-path <string>
                    [--thread-num <int>] 
                    [--is-ssl <int>]  [--]
                    [--version] [-h]

Where:
   --server-ip <string>
     (required)  server-ip

   --port <string>
     (required)  port

   --wav-path <string>
     (required)  the input could be: wav_path, e.g.: asr_example.wav;
     pcm_path, e.g.: asr_example.pcm; wav.scp, kaldi style wav list (wav_id \t wav_path)

   --thread-num <int>
     thread-num

   --is-ssl <int>
     is-ssl is 1 means use wss connection, or use ws connection

example:
./funasr-wss-client --server-ip 127.0.0.1 --port 10095 --wav-path test.wav --thread-num 1 --is-ssl 1

result json, example like:
{"mode":"offline","text":"欢迎大家来体验达摩院推出的语音识别模型","wav_name":"wav2"}
```


## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [zhaoming](https://github.com/zhaomingwork/FunASR/tree/add-offline-websocket-srv/funasr/runtime/websocket) for contributing the websocket(cpp-api).


