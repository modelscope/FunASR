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
```shell
git clone https://github.com/alibaba-damo-academy/FunASR.git && cd funasr/runtime/websocket
mkdir build && cd build
cmake  -DCMAKE_BUILD_TYPE=release .. -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.14.0
make
```
## Run the websocket server

```shell
cd bin
websocketmain  [--model_thread_num <int>] [--decoder_thread_num
                        <int>] [--io_thread_num <int>] [--port <int>]
                        [--listen_ip <string>] [--wav-scp <string>]
                        [--wav-path <string>] [--punc-config <string>]
                        [--punc-model <string>] --am-config <string>
                        --am-cmvn <string> --am-model <string>
                        [--vad-config <string>] [--vad-cmvn <string>]
                        [--vad-model <string>] [--] [--version] [-h]
Where:
   --wav-scp <string>
     wave scp path
   --wav-path <string>
     wave file path

   --punc-config <string>
     punc config path
   --punc-model <string>
     punc model path

   --am-config <string>
     (required)  am config path
   --am-cmvn <string>
     (required)  am cmvn path
   --am-model <string>
     (required)  am model path

   --vad-config <string>
     vad config path
   --vad-cmvn <string>
     vad cmvn path
   --vad-model <string>
     vad model path
   --decoder_thread_num <int>
     number of threads for decoder
   --io_thread_num <int>
     number of threads for network io
  
   Required: --am-config <string> --am-cmvn <string> --am-model <string> 
   If use vad, please add: [--vad-config <string>] [--vad-cmvn <string>] [--vad-model <string>]
   If use punc, please add: [--punc-config <string>] [--punc-model <string>] 
example:
   websocketmain --am-config /FunASR/funasr/runtime/onnxruntime/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/config.yaml --am-model /FunASR/funasr/runtime/onnxruntime/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.onnx --am-cmvn /FunASR/funasr/runtime/onnxruntime/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/am.mvn
```

## Run websocket client test

```shell
Usage: websocketclient server_ip port wav_path threads_num

example:

websocketclient 127.0.0.1 8889 funasr/runtime/websocket/test.pcm.wav 64

result json, example like:
{"text":"一二三四五六七八九十一二三四五六七八九十"}
```


## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [zhaoming](https://github.com/zhaomingwork/FunASR/tree/add-offline-websocket-srv/funasr/runtime/websocket) for contributing the websocket(cpp-api).


