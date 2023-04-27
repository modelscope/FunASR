# Service with websocket-python
We can send streaming audio data to server in real-time with grpc client every 300 ms e.g., and get transcribed text when stop speaking.
The audio data is in streaming, the asr inference process is in offline.


## For the Server

### Install the modelscope and funasr

```shell
pip install -U modelscope funasr
# For the users in China, you could install with the command:
# pip install -U modelscope funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
git clone https://github.com/alibaba/FunASR.git && cd FunASR
```

### Install the requirements for server

```shell
cd funasr/runtime/python/websocket
pip install -r requirements_server.txt
```

### Start server
#### ASR offline server

[//]: # (```shell)

[//]: # (python ws_server_online.py --host "0.0.0.0" --port 10095 --asr_model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")

[//]: # (```)
#### ASR streaming server
```shell
python ws_server_online.py --host "0.0.0.0" --port 10095
```
#### 

#### ASR offline/online 2pass server

[//]: # (```shell)

[//]: # (python ws_server_online.py --host "0.0.0.0" --port 10095 --asr_model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")

[//]: # (```)

## For the client

Install the requirements for client
```shell
git clone https://github.com/alibaba/FunASR.git && cd FunASR
cd funasr/runtime/python/websocket
pip install -r requirements_client.txt
```

Start client

```shell
# --chunk_size, "5,10,5"=600ms, "8,8,4"=480ms
python ws_client.py --host "127.0.0.1" --port 10096 --chunk_size "5,10,5"
```

## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [cgisky1980](https://github.com/cgisky1980/FunASR) for contributing the websocket service.
