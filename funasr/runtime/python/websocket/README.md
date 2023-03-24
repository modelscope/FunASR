# Using funasr with websocket
We can send streaming audio data to server in real-time with grpc client every 300 ms e.g., and get transcribed text when stop speaking.
The audio data is in streaming, the asr inference process is in offline.


## For the Server

Install the modelscope and funasr

```shell
pip install "modelscope[audio_asr]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip install --editable ./
```

Install the requirements for server

```shell
cd funasr/runtime/python/websocket
pip install -r requirements_server.txt
```

Start server

```shell
python ASR_server.py --host "0.0.0.0" --port 10095 --asr_model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
```

## For the client

Install the requirements for client
```shell
git clone https://github.com/alibaba/FunASR.git && cd FunASR
cd funasr/runtime/python/websocket
pip install -r requirements_client.txt
```

Start client

```shell
python ASR_client.py --host "127.0.0.1" --port 10095 --chunk_size 300
```

## Acknowledge
1. We acknowledge [cgisky1980](https://github.com/cgisky1980/FunASR) for contributing the websocket service.
