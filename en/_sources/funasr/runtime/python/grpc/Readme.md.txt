# Service with grpc-python
We can send streaming audio data to server in real-time with grpc client every 10 ms e.g., and get transcribed text when stop speaking.
The audio data is in streaming, the asr inference process is in offline.

## For the Server

### Prepare server environment
Install the modelscope and funasr

```shell
pip install -U modelscope funasr
# For the users in China, you could install with the command:
# pip install -U modelscope funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
git clone https://github.com/alibaba/FunASR.git && cd FunASR
```

Install the requirements

```shell
cd funasr/runtime/python/grpc
pip install -r requirements_server.txt
```


### Generate protobuf file
Run on server, the two generated pb files are both used for server and client

```shell
# paraformer_pb2.py and paraformer_pb2_grpc.py are already generated, 
# regenerate it only when you make changes to ./proto/paraformer.proto file.
python -m grpc_tools.protoc  --proto_path=./proto -I ./proto    --python_out=. --grpc_python_out=./ ./proto/paraformer.proto
```

### Start grpc server

```
# Start server.
python grpc_main_server.py --port 10095 --backend pipeline
```


## For the client

### Install the requirements

```shell
git clone https://github.com/alibaba/FunASR.git && cd FunASR
cd funasr/runtime/python/grpc
pip install -r requirements_client.txt
```

### Generate protobuf file
Run on server, the two generated pb files are both used for server and client

```shell
# paraformer_pb2.py and paraformer_pb2_grpc.py are already generated, 
# regenerate it only when you make changes to ./proto/paraformer.proto file.
python -m grpc_tools.protoc  --proto_path=./proto -I ./proto    --python_out=. --grpc_python_out=./ ./proto/paraformer.proto
```

### Start grpc client
```
# Start client.
python grpc_main_client_mic.py --host 127.0.0.1 --port 10095
```


## Workflow in desgin

<div align="left"><img src="proto/workflow.png" width="400"/>

## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).