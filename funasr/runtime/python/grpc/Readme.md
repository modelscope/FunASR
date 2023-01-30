# Using paraformer with grpc
We can send streaming audio data to server in real-time with grpc client every 10 ms e.g., and get transcribed text when stop speaking.
The audio data is in streaming, the asr inference process is in offline.


## Steps

Step 1) Prepare server environment (on server).
```
# Optional, modelscope cuda docker is preferred.
CID=`docker run --network host -d -it --gpus '"device=0"' registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.3.0-py37-torch1.11.0-tf1.15.5-1.2.0`
echo $CID
docker exec -it $CID /bin/bash
cd /opt/conda/lib/python3.7/site-packages/funasr/runtime/python/grpc
```

Step 2) Generate protobuf file (for server and client).
```
# Optional, paraformer_pb2.py and paraformer_pb2_grpc.py are already generated.
python -m grpc_tools.protoc  --proto_path=./proto -I ./proto    --python_out=. --grpc_python_out=./ ./proto/paraformer.proto
```

Step 3) Start grpc server (on server).
```
python grpc_main_server.py --port 10095
```

Step 4) Start grpc client (on client with microphone).
```
# Install dependency. Optional.
python -m pip install pyaudio webrtcvad
```
```
# Start client.
python grpc_main_client_mic.py --host 127.0.0.1 --port 10095
```


## Workflow in desgin
![avatar](proto/workflow.png)


## Reference
We borrow or refer to some code from:

1)https://github.com/wenet-e2e/wenet/tree/main/runtime/core/grpc

2)https://github.com/Open-Speech-EkStep/inference_service/blob/main/realtime_inference_service.py