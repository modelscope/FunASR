# Using paraformer with grpc
We can send streaming audio data to server in real-time with grpc client every 10 ms e.g., and get transcribed text when stop speaking.
The audio data is in streaming, the asr inference process is in offline.


## Steps

Step 1-1) Prepare server modelscope pipeline environment (on server).  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Install modelscope and funasr with pip or with cuda-docker image.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Option 1: Install modelscope and funasr with [pip](https://github.com/alibaba-damo-academy/FunASR#installation)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Option 2: or install with cuda-docker image as: 

```
CID=`docker run --network host -d -it --gpus '"device=0"' registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.3.0-py37-torch1.11.0-tf1.15.5-1.2.0`
echo $CID
docker exec -it $CID /bin/bash
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Get funasr source code and get into grpc directory.
```
git clone https://github.com/alibaba-damo-academy/FunASR
cd FunASR/funasr/runtime/python/grpc/
```

Step 1-2) Optional, Prepare server onnxruntime environment (on server). 

Install [`onnx_paraformer`](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/onnxruntime).

- Build the onnx_paraformer `whl`
```
git clone https://github.com/alibaba/FunASR.git && cd FunASR
cd funasr/runtime/python/onnxruntime/rapid_paraformer
python setup.py build
python setup.py install
```

[//]: # ()
[//]: # (- Install the build `whl`)

[//]: # (```)

[//]: # (pip install dist/rapid_paraformer-0.0.1-py3-none-any.whl)

[//]: # (```)

Export the model, more details ref to [export docs](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/onnxruntime).
```shell
python -m funasr.export.export_model --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type onnx --quantize True
```

Step 2) Optional, generate protobuf file (run on server, the two generated pb files are both used for server and client).
```
# Optional, Install dependency.
python -m pip install grpcio grpcio-tools
```

```
# paraformer_pb2.py and paraformer_pb2_grpc.py are already generated, 
# regenerate it only when you make changes to ./proto/paraformer.proto file.
python -m grpc_tools.protoc  --proto_path=./proto -I ./proto    --python_out=. --grpc_python_out=./ ./proto/paraformer.proto
```

Step 3) Start grpc server (on server).
```
# Optional, Install dependency.
python -m pip install grpcio grpcio-tools
```
```
# Start server.
python grpc_main_server.py --port 10095 --backend pipeline
```

If you want run server with onnxruntime, please set `backend` and `onnx_dir` paramater.
```
# Start server.
python grpc_main_server.py --port 10095 --backend onnxruntime --onnx_dir /models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```


Step 4) Start grpc client (on client with microphone).
```
# Optional, Install dependency.
python -m pip install pyaudio webrtcvad grpcio grpcio-tools
```
```
# Start client.
python grpc_main_client_mic.py --host 127.0.0.1 --port 10095
```


## Workflow in desgin
![avatar](proto/workflow.png)


## Reference
We borrow from or refer to some code as:

1)https://github.com/wenet-e2e/wenet/tree/main/runtime/core/grpc

2)https://github.com/Open-Speech-EkStep/inference_service/blob/main/realtime_inference_service.py