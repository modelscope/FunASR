# GRPC python Client for 2pass decoding
The client can send streaming or full audio data to server as you wish, and get transcribed text once the server respond (depends on mode)

In the demo client, audio_chunk_duration is set to 1000ms, and send_interval is set to 100ms

### 1. Install the requirements
```shell
git clone https://github.com/alibaba/FunASR.git && cd FunASR/funasr/runtime/python/grpc
pip install -r requirements.txt
```

### 2. Generate protobuf file
```shell
# paraformer_pb2.py and paraformer_pb2_grpc.py are already generated, 
# regenerate it only when you make changes to ./proto/paraformer.proto file.
python -m grpc_tools.protoc --proto_path=./proto -I ./proto --python_out=. --grpc_python_out=./ ./proto/paraformer.proto
```

### 3. Start grpc client
```
# Start client.
python grpc_main_client.py --host 127.0.0.1 --port 10100 --wav_path /path/to/your_test_wav.wav
```

## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge burkliu (刘柏基, liubaiji@xverse.cn) for contributing the grpc service.
