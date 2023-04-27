# Service with grpc-cpp

## For the Server

### Build [onnxruntime](./onnxruntime_cpp.md) as it's document

### Compile and install grpc v1.52.0 in case of grpc bugs
```
export GRPC_INSTALL_DIR=/data/soft/grpc
export PKG_CONFIG_PATH=$GRPC_INSTALL_DIR/lib/pkgconfig

git clone -b v1.52.0 --depth=1  https://github.com/grpc/grpc.git
cd grpc
git submodule update --init --recursive

mkdir -p cmake/build
pushd cmake/build
cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$GRPC_INSTALL_DIR \
      ../..
make
make install
popd

echo "export GRPC_INSTALL_DIR=/data/soft/grpc" >> ~/.bashrc
echo "export PKG_CONFIG_PATH=\$GRPC_INSTALL_DIR/lib/pkgconfig" >> ~/.bashrc
echo "export PATH=\$GRPC_INSTALL_DIR/bin/:\$PKG_CONFIG_PATH:\$PATH" >> ~/.bashrc
source ~/.bashrc
```

### Compile and start grpc onnx paraformer server
```
# set -DONNXRUNTIME_DIR=/path/to/asrmodel/onnxruntime-linux-x64-1.14.0
./rebuild.sh
```

### Start grpc paraformer server
```
./cmake/build/paraformer-server     --port-id <string> [--punc-config
                                    <string>] [--punc-model <string>]
                                    --am-config <string> --am-cmvn <string>
                                    --am-model <string> [--vad-config
                                    <string>] [--vad-cmvn <string>]
                                    [--vad-model <string>] [--] [--version]
                                    [-h]
Where:
   --port-id <string>
     (required)  port id

   --am-config <string>
     (required)  am config path
   --am-cmvn <string>
     (required)  am cmvn path
   --am-model <string>
     (required)  am model path

   --punc-config <string>
     punc config path
   --punc-model <string>
     punc model path

   --vad-config <string>
     vad config path
   --vad-cmvn <string>
     vad cmvn path
   --vad-model <string>
     vad model path

   Required: --port-id <string> --am-config <string> --am-cmvn <string> --am-model <string> 
   If use vad, please add: [--vad-config <string>] [--vad-cmvn <string>] [--vad-model <string>]
   If use punc, please add: [--punc-config <string>] [--punc-model <string>] 
```

## For the client

### Install the requirements as in [grpc-python](./docs/grpc_python.md)

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

[//]: # (```)

[//]: # (# go to ../python/grpc to find this package)

[//]: # (import paraformer_pb2)

[//]: # ()
[//]: # ()
[//]: # (class RecognizeStub:)

[//]: # (    def __init__&#40;self, channel&#41;:)

[//]: # (        self.Recognize = channel.stream_stream&#40;)

[//]: # (                '/paraformer.ASR/Recognize',)

[//]: # (                request_serializer=paraformer_pb2.Request.SerializeToString,)

[//]: # (                response_deserializer=paraformer_pb2.Response.FromString,)

[//]: # (                &#41;)

[//]: # ()
[//]: # ()
[//]: # (async def send&#40;channel, data, speaking, isEnd&#41;:)

[//]: # (    stub = RecognizeStub&#40;channel&#41;)

[//]: # (    req = paraformer_pb2.Request&#40;&#41;)

[//]: # (    if data:)

[//]: # (        req.audio_data = data)

[//]: # (    req.user = 'zz')

[//]: # (    req.language = 'zh-CN')

[//]: # (    req.speaking = speaking)

[//]: # (    req.isEnd = isEnd)

[//]: # (    q = queue.SimpleQueue&#40;&#41;)

[//]: # (    q.put&#40;req&#41;)

[//]: # (    return stub.Recognize&#40;iter&#40;q.get, None&#41;&#41;)

[//]: # ()
[//]: # (# send the audio data once)

[//]: # (async def grpc_rec&#40;data, grpc_uri&#41;:)

[//]: # (    with grpc.insecure_channel&#40;grpc_uri&#41; as channel:)

[//]: # (        b = time.time&#40;&#41;)

[//]: # (        response = await send&#40;channel, data, False, False&#41;)

[//]: # (        resp = response.next&#40;&#41;)

[//]: # (        text = '')

[//]: # (        if 'decoding' == resp.action:)

[//]: # (            resp = response.next&#40;&#41;)

[//]: # (            if 'finish' == resp.action:)

[//]: # (                text = json.loads&#40;resp.sentence&#41;['text'])

[//]: # (        response = await send&#40;channel, None, False, True&#41;)

[//]: # (        return {)

[//]: # (                'text': text,)

[//]: # (                'time': time.time&#40;&#41; - b,)

[//]: # (                })

[//]: # ()
[//]: # (async def test&#40;&#41;:)

[//]: # (    # fc = FunAsrGrpcClient&#40;'127.0.0.1', 9900&#41;)

[//]: # (    # t = await fc.rec&#40;wav.tobytes&#40;&#41;&#41;)

[//]: # (    # print&#40;t&#41;)

[//]: # (    wav, _ = sf.read&#40;'z-10s.wav', dtype='int16'&#41;)

[//]: # (    uri = '127.0.0.1:9900')

[//]: # (    res = await grpc_rec&#40;wav.tobytes&#40;&#41;, uri&#41;)

[//]: # (    print&#40;res&#41;)

[//]: # ()
[//]: # ()
[//]: # (if __name__ == '__main__':)

[//]: # (    asyncio.run&#40;test&#40;&#41;&#41;)

[//]: # ()
[//]: # (```)


## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [DeepScience](https://www.deepscience.cn) for contributing the grpc service.
