## paraformer grpc onnx server in c++


#### Step 1. Build ../onnxruntime as it's document
```
#put onnx-lib & onnx-asr-model & vocab.txt into /path/to/asrmodel(eg: /data/asrmodel)
ls /data/asrmodel/
onnxruntime-linux-x64-1.14.0  speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

file /data/asrmodel/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/vocab.txt
UTF-8 Unicode text
```

#### Step 2. Compile and install grpc v1.52.0 in case of grpc bugs
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

#### Step 3. Compile and start grpc onnx paraformer server
```
# set -DONNXRUNTIME_DIR=/path/to/asrmodel/onnxruntime-linux-x64-1.14.0
./rebuild.sh
```

#### Step 4. Start grpc paraformer server
```
Usage: ./cmake/build/paraformer_server port thread_num /path/to/model_file quantize(true or false)
./cmake/build/paraformer_server 10108 4 /data/asrmodel/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch false
```



#### Step 5. Start grpc python paraformer client  on PC with MIC
```
cd ../python/grpc
python grpc_main_client_mic.py  --host $server_ip --port 10108
```
The `grpc_main_client_mic.py` follows the [original design] (https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/grpc#workflow-in-desgin) by sending audio_data with chunks. If you want to send audio_data in one request, here is an example:

```
# go to ../python/grpc to find this package
import paraformer_pb2


class RecognizeStub:
    def __init__(self, channel):
        self.Recognize = channel.stream_stream(
                '/paraformer.ASR/Recognize',
                request_serializer=paraformer_pb2.Request.SerializeToString,
                response_deserializer=paraformer_pb2.Response.FromString,
                )


async def send(channel, data, speaking, isEnd):
    stub = RecognizeStub(channel)
    req = paraformer_pb2.Request()
    if data:
        req.audio_data = data
    req.user = 'zz'
    req.language = 'zh-CN'
    req.speaking = speaking
    req.isEnd = isEnd
    q = queue.SimpleQueue()
    q.put(req)
    return stub.Recognize(iter(q.get, None))

# send the audio data once
async def grpc_rec(data, grpc_uri):
    with grpc.insecure_channel(grpc_uri) as channel:
        b = time.time()
        response = await send(channel, data, False, False)
        resp = response.next()
        text = ''
        if 'decoding' == resp.action:
            resp = response.next()
            if 'finish' == resp.action:
                text = json.loads(resp.sentence)['text']
        response = await send(channel, None, False, True)
        return {
                'text': text,
                'time': time.time() - b,
                }

async def test():
    # fc = FunAsrGrpcClient('127.0.0.1', 9900)
    # t = await fc.rec(wav.tobytes())
    # print(t)
    wav, _ = sf.read('z-10s.wav', dtype='int16')
    uri = '127.0.0.1:9900'
    res = await grpc_rec(wav.tobytes(), uri)
    print(res)


if __name__ == '__main__':
    asyncio.run(test())

```
