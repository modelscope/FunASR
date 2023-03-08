## paraformer grpc onnx server


#### Build ../onnxruntime as it's document
```
#put onnx-lib & onnx-asr-model & vocab.txt into /data/asrmodel
ls /data/asrmodel/
onnxruntime-linux-x64-1.14.0  speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

file /data/asrmodel/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/vocab.txt
UTF-8 Unicode text
```

#### Compile and install grpc v1.52.0 in case of grpc bugs
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
popd

echo "export GRPC_INSTALL_DIR=/data/soft/grpc" >> ~/.bashrc
echo "export PKG_CONFIG_PATH=\$GRPC_INSTALL_DIR/lib/pkgconfig" >> ~/.bashrc
echo "export PATH=\$GRPC_INSTALL_DIR/bin/:\$PKG_CONFIG_PATH:\$PATH" >> ~/.bashrc
source ~/.bashrc
```

#### Compile grpc onnx paraformer server
```
./rebuild.sh
```



#### Start grpc python paraformer client  on PC with MIC
```
cd ../python/grpc
python grpc_main_client_mic.py  --host $server_ip --port 10108
```
