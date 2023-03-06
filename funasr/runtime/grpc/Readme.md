## paraformer grpc onnx server



#### Step 1. Compile and install grpc v1.52.0 in case of grpc bugs
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



#### Step 2. Compile grpc onnx paraformer server
```
#depends on ../onnxruntime
#file vocab.txt : UTF-8 Unicode text

./rebuild.sh

```



#### Step 3. Start grpc python paraformer client  on PC with MIC
```
cd ../python/grpc
python grpc_main_client_mic.py  --host 127.0.0.1 --port 10108
```


