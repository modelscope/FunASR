#!/bin/bash

rm cmake -rf
mkdir -p cmake/build

cd cmake/build

cmake  -DCMAKE_BUILD_TYPE=release ../.. -DONNXRUNTIME_DIR=/data/asrmodel/onnxruntime-linux-x64-1.14.0
make


echo "Build cmake/build/paraformer_server successfully!"
