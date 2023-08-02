#!/bin/bash

rm build -rf
mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=release ../ \
  -DONNXRUNTIME_DIR=/cfs/user/burkliu/work2023/FunASR/funasr/runtime/onnxruntime/onnxruntime-linux-x64-1.14.0 \
  -DFFMPEG_DIR=/cfs/user/burkliu/work2023/FunASR/funasr/runtime/onnxruntime/ffmpeg-N-111383-g20b8688092-linux64-gpl-shared
cmake --build . -j 4

echo "Build build/paraformer_server successfully!"
