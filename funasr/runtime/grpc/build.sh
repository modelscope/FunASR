#!/bin/bash

mode=debug #[debug|release]
onnxruntime_dir=`pwd`/../onnxruntime/onnxruntime-linux-x64-1.14.0
ffmpeg_dir=`pwd`/../onnxruntime/ffmpeg-N-111383-g20b8688092-linux64-gpl-shared


rm build -rf
mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=$mode ../ -DONNXRUNTIME_DIR=$onnxruntime_dir -DFFMPEG_DIR=$ffmpeg_dir
cmake --build . -j 4

echo "Build server successfully!"
