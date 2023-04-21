## Build on Linux X64

git clone  https://github.com/alibaba-damo-academy/FunASR.git

cd FunASR/funasr/runtime/rapidpunc

mkdir build

cd build

\# dowload onnxruntime:

wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.tgz



tar zxf  onnxruntime-linux-x64-1.14.1.tgz

 cmake -DONNXRUNTIME_DIR=/data/linux/FunASR/funasr/runtime/rapidpunc/build/onnxruntime-linux-x64-1.14.1 ..

make -j16



Then get two files: 

librapidpunc.so   

rapidpunc_tester





## run

rapidpunc_tester /path/to/model/dir  /path/to/wave/file



in a model_dir, it should include: punc.yaml  model.onnx

