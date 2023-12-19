# Please ref to [websocket service](https://github.com/alibaba-damo-academy/FunASR/tree/main/runtime/websocket)

# If you want to compile the file yourself, you can follow the steps below.
## Building for Linux/Unix
### Download onnxruntime
```shell
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/onnxruntime-linux-x64-1.14.0.tgz
tar -zxvf onnxruntime-linux-x64-1.14.0.tgz
```

### Download ffmpeg
```shell
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/ffmpeg-master-latest-linux64-gpl-shared.tar.xz
tar -xvf ffmpeg-master-latest-linux64-gpl-shared.tar.xz
```

### Install deps
```shell
# openblas
sudo apt-get install libopenblas-dev #ubuntu
# sudo yum -y install openblas-devel #centos

# openssl
apt-get install libssl-dev #ubuntu 
# yum install openssl-devel #centos
```

### Build runtime
```shell
git clone https://github.com/alibaba-damo-academy/FunASR.git && cd FunASR/runtime/onnxruntime
mkdir build && cd build
cmake  -DCMAKE_BUILD_TYPE=release .. -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.14.0 -DFFMPEG_DIR=/path/to/ffmpeg-master-latest-linux64-gpl-shared
make -j 4
```


## Building for Windows
### Download onnxruntime
https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/onnxruntime-win-x64-1.16.1.zip

Download and unzip to d:/onnxruntime-win-x64-1.16.1

### Download ffmpeg
https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/ffmpeg-master-latest-win64-gpl-shared.zip

Download and unzip to d:/ffmpeg-master-latest-win64-gpl-shared

### Build runtime
```
git clone https://github.com/alibaba-damo-academy/FunASR.git
cd FunASR/runtime/onnxruntime
mkdir build
cd build
cmake ../ -D FFMPEG_DIR=d:/ffmpeg-master-latest-win64-gpl-shared -D ONNXRUNTIME_DIR=d:/onnxruntime-win-x64-1.16.1
```

Visual Studio open FunASR/runtime/onnxruntime/build/FunASROnnx.sln start build. 
After compilation, the executable file is located here: FunASR/runtime/onnxruntime/build/bin/Debug.
Copy the required DLL libraries from (onnxruntime-win-x64-1.16.1/lib, ffmpeg-master-latest-win64-gpl-shared/bin) to this location: FunASR/runtime/onnxruntime/build/bin/Debug

