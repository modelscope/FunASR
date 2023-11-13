# Advanced Development Guide (File transcription service) ([click](../docs/SDK_advanced_guide_offline.md))
# Real-time Speech Transcription Service Development Guide ([click](../docs/SDK_advanced_guide_online.md))


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
git clone https://github.com/alibaba-damo-academy/FunASR.git && cd FunASR/runtime/websocket
mkdir build && cd build
cmake  -DCMAKE_BUILD_TYPE=release .. -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.14.0 -DFFMPEG_DIR=/path/to/ffmpeg-master-latest-linux64-gpl-shared
make -j 4
```


## Building for Windows
### Download onnxruntime
https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/onnxruntime-win-x64-1.16.1.zip

Download to d:\ffmpeg-master-latest-win64-gpl-shared

### Download ffmpeg
https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/ffmpeg-master-latest-win64-gpl-shared.zip

Download to d:\onnxruntime-win-x64-1.16.1

### Download openssl
https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/openssl-1.1.1w.tar.gz 

Download to d:/src/openssl-1.1.1w

Open x64 Native Tools Command Prompt and execute the following compilation steps
```
d:
cd d:/src/openssl-1.1.1w
perl Configure VC-WIN64A --prefix=d:/openssl-1.1.1w
nmake
namke install

```

### Build runtime
```
git clone https://github.com/alibaba-damo-academy/FunASR.git 
cd FunASR/runtime/websocket
mkdir build
cd build
cmake ../ -D OPENSSL_ROOT_DIR=d:/openssl-1.1.1w -D FFMPEG_DIR=d:/ffmpeg-master-latest-win64-gpl-shared -D ONNXRUNTIME_DIR=d:/onnxruntime-win-x64-1.16.1
```
Open FunASRWebscoket.sln in Visual Studio and complete the compilation.

