# FunASR离线文件转写服务开发指南([点击此处](../docs/SDK_advanced_guide_offline_zh.md))

# FunASR实时语音听写服务开发指南([点击此处](../docs/SDK_advanced_guide_online_zh.md))

# 如果您想自己编译文件，可以参考下述步骤
## Linux/Unix 平台编译
### 下载 onnxruntime
```shell
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/onnxruntime-linux-x64-1.14.0.tgz
tar -zxvf onnxruntime-linux-x64-1.14.0.tgz
```

### 下载 ffmpeg
```shell
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/ffmpeg-master-latest-linux64-gpl-shared.tar.xz
tar -xvf ffmpeg-master-latest-linux64-gpl-shared.tar.xz
```

### 安装依赖
```shell
# openblas
sudo apt-get install libopenblas-dev #ubuntu
# sudo yum -y install openblas-devel #centos

# openssl
apt-get install libssl-dev #ubuntu 
# yum install openssl-devel #centos
```

### 编译 runtime

```shell
git clone https://github.com/alibaba-damo-academy/FunASR.git && cd FunASR/runtime/websocket
mkdir build && cd build
cmake  -DCMAKE_BUILD_TYPE=release .. -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.14.0 -DFFMPEG_DIR=/path/to/ffmpeg-master-latest-linux64-gpl-shared
make -j 4
```


## Windows 平台编译
### 下载 onnxruntime
https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/onnxruntime-win-x64-1.16.1.zip

下载并解压到 d:/onnxruntime-win-x64-1.16.1

### 下载 ffmpeg
https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/ffmpeg-master-latest-win64-gpl-shared.zip

下载并解压到 d:/ffmpeg-master-latest-win64-gpl-shared

### 编译 openssl
https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/openssl-1.1.1w.zip

下载解压到 d:/openssl-1.1.1w

### 编译 runtime
```
git clone https://github.com/alibaba-damo-academy/FunASR.git 
cd FunASR/runtime/websocket
mkdir build
cd build
cmake ../ -D OPENSSL_ROOT_DIR=d:/openssl-1.1.1w -D FFMPEG_DIR=d:/ffmpeg-master-latest-win64-gpl-shared -D ONNXRUNTIME_DIR=d:/onnxruntime-win-x64-1.16.1
```
Visual Studio 打开 FunASR/runtime/websocket/build/FunASRWebscoket.sln 完成编译；
编译后的可执行文件位于：FunASR/runtime/websocket/build/bin/Debug;
从 onnxruntime-win-x64-1.16.1/lib, ffmpeg-master-latest-win64-gpl-shared/bin, openssl-1.1.1w/bin copy相关的DLL库至: FunASR/runtime/websocket/build/bin/Debug

