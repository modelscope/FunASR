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
git clone https://github.com/alibaba-damo-academy/FunASR.git && cd FunASR/runtime/http
mkdir build && cd build
cmake  -DCMAKE_BUILD_TYPE=release .. -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.14.0 -DFFMPEG_DIR=/path/to/ffmpeg-master-latest-linux64-gpl-shared
make -j 4
```

### 测试

```shell
curl -F \"file=@example.wav\" 127.0.0.1:80
```

### 运行

```shell
./funasr-http-server  \
  --lm-dir '' \
  --itn-dir '' \
  --download-model-dir ${download_model_dir} \
  --model-dir ${model_dir} \
  --vad-dir ${vad_dir} \
  --punc-dir ${punc_dir} \
  --decoder-thread-num ${decoder_thread_num} \
  --io-thread-num  ${io_thread_num} \
  --port ${port} \
```



