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

# need to install boost lib
apt install libboost-dev libboost-system-dev #ubuntu
# openblas
sudo apt-get install libopenblas-dev #ubuntu
# sudo yum -y install openblas-devel #centos

# openssl
apt-get install libssl-dev #ubuntu 
# yum install openssl-devel #centos
```

### Build runtime
```shell
git clone https://github.com/alibaba-damo-academy/FunASR.git && cd FunASR/runtime/http
mkdir build && cd build
cmake  -DCMAKE_BUILD_TYPE=release .. -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.14.0 -DFFMPEG_DIR=/path/to/ffmpeg-master-latest-linux64-gpl-shared
make -j 4
```

### test

```shell
curl -F \"file=@example.wav\" 127.0.0.1:80
```

### run

```shell
./funasr-http-server  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx --punc-dir damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx --itn-dir ''  --lm-dir ''  --port 10001
```

