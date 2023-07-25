(简体中文|[English](./readme.md))

# 采用websocket协议的c++部署方案

## 快速上手
### 启动docker镜像

通过下述命令拉取并启动FunASR runtime-SDK的docker镜像：

```shell
sudo docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.1.0

sudo docker run -p 10095:10095 -it --privileged=true -v /root:/workspace/models registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.1.0
```
如果您没有安装docker，可参考[Docker安装](https://alibaba-damo-academy.github.io/FunASR/en/installation/docker.html)

### 服务端启动

docker启动之后，启动 funasr-wss-server服务程序：
```shell
cd FunASR/funasr/runtime
./run_server.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx  \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx
```
服务端详细参数介绍可参考[服务端参数介绍](#命令参数介绍)

### 客户端测试与使用

下载客户端测试工具目录samples
```shell
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz
```
我们以Python语言客户端为例，进行说明，支持多种音频格式输入（.wav, .pcm, .mp3等），也支持视频输入(.mp4等)，以及多文件列表wav.scp输入，其他版本客户端请参考文档（[点击此处](#客户端用法详解)），定制服务部署请参考[如何定制服务部署](#如何定制服务部署)
```shell
python3 wss_client_asr.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "../audio/asr_example.wav"
```

------------------

## 操作步骤详解

### 依赖库下载

Docker中已经预安装了依赖三方库，如果不用docker，请手动下载并安装（[三方库下载与安装](requirements_install.md)）


### 编译

```shell
git clone https://github.com/alibaba-damo-academy/FunASR.git && cd FunASR/funasr/runtime/websocket
mkdir build && cd build
cmake  -DCMAKE_BUILD_TYPE=release .. -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.14.0 -DFFMPEG_DIR=/path/to/ffmpeg-N-111383-g20b8688092-linux64-gpl-shared
make
```

### 启动服务部署

#### 命令参数介绍：
```text
--download-model-dir 模型下载地址，通过设置model ID从Modelscope下载模型。如果从本地模型启动，可以不设置。
--model-dir  modelscope 中 ASR model ID，或者本地模型绝对路径
--quantize  True为量化ASR模型，False为非量化ASR模型，默认是True
--vad-dir  modelscope 中 VAD model ID，或者本地模型绝对路径
--vad-quant   True为量化VAD模型，False为非量化VAD模型，默认是True
--punc-dir  modelscope 中 标点 model ID，或者本地模型绝对路径
--punc-quant   True为量化PUNC模型，False为非量化PUNC模型，默认是True
--port  服务端监听的端口号，默认为 10095
--decoder-thread-num  服务端启动的推理线程数，默认为 8
--io-thread-num  服务端启动的IO线程数，默认为 1
--certfile  ssl的证书文件，默认为：../../../ssl_key/server.crt
--keyfile   ssl的密钥文件，默认为：../../../ssl_key/server.key
```

#### 从modelscope中模型启动示例
```shell
./funasr-wss-server  \
  --download-model-dir /workspace/models \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx
```

注意：上面示例中，`model-dir`，`vad-dir`，`punc-dir`为模型在modelscope中模型名字，直接从modelscope下载模型并且导出量化后的onnx。如果需要从本地启动，需要改成本地绝对路径。

#### 从本地模型启动示例

##### 导出模型

```shell
python -m funasr.export.export_model \
--export-dir ./export \
--type onnx \
--quantize True \
--model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
--model-name damo/speech_fsmn_vad_zh-cn-16k-common-pytorch \
--model-name damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
```
导出过程详细介绍（[点击此处](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export)）

##### 启动服务
```shell
./funasr-wss-server  \
  --download-model-dir /workspace/models \
  --model-dir ./exportdamo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx \
  --vad-dir ./exportdamo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --punc-dir ./export/damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx
```



### 客户端用法详解

下载客户端测试工具目录samples
```shell
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz
```

在服务器上完成FunASR服务部署以后，可以通过如下的步骤来测试和使用离线文件转写服务。
目前分别支持以下几种编程语言客户端

- [Python](#python-client)
- [CPP](#cpp-client)
- [html网页版本](#Html网页版)
- [Java](#Java-client)

#### python-client
若想直接运行client进行测试，可参考如下简易说明，以python版本为例：

```shell
python3 wss_client_asr.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "../audio/asr_example.wav" --output_dir "./results"
```

命令参数说明：
```text
--host 为FunASR runtime-SDK服务部署机器ip，默认为本机ip（127.0.0.1），如果client与服务不在同一台服务器，需要改为部署机器ip
--port 10095 部署端口号
--mode offline表示离线文件转写
--audio_in 需要进行转写的音频文件，支持文件路径，文件列表wav.scp
--output_dir 识别结果保存路径
```

### cpp-client
进入samples/cpp目录后，可以用cpp进行测试，指令如下：
```shell
./funasr-wss-client --server-ip 127.0.0.1 --port 10095 --wav-path ../audio/asr_example.wav
```

命令参数说明：

```text
--server-ip 为FunASR runtime-SDK服务部署机器ip，默认为本机ip（127.0.0.1），如果client与服务不在同一台服务器，需要改为部署机器ip
--port 10095 部署端口号
--wav-path 需要进行转写的音频文件，支持文件路径
```

### Html网页版

在浏览器中打开 html/static/index.html，即可出现如下页面，支持麦克风输入与文件上传，直接进行体验

<img src="images/html.png"  width="900"/>

### Java-client

```shell
FunasrWsClient --host localhost --port 10095 --audio_in ./asr_example.wav --mode offline
```
详细可以参考文档（[点击此处](../java/readme.md)）



## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [zhaoming](https://github.com/zhaomingwork/FunASR/tree/add-offline-websocket-srv/funasr/runtime/websocket) for contributing the websocket(cpp-api).


