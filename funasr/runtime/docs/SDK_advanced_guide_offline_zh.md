# FunASR离线文件转写服务开发指南

FunASR提供可一键本地或者云端服务器部署的中文离线文件转写服务，内核为FunASR已开源runtime-SDK。FunASR-runtime结合了达摩院语音实验室在Modelscope社区开源的语音端点检测(VAD)、Paraformer-large语音识别(ASR)、标点检测(PUNC) 等相关能力，可以准确、高效的对音频进行高并发转写。

本文档为FunASR离线文件转写服务开发指南。如果您想快速体验离线文件转写服务，可参考[快速上手](#快速上手)。

## 快速上手
### 镜像启动

通过下述命令拉取并启动FunASR runtime-SDK的docker镜像：

```shell
sudo docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.1.0

sudo docker run -p 10095:10095 -it --privileged=true -v /root:/workspace/models registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.1.0
```
如果您没有安装docker，可参考[Docker安装](#Docker安装)

### 服务端启动

docker启动之后，启动 funasr-wss-server服务程序：
```shell
cd FunASR/funasr/runtime
nohup bash run_server.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx  \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx > log.out 2>&1 &

# 如果您想关闭ssl，增加参数：--certfile 0
# 如果您想使用时间戳或者热词模型进行部署，请设置--model-dir为对应模型：damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx（时间戳）或者 damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx（热词）
```
服务端详细参数介绍可参考[服务端参数介绍](#服务端参数介绍)
### 客户端测试与使用

下载客户端测试工具目录samples
```shell
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz
```
我们以Python语言客户端为例，进行说明，支持多种音频格式输入（.wav, .pcm, .mp3等），也支持视频输入(.mp4等)，以及多文件列表wav.scp输入，其他版本客户端请参考文档（[点击此处](#客户端用法详解)），定制服务部署请参考[如何定制服务部署](#如何定制服务部署)
```shell
python3 funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "../audio/asr_example.wav"
```

------------------
## Docker安装

下述步骤为手动安装docker环境的步骤：

### docker环境安装
```shell
# Ubuntu：
curl -fsSL https://test.docker.com -o test-docker.sh 
sudo sh test-docker.sh 
# Debian：
curl -fsSL https://get.docker.com -o get-docker.sh 
sudo sh get-docker.sh 
# CentOS：
curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun 
# MacOS：
brew install --cask --appdir=/Applications docker
```

安装详见：https://alibaba-damo-academy.github.io/FunASR/en/installation/docker.html

### docker启动

```shell
sudo systemctl start docker
```


## 客户端用法详解

在服务器上完成FunASR服务部署以后，可以通过如下的步骤来测试和使用离线文件转写服务。
目前分别支持以下几种编程语言客户端

- [Python](#python-client)
- [CPP](#cpp-client)
- [html网页版本](#Html网页版)
- [Java](#Java-client)

### python-client
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



## 服务端参数介绍：

funasr-wss-server支持从Modelscope下载模型，设置模型下载地址（--download-model-dir，默认为/workspace/models）及model ID（--model-dir、--vad-dir、--punc-dir）,示例如下：
```shell
cd /workspace/FunASR/funasr/runtime/websocket/build/bin
./funasr-wss-server  \
  --download-model-dir /workspace/models \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx \
  --decoder-thread-num 32 \
  --io-thread-num  8 \
  --port 10095 \
  --certfile  ../../../ssl_key/server.crt \
  --keyfile ../../../ssl_key/server.key
 ```
命令参数介绍：
```text
--download-model-dir 模型下载地址，通过设置model ID从Modelscope下载模型
--model-dir  modelscope model ID
--quantize  True为量化ASR模型，False为非量化ASR模型，默认是True
--vad-dir  modelscope model ID
--vad-quant   True为量化VAD模型，False为非量化VAD模型，默认是True
--punc-dir  modelscope model ID
--punc-quant   True为量化PUNC模型，False为非量化PUNC模型，默认是True
--port  服务端监听的端口号，默认为 10095
--decoder-thread-num  服务端启动的推理线程数，默认为 8
--io-thread-num  服务端启动的IO线程数，默认为 1
--certfile  ssl的证书文件，默认为：../../../ssl_key/server.crt
--keyfile   ssl的密钥文件，默认为：../../../ssl_key/server.key
```

funasr-wss-server同时也支持从本地路径加载模型（本地模型资源准备详见[模型资源准备](#模型资源准备)）示例如下：
```shell
cd /workspace/FunASR/funasr/runtime/websocket/build/bin
./funasr-wss-server  \
  --model-dir /workspace/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx \
  --vad-dir /workspace/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --punc-dir /workspace/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx \
  --decoder-thread-num 32 \
  --io-thread-num  8 \
  --port 10095 \
  --certfile  ../../../ssl_key/server.crt \
  --keyfile ../../../ssl_key/server.key
 ```
命令参数介绍：
```text
--model-dir  ASR模型路径，默认为：/workspace/models/asr
--quantize   True为量化ASR模型，False为非量化ASR模型，默认是True
--vad-dir  VAD模型路径，默认为：/workspace/models/vad
--vad-quant   True为量化VAD模型，False为非量化VAD模型，默认是True
--punc-dir  PUNC模型路径，默认为：/workspace/models/punc
--punc-quant   True为量化PUNC模型，False为非量化PUNC模型，默认是True
--port  服务端监听的端口号，默认为 10095
--decoder-thread-num  服务端启动的推理线程数，默认为 8
--io-thread-num  服务端启动的IO线程数，默认为 1
--certfile ssl的证书文件，默认为：../../../ssl_key/server.crt
--keyfile  ssl的密钥文件，默认为：../../../ssl_key/server.key
```

## 模型资源准备

如果您选择通过funasr-wss-server从Modelscope下载模型，可以跳过本步骤。

FunASR离线文件转写服务中的vad、asr和punc模型资源均来自Modelscope，模型地址详见下表：

| 模型 | Modelscope链接                                                                                                  |
|------|---------------------------------------------------------------------------------------------------------------|
| VAD  | https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx/summary |
| ASR  | https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx/summary                           |
| PUNC | https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx/summary               |

离线文件转写服务中部署的是量化后的ONNX模型，下面介绍下如何导出ONNX模型及其量化：您可以选择从Modelscope导出ONNX模型、从finetune后的资源导出模型：

### 从Modelscope导出ONNX模型

从Modelscope网站下载对应model name的模型，然后导出量化后的ONNX模型：

```shell
python -m funasr.export.export_model \
--export-dir ./export \
--type onnx \
--quantize True \
--model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
--model-name damo/speech_fsmn_vad_zh-cn-16k-common-pytorch \
--model-name damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
```

命令参数介绍：
```text
--model-name  Modelscope上的模型名称，例如damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
--export-dir  ONNX模型导出地址
--type 模型类型，目前支持 ONNX、torch
--quantize  int8模型量化
```
### 从finetune后的资源导出模型

假如您想部署finetune后的模型，可以参考如下步骤：

将您finetune后需要部署的模型（例如10epoch.pb），重命名为model.pb，并将原modelscope中模型model.pb替换掉，假如替换后的模型路径为/path/to/finetune/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch，通过下述命令把finetune后的模型转成onnx模型：

```shell
python -m funasr.export.export_model --model-name /path/to/finetune/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type onnx --quantize True
```



## 如何定制服务部署

FunASR-runtime的代码已开源，如果服务端和客户端不能很好的满足您的需求，您可以根据自己的需求进行进一步的开发：
### c++ 客户端：

https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/websocket

### python 客户端：

https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/websocket

### 自定义客户端：

如果您想定义自己的client，websocket通信协议为：

```text
# 首次通信
{"mode": "offline", "wav_name": wav_name, "is_speaking": True}
# 发送wav数据
bytes数据
# 发送结束标志
{"is_speaking": False}
```

### c++ 服务端：

#### VAD
```c++
// VAD模型的使用分为FsmnVadInit和FsmnVadInfer两个步骤：
FUNASR_HANDLE vad_hanlde=FsmnVadInit(model_path, thread_num);
// 其中：model_path 包含"model-dir"、"quantize"，thread_num为onnx线程数；
FUNASR_RESULT result=FsmnVadInfer(vad_hanlde, wav_file.c_str(), NULL, 16000);
// 其中：vad_hanlde为FunOfflineInit返回值，wav_file为音频路径，sampling_rate为采样率(默认16k)
```

使用示例详见：https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/onnxruntime/bin/funasr-onnx-offline-vad.cpp

#### ASR
```text
// ASR模型的使用分为FunOfflineInit和FunOfflineInfer两个步骤：
FUNASR_HANDLE asr_hanlde=FunOfflineInit(model_path, thread_num);
// 其中：model_path 包含"model-dir"、"quantize"，thread_num为onnx线程数；
FUNASR_RESULT result=FunOfflineInfer(asr_hanlde, wav_file.c_str(), RASR_NONE, NULL, 16000);
// 其中：asr_hanlde为FunOfflineInit返回值，wav_file为音频路径，sampling_rate为采样率(默认16k)
```

使用示例详见：https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/onnxruntime/bin/funasr-onnx-offline.cpp

#### PUNC
```text
// PUNC模型的使用分为CTTransformerInit和CTTransformerInfer两个步骤：
FUNASR_HANDLE punc_hanlde=CTTransformerInit(model_path, thread_num);
// 其中：model_path 包含"model-dir"、"quantize"，thread_num为onnx线程数；
FUNASR_RESULT result=CTTransformerInfer(punc_hanlde, txt_str.c_str(), RASR_NONE, NULL);
// 其中：punc_hanlde为CTTransformerInit返回值，txt_str为文本
```
使用示例详见：https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/onnxruntime/bin/funasr-onnx-offline-punc.cpp
