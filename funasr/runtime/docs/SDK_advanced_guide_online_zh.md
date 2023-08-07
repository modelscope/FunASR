# FunASR实时语音转写服务开发指南

FunASR提供可便捷本地或者云端服务器部署的实时语音转写服务，内核为FunASR已开源runtime-SDK。
集成了达摩院语音实验室在Modelscope社区开源的语音端点检测(VAD)、Paraformer-large非流式语音识别(ASR)、Paraformer-large流式语音识别(ASR)、标点(PUNC) 等相关能力。软件包既可以实时地进行语音转文字，而且能够在说话句尾用高精度的转写文字修正输出，输出文字带有标点，支持高并发多路请求

本文档为FunASR实时转写服务开发指南。如果您想快速体验实时语音转写服务，可参考[快速上手](#快速上手)。

## 快速上手
### 镜像启动

通过下述命令拉取并启动FunASR runtime-SDK的docker镜像：

```shell
sudo docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.0

sudo docker run -p 10095:10095 -it --privileged=true -v /root:/workspace/models registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.0
```
如果您没有安装docker，可参考[Docker安装](#Docker安装)

### 服务端启动

docker启动之后，启动 funasr-wss-server-2pass服务程序：
```shell
cd FunASR/funasr/runtime
./run_server_2pass.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx  \
  --online-model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx  \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx
```
服务端详细参数介绍可参考[服务端参数介绍](#服务端参数介绍)
### 客户端测试与使用

下载客户端测试工具目录samples
```shell
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_online_samples.tar.gz
```
我们以Python语言客户端为例，进行说明，支持多种音频格式输入（.wav, .pcm, .mp3等），也支持视频输入(.mp4等)，以及多文件列表wav.scp输入，其他版本客户端请参考文档（[点击此处](#客户端用法详解)），定制服务部署请参考[如何定制服务部署](#如何定制服务部署)
```shell
python3 wss_client_asr.py --host "127.0.0.1" --port 10095 --mode 2pass
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
python3 wss_client_asr.py --host "127.0.0.1" --port 10095 --mode 2pass --audio_in "../audio/asr_example.wav" --output_dir "./results"
```

命令参数说明：
```text
--host 为FunASR runtime-SDK服务部署机器ip，默认为本机ip（127.0.0.1），如果client与服务不在同一台服务器，需要改为部署机器ip
--port 10095 部署端口号
--mode 2pass 表示online+offline
--audio_in 需要进行转写的音频文件，支持文件路径，文件列表wav.scp
--output_dir 识别结果保存路径
```

### cpp-client
进入samples/cpp目录后，可以用cpp进行测试，指令如下：
```shell
./funasr-wss-client-2pass --server-ip 127.0.0.1 --port 10095 --wav-path ../audio/asr_example.wav
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
./funasr-wss-server-2pass  \
  --download-model-dir /workspace/models \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx \
  --online-model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx  \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx \
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
--online-model-dir  modelscope model ID
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

## 模型资源准备

如果您选择通过funasr-wss-server-2pass 从Modelscope下载模型，可以跳过本步骤。

FunASR离线文件转写服务中的vad、asr和punc模型资源均来自Modelscope，模型地址详见下表：

| 模型 | Modelscope链接                                                                                                  |
|------|---------------------------------------------------------------------------------------------------------------|
| VAD  | https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx/summary  |
| ASR  | https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx/summary                           |
| ASR  | https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx/summary                          |
| PUNC | https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx/summary               |

实时转写服务中部署的是量化后的ONNX模型，下面介绍下如何导出ONNX模型及其量化：您可以选择从Modelscope导出ONNX模型、从finetune后的资源导出模型：

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
首次通信
message为（需要用json序列化）：
{"mode": "offline", "wav_name": "wav_name", "is_speaking": True, "wav_format":"pcm", "chunk_size":[5,10,5]}
参数介绍：
`mode`：`offline`，表示推理模式为一句话识别；`online`，表示推理模式为实时语音识别；`2pass`：表示为实时语音识别，并且说话句尾采用离线模型进行纠错。
`wav_name`：表示需要推理音频文件名
`wav_format`：表示音视频文件后缀名，可选pcm、mp3、mp4等（备注，1.0版本只支持pcm音频流）
`is_speaking`：表示断句尾点，例如，vad切割点，或者一条wav结束
`chunk_size`：表示流式模型latency配置，`[5,10,5]`，表示当前音频为600ms，并且回看300ms，右看300ms。
`audio_fs`：当输入音频为pcm数据时，需要加上音频采样率参数

发送音频数据
直接将音频数据，移除头部信息后的bytes数据发送，支持音频采样率为80000，16000
发送结束标志
音频数据发送结束后，需要发送结束标志（需要用json序列化）：
{"is_speaking": False}
```
