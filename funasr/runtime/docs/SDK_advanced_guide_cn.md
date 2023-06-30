# FunASR离线文件转写服务开发指南

FunASR提供可一键本地或者云端服务器部署的中文离线文件转写服务，内核为FunASR已开源runtime-SDK。FunASR-runtime结合了达摩院语音实验室在Modelscope社区开源的语音端点检测(VAD)、Paraformer-large语音识别(ASR)、标点检测(PUNC) 等相关能力，可以准确、高效的对音频进行高并发转写。

本文档为FunASR离线文件转写服务开发指南。如果您想快速体验离线文件转写服务，请参考FunASR离线文件转写服务一键部署示例（[点击此处](./SDK_tutorial_cn.md)）。

## Docker安装

下述步骤为手动安装docker及docker镜像的步骤，如您docker镜像已启动，可以忽略本步骤：

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

### 镜像拉取及启动

通过下述命令拉取并启动FunASR runtime-SDK的docker镜像：

```shell
sudo docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.0.1

sudo docker run -p 10095:10095 -it --privileged=true -v /root:/workspace/models registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.0.1
```

命令参数介绍：
```text
-p <宿主机端口>:<映射到docker端口>
如示例，宿主机(ecs)端口10095映射到docker端口10095上。前提是确保ecs安全规则打开了10095端口。
-v <宿主机路径>:<挂载至docker路径>
如示例，宿主机路径/root挂载至docker路径/workspace/models
```


## 服务端启动

docker启动之后，启动 funasr-wss-server服务程序：

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
--download-model-dir #模型下载地址，通过设置model ID从Modelscope下载模型
--model-dir # modelscope model ID
--quantize  # True为量化ASR模型，False为非量化ASR模型，默认是True
--vad-dir # modelscope model ID
--vad-quant  # True为量化VAD模型，False为非量化VAD模型，默认是True
--punc-dir # modelscope model ID
--punc-quant  # True为量化PUNC模型，False为非量化PUNC模型，默认是True
--port # 服务端监听的端口号，默认为 10095
--decoder-thread-num # 服务端启动的推理线程数，默认为 8
--io-thread-num # 服务端启动的IO线程数，默认为 1
--certfile <string> # ssl的证书文件，默认为：../../../ssl_key/server.crt
--keyfile <string> # ssl的密钥文件，默认为：../../../ssl_key/server.key
```

funasr-wss-server同时也支持从本地路径加载模型（本地模型资源准备详见[模型资源准备](#anchor-1)）示例如下：
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
--model-dir # ASR模型路径，默认为：/workspace/models/asr
--quantize  # True为量化ASR模型，False为非量化ASR模型，默认是True
--vad-dir # VAD模型路径，默认为：/workspace/models/vad
--vad-quant  # True为量化VAD模型，False为非量化VAD模型，默认是True
--punc-dir # PUNC模型路径，默认为：/workspace/models/punc
--punc-quant  # True为量化PUNC模型，False为非量化PUNC模型，默认是True
--port # 服务端监听的端口号，默认为 10095
--decoder-thread-num # 服务端启动的推理线程数，默认为 8
--io-thread-num # 服务端启动的IO线程数，默认为 1
--certfile <string> # ssl的证书文件，默认为：../../../ssl_key/server.crt
--keyfile <string> # ssl的密钥文件，默认为：../../../ssl_key/server.key
```

## <a id="anchor-1">模型资源准备</a>

如果您选择通过funasr-wss-server从Modelscope下载模型，可以跳过本步骤。

FunASR离线文件转写服务中的vad、asr和punc模型资源均来自Modelscope，模型地址详见下表：

| 模型 | Modelscope链接                                                                                                     |
|------|------------------------------------------------------------------------------------------------------------------|
| VAD  | https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary |
| ASR  | https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary                           |
| PUNC | https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary               |

离线文件转写服务中部署的是量化后的ONNX模型，下面介绍下如何导出ONNX模型及其量化：您可以选择从Modelscope导出ONNX模型、从本地文件导出ONNX模型或者从finetune后的资源导出模型：

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

### 从本地文件导出ONNX模型

设置model name为模型本地路径，导出量化后的ONNX模型：

```shell
python -m funasr.export.export_model --model-name /workspace/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type onnx --quantize True
```
命令参数介绍：
```text
--model-name  模型本地路径，例如/workspace/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
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

## 客户端启动

在服务器上完成FunASR离线文件转写服务部署以后，可以通过如下的步骤来测试和使用离线文件转写服务。目前FunASR-bin支持多种方式启动客户端，如下是基于python-client、c++-client的命令行实例及自定义客户端Websocket通信协议：

### python-client
```shell
python wss_client_asr.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "./data/wav.scp" --send_without_sleep --output_dir "./results"
```
命令参数介绍：
```text
--host # 服务端ip地址，本机测试可设置为 127.0.0.1
--port # 服务端监听端口号
--audio_in # 音频输入，输入可以是：wav路径 或者 wav.scp路径（kaldi格式的wav list，wav_id \t wav_path）
--output_dir # 识别结果输出路径
--ssl # 是否使用SSL加密，默认使用
--mode # offline模式
```

### c++-client：
```shell
. /funasr-wss-client --server-ip 127.0.0.1 --port 10095 --wav-path test.wav --thread-num 1 --is-ssl 1
```
命令参数介绍：
```text
--server-ip # 服务端ip地址，本机测试可设置为 127.0.0.1
--port # 服务端监听端口号
--wav-path # 音频输入，输入可以是：wav路径 或者 wav.scp路径（kaldi格式的wav list，wav_id \t wav_path）
--thread-num # 客户端线程数
--is-ssl # 是否使用SSL加密，默认使用
```

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

## 如何定制服务部署

FunASR-runtime的代码已开源，如果服务端和客户端不能很好的满足您的需求，您可以根据自己的需求进行进一步的开发：
### c++ 客户端：

https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/websocket

### python 客户端：

https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/websocket
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
