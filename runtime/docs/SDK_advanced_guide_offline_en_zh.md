# FunASR英文离线文件转写服务开发指南

FunASR提供可一键本地或者云端服务器部署的英文离线文件转写服务，内核为FunASR已开源runtime-SDK。FunASR-runtime结合了达摩院语音实验室在Modelscope社区开源的语音端点检测(VAD)、Paraformer-large语音识别(ASR)、标点检测(PUNC) 等相关能力，可以准确、高效的对音频进行高并发转写。

本文档为FunASR离线文件转写服务开发指南。如果您想快速体验离线文件转写服务，可参考[快速上手](#快速上手)。

| 时间         | 详情            | 镜像版本                            | 镜像ID         |
|------------|---------------|---------------------------------|--------------|
| 2024.05.15 | 适配FunASR 1.0模型结构 | funasr-runtime-sdk-en-cpu-0.1.6 | 84d781d07997 |
| 2024.03.05 | docker镜像支持arm64平台，升级modelscope版本 | funasr-runtime-sdk-en-cpu-0.1.5 | 7cca2abc5901 |
| 2024.01.25 | 优化vad数据处理方式，大幅降低峰值内存占用；内存泄漏优化 | funasr-runtime-sdk-en-cpu-0.1.3 | c00f9ce7a195 |
| 2024.01.03 | 修复已知的crash问题及内存泄漏问题 | funasr-runtime-sdk-en-cpu-0.1.2 | 0cdd9f4a4bb5 |
| 2023.11.08 | runtime结构变化适配 | funasr-runtime-sdk-en-cpu-0.1.1 | 27017f70f72a |
| 2023.10.16 | 1.0 发布        | funasr-runtime-sdk-en-cpu-0.1.0 | e0de03eb0163 |


## 服务器配置

用户可以根据自己的业务需求，选择合适的服务器配置，推荐配置为：
- 配置1: （X86，计算型），4核vCPU，内存8G，单机可以支持大约32路的请求
- 配置2: （X86，计算型），16核vCPU，内存32G，单机可以支持大约64路的请求
- 配置3: （X86，计算型），64核vCPU，内存128G，单机可以支持大约200路的请求

详细性能测试报告（[点击此处](./benchmark_onnx_cpp.md)）

云服务厂商，针对新用户，有3个月免费试用活动，申请教程（[点击此处](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/aliyun_server_tutorial.md)）


## 快速上手
### docker安装
如果您已安装docker，忽略本步骤！!
通过下述命令在服务器上安装docker：
```shell
curl -O https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/shell/install_docker.sh；
sudo bash install_docker.sh
```
docker安装失败请参考 [Docker Installation](https://alibaba-damo-academy.github.io/FunASR/en/installation/docker.html)

### 镜像启动
通过下述命令拉取并启动FunASR runtime-SDK的docker镜像：
```shell
sudo docker pull \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-en-cpu-0.1.6
mkdir -p ./funasr-runtime-resources/models
sudo docker run -p 10097:10095 -it --privileged=true \
  -v $PWD/funasr-runtime-resources/models:/workspace/models \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-en-cpu-0.1.6
```

### 服务端启动

docker启动之后，启动 funasr-wss-server服务程序：
```shell
cd FunASR/runtime
nohup bash run_server.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large_asr_nat-en-16k-common-vocab10020-onnx  \
  --punc-dir damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx  > log.txt 2>&1 &

# 如果您想关闭ssl，增加参数：--certfile 0

```
服务端详细参数介绍可参考[服务端用法详解](#服务端用法详解)
### 客户端测试与使用

下载客户端测试工具目录samples
```shell
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz
```
我们以Python语言客户端为例，进行说明，支持多种音频格式输入（.wav, .pcm, .mp3等），也支持视频输入(.mp4等)，以及多文件列表wav.scp输入，其他版本客户端请参考文档（[点击此处](#客户端用法详解)），定制服务部署请参考[如何定制服务部署](#如何定制服务部署)
```shell
python3 funasr_wss_client.py --host "127.0.0.1" --port 10097 --mode offline --audio_in "../audio/asr_example.wav"
```

------------------
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
python3 funasr_wss_client.py --host "127.0.0.1" --port 10097 --mode offline \
        --audio_in "../audio/asr_example.wav" --output_dir "./results"
```

命令参数说明：
```text
--host 为FunASR runtime-SDK服务部署机器ip，默认为本机ip（127.0.0.1），如果client与服务不在同一台服务器，
       需要改为部署机器ip
--port 10097 部署端口号
--mode offline表示离线文件转写
--audio_in 需要进行转写的音频文件，支持文件路径，文件列表wav.scp
--thread_num 设置并发发送线程数，默认为1
--ssl 设置是否开启ssl证书校验，默认1开启，设置为0关闭
--hotword 热词文件，每行一个热词，格式(热词 权重)：阿里巴巴 20
--use_itn 设置是否使用itn，默认1开启，设置为0关闭
```

### cpp-client
进入samples/cpp目录后，可以用cpp进行测试，指令如下：
```shell
./funasr-wss-client --server-ip 127.0.0.1 --port 10097 --wav-path ../audio/asr_example.wav
```

命令参数说明：

```text
--server-ip 为FunASR runtime-SDK服务部署机器ip，默认为本机ip（127.0.0.1），如果client与服务不在同一台服务器，
            需要改为部署机器ip
--port 10097 部署端口号
--wav-path 需要进行转写的音频文件，支持文件路径
--hotword 热词文件，每行一个热词，格式(热词 权重)：阿里巴巴 20
--use-itn 设置是否使用itn，默认1开启，设置为0关闭
```

### Html网页版

在浏览器中打开 html/static/index.html，即可出现如下页面，支持麦克风输入与文件上传，直接进行体验

<img src="images/html.png"  width="900"/>

### Java-client

```shell
FunasrWsClient --host localhost --port 10097 --audio_in ./asr_example.wav --mode offline
```
详细可以参考文档（[点击此处](../java/readme.md)）

## 服务端用法详解：

### 启动FunASR服务
```shell
cd /workspace/FunASR/runtime
nohup bash run_server.sh \
  --download-model-dir /workspace/models \
  --model-dir damo/speech_paraformer-large_asr_nat-en-16k-common-vocab10020-onnx \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --punc-dir damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx \
  --certfile  ../../../ssl_key/server.crt \
  --keyfile ../../../ssl_key/server.key > log.txt 2>&1 &
 ```
**run_server.sh命令参数介绍**
```text
--download-model-dir 模型下载地址，通过设置model ID从Modelscope下载模型
--model-dir  modelscope model ID 或者 本地模型路径
--vad-dir  modelscope model ID 或者 本地模型路径
--punc-dir  modelscope model ID 或者 本地模型路径
--itn-dir modelscope model ID 或者 本地模型路径
--port  服务端监听的端口号，默认为 10095
--decoder-thread-num  服务端线程池个数(支持的最大并发路数)，
                      脚本会根据服务器线程数自动配置decoder-thread-num、io-thread-num
--io-thread-num  服务端启动的IO线程数
--model-thread-num  每路识别的内部线程数(控制ONNX模型的并行)，默认为 1，
                    其中建议 decoder-thread-num*model-thread-num 等于总线程数
--certfile  ssl的证书文件，默认为：../../../ssl_key/server.crt，如果需要关闭ssl，参数设置为0
--keyfile   ssl的密钥文件，默认为：../../../ssl_key/server.key
```

### 关闭FunASR服务
```text
# 查看 funasr-wss-server 对应的PID
ps -x | grep funasr-wss-server
kill -9 PID
```

### 修改模型及其他参数
替换正在使用的模型或者其他参数，需先关闭FunASR服务，修改需要替换的参数，并重新启动FunASR服务。其中模型需为ModelScope中的ASR/VAD/PUNC模型，或者从ModelScope中模型finetune后的模型。
```text
# 例如替换ASR模型为 damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx，则如下设置参数 --model-dir
    --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx 
# 设置端口号 --port
    --port <port number>
# 设置服务端启动的推理线程数 --decoder-thread-num
    --decoder-thread-num <decoder thread num>
# 设置服务端启动的IO线程数 --io-thread-num
    --io-thread-num <io thread num>
# 关闭SSL证书 
    --certfile 0
```

执行上述指令后，启动英文离线文件转写服务。如果模型指定为ModelScope中model id，会自动从MoldeScope中下载如下模型：
[FSMN-VAD模型](https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx/summary),
[Paraformer-lagre模型](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-en-16k-common-vocab10020-onnx/summary),
[CT-Transformer标点预测模型](https://www.modelscope.cn/models/damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx/summary)

如果，您希望部署您finetune后的模型（例如10epoch.pb），需要手动将模型重命名为model.pb，并将原modelscope中模型model.pb替换掉，将路径指定为`model_dir`即可。

## 如何定制服务部署

FunASR-runtime的代码已开源，如果服务端和客户端不能很好的满足您的需求，您可以根据自己的需求进行进一步的开发：
### c++ 客户端：

https://github.com/alibaba-damo-academy/FunASR/tree/main/runtime/websocket

### python 客户端：

https://github.com/alibaba-damo-academy/FunASR/tree/main/runtime/python/websocket

### 自定义客户端：

如果您想定义自己的client，参考[websocket通信协议](./websocket_protocol_zh.md)

### c++ 服务端：

#### VAD
```c++
// VAD模型的使用分为FsmnVadInit和FsmnVadInfer两个步骤：
FUNASR_HANDLE vad_hanlde=FsmnVadInit(model_path, thread_num);
// 其中：model_path 包含"model-dir"、"quantize"，thread_num为onnx线程数；
FUNASR_RESULT result=FsmnVadInfer(vad_hanlde, wav_file.c_str(), NULL, 16000);
// 其中：vad_hanlde为FunOfflineInit返回值，wav_file为音频路径，sampling_rate为采样率(默认16k)
```

使用示例详见：https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/onnxruntime/bin/funasr-onnx-offline-vad.cpp

#### ASR
```text
// ASR模型的使用分为FunOfflineInit和FunOfflineInfer两个步骤：
FUNASR_HANDLE asr_hanlde=FunOfflineInit(model_path, thread_num);
// 其中：model_path 包含"model-dir"、"quantize"，thread_num为onnx线程数；
FUNASR_RESULT result=FunOfflineInfer(asr_hanlde, wav_file.c_str(), RASR_NONE, NULL, 16000);
// 其中：asr_hanlde为FunOfflineInit返回值，wav_file为音频路径，sampling_rate为采样率(默认16k)
```

使用示例详见：https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/onnxruntime/bin/funasr-onnx-offline.cpp

#### PUNC
```text
// PUNC模型的使用分为CTTransformerInit和CTTransformerInfer两个步骤：
FUNASR_HANDLE punc_hanlde=CTTransformerInit(model_path, thread_num);
// 其中：model_path 包含"model-dir"、"quantize"，thread_num为onnx线程数；
FUNASR_RESULT result=CTTransformerInfer(punc_hanlde, txt_str.c_str(), RASR_NONE, NULL);
// 其中：punc_hanlde为CTTransformerInit返回值，txt_str为文本
```
使用示例详见：https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/onnxruntime/bin/funasr-onnx-offline-punc.cpp
