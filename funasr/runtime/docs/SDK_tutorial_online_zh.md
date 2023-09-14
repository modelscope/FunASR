(简体中文|[English](./SDK_tutorial_online.md))

# FunASR实时语音听写便捷部署教程

FunASR提供可便捷本地或者云端服务器部署的实时语音听写服务，内核为FunASR已开源的runtime-SDK。
FunASR集成了达摩院语音实验室在Modelscope社区开源的语音端点检测(VAD)、Paraformer-large非流式语音识别(ASR)、Paraformer-large流式语音识别(ASR)、标点预测(PUNC) 等相关能力。软件包既可以实时地进行语音转文字，而且能够在说话句尾用高精度的转写文字修正输出，输出文字带有标点，支持高并发多路请求

## 服务器配置
 
用户可以根据自己的业务需求，选择合适的服务器配置，推荐配置为：
- 配置1: （X86，计算型），4核vCPU，内存8G，单机可以支持大约16路的请求
- 配置2: （X86，计算型），16核vCPU，内存32G，单机可以支持大约32路的请求
- 配置3: （X86，计算型），64核vCPU，内存128G，单机可以支持大约100路的请求

详细性能测试报告（[点击此处](./benchmark_onnx_cpp.md)）

云服务厂商，针对新用户，有3个月免费试用活动，申请教程（[点击此处](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/docs/aliyun_server_tutorial.md)）

## 快速上手

### 服务端启动

`注意`：一键部署工具，过程分为：安装docker、下载docker镜像、启动服务。如果用户希望直接从FunASR docker镜像启动，可以参考开发指南（[点击此处](./SDK_advanced_guide_online_zh.md)）

下载部署工具`funasr-runtime-deploy-online-cpu-zh.sh`

```shell
curl -O https://raw.githubusercontent.com/alibaba-damo-academy/FunASR/main/funasr/runtime/deploy_tools/funasr-runtime-deploy-online-cpu-zh.sh;
# 如遇到网络问题，中国大陆用户，可以使用下面的命令：
# curl -O https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/shell/funasr-runtime-deploy-online-cpu-zh.sh;
```

执行部署工具，在提示处输入回车键即可完成服务端安装与部署。目前便捷部署工具暂时仅支持Linux环境，其他环境部署参考开发指南（[点击此处](./SDK_advanced_guide_online_zh.md)）
```shell
sudo bash funasr-runtime-deploy-online-cpu-zh.sh install --workspace ./funasr-runtime-resources
```
**注：如果需要部署时间戳模型或者热词模型，在安装部署步骤2时选择对应模型，其中1为paraformer-large模型，2为paraformer-large 时间戳模型，3为paraformer-large 热词模型**

### 客户端测试与使用

运行上面安装指令后，会在./funasr-runtime-resources（默认安装目录）中下载客户端测试工具目录samples（手动下载，[点击此处](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz)），
我们以Python语言客户端为例，支持麦克风输入，其他版本客户端请参考文档（[点击此处](#客户端用法详解)）

```shell
python3 funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode 2pass
```

## 客户端用法详解

在服务器上完成FunASR服务部署以后，可以通过如下的步骤来测试和使用实时语音听写服务。
目前分别支持以下几种编程语言客户端

- [Python](#python-client)
- [CPP](#cpp-client)
- [html](#html-client)
- [java](#java-client)
- [c\#](#c\#)

更多版本客户端支持请参考[websocket/grpc协议](./websocket_protocol_zh.md)

### python-client
若想直接运行client进行测试，可参考如下简易说明，以python版本为例：

```shell
python3 funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode 2pass
```

命令参数说明：
```text
--host 为FunASR runtime-SDK服务部署机器ip，默认为本机ip（127.0.0.1），如果client与服务不在同一台服务器，需要改为部署机器ip
--port 10095 部署端口号
--mode：`offline`表示推理模式为一句话识别；`online`表示推理模式为实时语音识别；`2pass`表示为实时语音识别，并且说话句尾采用离线模型进行纠错。
--chunk_size：表示流式模型latency配置`[5,10,5]`，表示当前音频解码片段为600ms，并且回看300ms，右看300ms。
--audio_in 需要进行转写的音频文件，支持文件路径，文件列表wav.scp
--thread_num 设置并发发送线程数，默认为1
--ssl 设置是否开启ssl证书校验，默认1开启，设置为0关闭+
--hotword 如果模型为热词模型，可以设置热词: *.txt(每行一个热词) 或者空格分隔的热词字符串 (could be: 阿里巴巴 达摩院)
--use_itn 设置是否使用itn，默认1开启，设置为0关闭
```

### cpp-client
进入samples/cpp目录后，可以用cpp进行测试，指令如下：
```shell
./funasr-wss-client-2pass --server-ip 127.0.0.1 --port 10095 --mode 2pass --wav-path ../audio/asr_example.wav
```

命令参数说明：

```text
--server-ip 为FunASR runtime-SDK服务部署机器ip，默认为本机ip（127.0.0.1），如果client与服务不在同一台服务器，需要改为部署机器ip
--port 10095 部署端口号
--mode：`offline`表示推理模式为一句话识别；`online`表示推理模式为实时语音识别；`2pass`表示为实时语音识别，并且说话句尾采用离线模型进行纠错。
--chunk-size：表示流式模型latency配置`[5,10,5]`，表示当前音频解码片段为600ms，并且回看300ms，右看300ms。
--wav-path 需要进行转写的音频文件，支持文件路径
--thread-num 设置并发发送线程数，默认为1
--is-ssl 设置是否开启ssl证书校验，默认1开启，设置为0关闭
--hotword 如果模型为热词模型，可以设置热词: *.txt(每行一个热词) 或者空格分隔的热词字符串 (could be: 阿里巴巴 达摩院)
--use-itn 设置是否使用itn，默认1开启，设置为0关闭
```

### html-client

在浏览器中打开 html/static/index.html，即可出现如下页面，支持麦克风输入与文件上传，直接进行体验

<img src="images/html.png"  width="900"/>

### java-client

```shell
FunasrWsClient --host localhost --port 10095 --mode 2pass
```
详细可以参考文档（[点击此处](../java/readme.md)）

## 服务端用法详解

### 启动已经部署过的FunASR服务
一键部署后若出现重启电脑等关闭Docker的动作，可通过如下命令直接启动FunASR服务，启动配置为上次一键部署的设置。

```shell
sudo bash funasr-runtime-deploy-online-cpu-zh.sh start
```



### 关闭FunASR服务

```shell
sudo bash funasr-runtime-deploy-online-cpu-zh.sh stop
```

### 释放FunASR服务

释放已经部署的FunASR服务。
```shell
sudo bash funasr-runtime-deploy-online-cpu-zh.sh remove
```

### 重启FunASR服务

根据上次一键部署的设置重启启动FunASR服务。
```shell
sudo bash funasr-runtime-deploy-online-cpu-zh.sh restart
```

### 替换模型并重启FunASR服务

替换正在使用的模型，并重新启动FunASR服务。模型需为ModelScope中的ASR/VAD/PUNC模型，或者从ModelScope中模型finetune后的模型。

```shell
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update [--asr_model | --vad_model | --punc_model] <model_id or local model path>

e.g
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update --asr_model damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```

### 更新参数并重启FunASR服务

更新已配置参数，并重新启动FunASR服务生效。可更新参数包括宿主机和Docker的端口号，以及推理和IO的线程数量。

```shell
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update [--host_port | --docker_port] <port number>
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update [--decode_thread_num | --io_thread_num] <the number of threads>
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update [--workspace] <workspace in local>
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update [--ssl] <0: close SSL; 1: open SSL, default:1>

e.g
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update --decode_thread_num 32
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update --workspace ./funasr-runtime-resources
```

### 关闭SSL证书

```shell
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update --ssl 0
```

## 联系我们

在您使用过程中，如果遇到问题，欢迎加入用户群进行反馈


|                                    钉钉用户群                                     |                                      微信               |
|:----------------------------------------------------------------------------:|:-----------------------------------------------------:|
| <div align="left"><img src="../../../docs/images/dingding.jpg" width="250"/> | <img src="../../../docs/images/wechat.png" width="232"/></div> |


## 视频demo

[点击此处]()















