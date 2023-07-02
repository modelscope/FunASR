# FunASR离线文件转写服务便捷部署教程

FunASR提供可便捷本地或者云端服务器部署的离线文件转写服务，内核为FunASR已开源runtime-SDK。
集成了达摩院语音实验室在Modelscope社区开源的语音端点检测(VAD)、Paraformer-large语音识别(ASR)、标点恢复(PUNC) 等相关能力，拥有完整的语音识别链路，可以将几十个小时的音频或视频识别成带标点的文字，而且支持上百路请求同时进行转写。

## 服务器配置

用户可以根据自己的业务需求，选择合适的服务器配置，推荐配置为：
- 配置1: （X86，计算型），4核vCPU，内存8G，单机可以支持大约32路的请求
- 配置2: （X86，计算型），16核vCPU，内存32G，单机可以支持大约64路的请求
- 配置3: （X86，计算型），64核vCPU，内存128G，单机可以支持大约200路的请求

云服务厂商，针对新用户，有3个月免费试用活动，申请教程（[点击此处](./aliyun_server_tutorial.md)）

## 快速上手

### 服务端启动

下载部署工具`funasr-runtime-deploy-offline-cpu-zh.sh`

```shell
curl -O https://raw.githubusercontent.com/alibaba-damo-academy/FunASR/main/funasr/runtime/deploy_tools/funasr-runtime-deploy-offline-cpu-zh.sh;
# 如遇到网络问题，中国大陆用户，可以用个下面的命令：
# curl -O https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/shell/funasr-runtime-deploy-offline-cpu-zh.sh;
```

执行部署工具，在提示处输入回车键即可完成服务端安装与部署。目前便捷部署工具暂时仅支持Linux环境，其他环境部署参考开发指南（[点击此处](./SDK_advanced_guide_zh.md)）
```shell
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh install --workspace /root/funasr-runtime-resources
```

### 客户端测试与使用

运行上面安装指令后，会在/root/funasr-runtime-resources（默认安装目录）中下载客户端测试工具目录samples，
我们以Python语言客户端为例，进行说明，支持多种音频格式输入（.wav, .pcm, .mp3等），也支持视频输入(.mp4等)，以及多文件列表wav.scp输入，其他版本客户端请参考文档（[点击此处](#客户端用法详解)）

```shell
python3 wss_client_asr.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "../audio/asr_example.wav" --output_dir "./results"
```

## 客户端用法详解

在服务器上完成FunASR服务部署以后，可以通过如下的步骤来测试和使用离线文件转写服务。
目前分别支持以下几种编程语言客户端

- [Python](#python-client)
- [CPP](#cpp-client)
- [Java]()
- [html网页版本](#Html网页版)

更多版本客户端支持请参考[开发指南](./SDK_advanced_guide_offline_zh.md)

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
export LD_LIBRARY_PATH=./libs:$LD_LIBRARY_PATH
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


## 服务端用法详解

### 启动已经部署过的FunASR服务
一键部署后若出现重启电脑等关闭Docker的动作，可通过如下命令直接启动FunASR服务，启动配置为上次一键部署的设置。

```shell
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh start
```

### 关闭FunASR服务

```shell
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh stop
```

### 重启FunASR服务

根据上次一键部署的设置重启启动FunASR服务。
```shell
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh restart
```

### 替换模型并重启FunASR服务

替换正在使用的模型，并重新启动FunASR服务。模型需为ModelScope中的ASR/VAD/PUNC模型，或者从ModelScope中模型finetune后的模型。

```shell
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh update [--asr_model | --vad_model | --punc_model] <model_id or local model path>

e.g
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh update --asr_model damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```

### 更新参数并重启FunASR服务

更新已配置参数，并重新启动FunASR服务生效。可更新参数包括宿主机和Docker的端口号，以及推理和IO的线程数量。

```shell
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh update [--host_port | --docker_port] <port number>
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh update [--decode_thread_num | --io_thread_num] <the number of threads>
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh update [--workspace] <workspace in local>

e.g
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh update --decode_thread_num 32
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh update --workspace /root/funasr-runtime-resources
```


## 服务端启动过程配置详解

##### 选择FunASR Docker镜像
推荐选择1)使用我们的最新发布版镜像，也可选择历史版本。
```text
[1/5]
  Getting the list of docker images, please wait a few seconds.
    [DONE]

  Please choose the Docker image.
    1) registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.1.0
  Enter your choice, default(1):
  You have chosen the Docker image: registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.1.0
```


##### 设置宿主机提供给FunASR的端口
设置提供给Docker的宿主机端口，默认为10095。请保证此端口可用。
```text
[2/5]
  Please input the opened port in the host used for FunASR server.
  Setting the opened host port [1-65535], default(10095):
  The port of the host is 10095
  The port in Docker for FunASR server is 10095
```


## 视频demo

[点击此处]()















