# FunASR离线文件转写服务便捷部署教程

FunASR提供可便捷本地或者云端服务器部署的离线文件转写服务，内核为FunASR已开源runtime-SDK。集成了达摩院语音实验室在Modelscope社区开源的语音端点检测(VAD)、Paraformer-large语音识别(ASR)、标点恢复(PUNC) 等相关能力，可以准确、高效的对音频进行高并发转写。

## 环境安装与启动服务

服务器配置与申请（免费试用1～3个月）（[点击此处](./aliyun_server_tutorial.md)）
### 获得脚本工具并一键部署

通过以下命令运行一键部署服务，按照提示逐步完成FunASR runtime-SDK服务的部署和运行。目前暂时仅支持Linux环境，其他环境参考文档[高阶开发指南](./SDK_advanced_guide_cn.md)

[//]: # (受限于网络，funasr-runtime-deploy.sh一键部署工具的下载可能不顺利，遇到数秒还未下载进入一键部署工具的情况，请Ctrl + C 终止后再次运行以下命令。)

```shell
curl -O https://raw.githubusercontent.com/alibaba-damo-academy/FunASR/main/funasr/runtime/funasr-runtime-deploy.sh; sudo bash funasr-runtime-deploy.sh install
# 如遇到网络问题，中国大陆用户，可以用个下面的命令：
# curl -O https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/shell/funasr-runtime-deploy.sh; sudo bash funasr-runtime-deploy.sh install

```

#### 启动过程配置详解

##### 选择FunASR Docker镜像
推荐选择latest使用我们的最新镜像，也可选择历史版本。
```text
[1/9]
  Please choose the Docker image.
    1) registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-latest
    2) registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.1.0
  Enter your choice: 1
  You have chosen the Docker image: registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-latest
```

##### 选择ASR/VAD/PUNC模型

你可以选择ModelScope中的模型，也可以选<model_name>自行填入ModelScope中的模型名，将会在Docker运行时自动下载。同时也可以选择<model_path>填入宿主机中的本地模型路径。

```text
[2/9]
  Please input [Y/n] to confirm whether to automatically download model_id in ModelScope or use a local model.
  [y] With the model in ModelScope, the model will be automatically downloaded to Docker(/workspace/models).
      If you select both the local model and the model in ModelScope, select [y].
  [n] Use the models on the localhost, the directory where the model is located will be mapped to Docker.
  Setting confirmation[Y/n]: 
  You have chosen to use the model in ModelScope, please set the model ID in the next steps, and the model will be automatically downloaded in (/workspace/models) during the run.

  Please enter the local path to download models, the corresponding path in Docker is /workspace/models.
  Setting the local path to download models, default(/root/models): 
  The local path(/root/models) set will store models during the run.

  [2.1/9]
    Please select ASR model_id in ModelScope from the list below.
    1) damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx
    2) model_name
    3) model_path
  Enter your choice: 1
    The model ID is damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx
    The model dir in Docker is /workspace/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx

  [2.2/9]
    Please select VAD model_id in ModelScope from the list below.
    1) damo/speech_fsmn_vad_zh-cn-16k-common-onnx
    2) model_name
    3) model_path
  Enter your choice: 1
    The model ID is damo/speech_fsmn_vad_zh-cn-16k-common-onnx
    The model dir in Docker is /workspace/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx

  [2.3/9]
    Please select PUNC model_id in ModelScope from the list below.
    1) damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx
    2) model_name
    3) model_path
  Enter your choice: 1
    The model ID is damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx
    The model dir in Docker is /workspace/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx
```

##### 输入宿主机中FunASR服务可执行程序路径

输入FunASR服务可执行程序的宿主机路径，Docker运行时将自动挂载到Docker中运行。默认不输入的情况下将指定Docker中默认的/workspace/FunASR/funasr/runtime/websocket/build/bin/funasr-wss-server。

```text
[3/9]
  Please enter the path to the excutor of the FunASR service on the localhost.
  If not set, the default /workspace/FunASR/funasr/runtime/websocket/build/bin/funasr-wss-server in Docker is used.
  Setting the path to the excutor of the FunASR service on the localhost: 
  Corresponding, the path of FunASR in Docker is /workspace/FunASR/funasr/runtime/websocket/build/bin/funasr-wss-server
```

##### 设置宿主机提供给FunASR的端口
设置提供给Docker的宿主机端口，默认为10095。请保证此端口可用。
```text
[4/9]
  Please input the opened port in the host used for FunASR server.
  Default: 10095
  Setting the opened host port [1-65535]: 
  The port of the host is 10095
  The port in Docker for FunASR server is 10095
```


##### 设置FunASR服务的推理线程数
设置FunASR服务的推理线程数，默认为宿主机核数，同时自动设置服务的IO线程数，为推理线程数的四分之一。
```text
[5/9]
  Please input thread number for FunASR decoder.
  Default: 1
  Setting the number of decoder thread: 

  The number of decoder threads is 1
  The number of IO threads is 1
```

##### 所有设置参数展示及确认

展示前面6步设置的参数，确认则将所有参数存储到/var/funasr/config，并开始启动Docker，否则提示用户进行重新设置。

```text

[6/9]
  Show parameters of FunASR server setting and confirm to run ...

  The current Docker image is                                    : registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-latest
  The model is downloaded or stored to this directory in local   : /root/models
  The model will be automatically downloaded to the directory    : /workspace/models
  The ASR model_id used                                          : damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx
  The ASR model directory corresponds to the directory in Docker : /workspace/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx
  The VAD model_id used                                          : damo/speech_fsmn_vad_zh-cn-16k-common-onnx
  The VAD model directory corresponds to the directory in Docker : /workspace/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx
  The PUNC model_id used                                         : damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx
  The PUNC model directory corresponds to the directory in Docker: /workspace/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx

  The path in the docker of the FunASR service executor          : /workspace/FunASR/funasr/runtime/websocket/build/bin/funasr-wss-server
  Set the host port used for use by the FunASR service           : 10095
  Set the docker port used by the FunASR service                 : 10095
  Set the number of threads used for decoding the FunASR service : 1
  Set the number of threads used for IO the FunASR service       : 1

  Please input [Y/n] to confirm the parameters.
  [y] Verify that these parameters are correct and that the service will run.
  [n] The parameters set are incorrect, it will be rolled out, please rerun.
  read confirmation[Y/n]: 

  Will run FunASR server later ...
  Parameters are stored in the file /var/funasr/config
```

##### 检查Docker服务

检查当前宿主机是否安装了Docker服务，若未安装，则安装Docker并启动。

```text
[7/9]
  Start install docker for ubuntu 
  Get docker installer: curl -fsSL https://test.docker.com -o test-docker.sh
  Get docker run: sudo sh test-docker.sh
# Executing docker install script, commit: c2de0811708b6d9015ed1a2c80f02c9b70c8ce7b
+ sh -c apt-get update -qq >/dev/null
+ sh -c DEBIAN_FRONTEND=noninteractive apt-get install -y -qq apt-transport-https ca-certificates curl >/dev/null
+ sh -c install -m 0755 -d /etc/apt/keyrings
+ sh -c curl -fsSL "https://download.docker.com/linux/ubuntu/gpg" | gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
+ sh -c chmod a+r /etc/apt/keyrings/docker.gpg
+ sh -c echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu focal test" > /etc/apt/sources.list.d/docker.list
+ sh -c apt-get update -qq >/dev/null
+ sh -c DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin docker-ce-rootless-extras docker-buildx-plugin >/dev/null
+ sh -c docker version
Client: Docker Engine - Community
 Version:           24.0.2

 ...
 ...

   Docker install success, start docker server.
```

##### 下载FunASR Docker镜像

下载并更新step1.1中选择的FunASR Docker镜像。

```text
[8/9]
  Pull docker image(registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-latest)...
funasr-runtime-cpu-0.0.1: Pulling from funasr_repo/funasr
7608715873ec: Pull complete 
3e1014c56f38: Pull complete 

 ...
 ...
```

##### 启动FunASR Docker

启动FunASR Docker，等待step1.2选择的模型下载完成并启动FunASR服务。

```text
[9/9]
  Construct command and run docker ...
943d8f02b4e5011b71953a0f6c1c1b9bc5aff63e5a96e7406c83e80943b23474

  Loading models:
    [ASR ][Done       ][==================================================][100%][1.10MB/s][v1.2.1]
    [VAD ][Done       ][==================================================][100%][7.26MB/s][v1.2.0]
    [PUNC][Done       ][==================================================][100%][ 474kB/s][v1.1.7]
  The service has been started.
  If you want to see an example of how to use the client, you can run sudo bash funasr-runtime-deploy.sh -c .
```

#### 启动已经部署过的FunASR服务
一键部署后若出现重启电脑等关闭Docker的动作，可通过如下命令直接启动FunASR服务，启动配置为上次一键部署的设置。

```shell
sudo bash funasr-runtime-deploy.sh start
```

#### 关闭FunASR服务

```shell
sudo bash funasr-runtime-deploy.sh stop
```

#### 重启FunASR服务

根据上次一键部署的设置重启启动FunASR服务。
```shell
sudo bash funasr-runtime-deploy.sh restart
```

#### 替换模型并重启FunASR服务

替换正在使用的模型，并重新启动FunASR服务。模型需为ModelScope中的ASR/VAD/PUNC模型，或者从ModelScope中模型finetune后的模型。

```shell
sudo bash funasr-runtime-deploy.sh update model <model ID>

e.g
sudo bash funasr-runtime-deploy.sh update model damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```

### 测试与使用离线文件转写服务

在服务器上完成FunASR服务部署以后，可以通过如下的步骤来测试和使用离线文件转写服务。目前分别支持Python、C++、Java版本client的的命令行运行，以及可在浏览器可直接体验的html网页版本，更多语言client支持参考文档【FunASR高阶开发指南】。
funasr-runtime-deploy.sh运行结束后，可通过命令以交互的形式自动下载测试样例samples到当前目录的funasr_samples中，并设置参数运行：

```shell
sudo bash funasr-runtime-deploy.sh client
```

可选择提供的Python和Linux C++范例程序，以Python范例为例：

```text
Will download sample tools for the client to show how speech recognition works.
  Please select the client you want to run.
    1) Python
    2) Linux_Cpp
  Enter your choice: 1

  Please enter the IP of server, default(127.0.0.1): 
  Please enter the port of server, default(10095): 
  Please enter the audio path, default(/root/funasr_samples/audio/asr_example.wav): 

  Run pip3 install click>=8.0.4
Looking in indexes: http://mirrors.cloud.aliyuncs.com/pypi/simple/
Requirement already satisfied: click>=8.0.4 in /usr/local/lib/python3.8/dist-packages (8.1.3)

  Run pip3 install -r /root/funasr_samples/python/requirements_client.txt
Looking in indexes: http://mirrors.cloud.aliyuncs.com/pypi/simple/
Requirement already satisfied: websockets in /usr/local/lib/python3.8/dist-packages (from -r /root/funasr_samples/python/requirements_client.txt (line 1)) (11.0.3)

  Run python3 /root/funasr_samples/python/wss_client_asr.py --host 127.0.0.1 --port 10095 --mode offline --audio_in /root/funasr_samples/audio/asr_example.wav --send_without_sleep --output_dir ./funasr_samples/python

  ...
  ...

  pid0_0: 欢迎大家来体验达摩院推出的语音识别模型。
Exception: sent 1000 (OK); then received 1000 (OK)
end

  If failed, you can try (python3 /root/funasr_samples/python/wss_client_asr.py --host 127.0.0.1 --port 10095 --mode offline --audio_in /root/funasr_samples/audio/asr_example.wav --send_without_sleep --output_dir ./funasr_samples/python) in your Shell.

```

#### python-client
若想直接运行client进行测试，可参考如下简易说明，以python版本为例：

```shell
python3 wss_client_asr.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "../audio/asr_example.wav" --send_without_sleep --output_dir "./results"
```

命令参数说明：
```text
--host 为FunASR runtime-SDK服务部署机器ip，默认为本机ip（127.0.0.1），如果client与服务不在同一台服务器，需要改为部署机器ip
--port 10095 部署端口号
--mode offline表示离线文件转写
--audio_in 需要进行转写的音频文件，支持文件路径，文件列表wav.scp
--output_dir 识别结果保存路径
```

#### cpp-client

```shell
export LD_LIBRARY_PATH=/root/funasr_samples/cpp/libs:$LD_LIBRARY_PATH
/root/funasr_samples/cpp/funasr-wss-client --server-ip 127.0.0.1 --port 10095 --wav-path /root/funasr_samples/audio/asr_example.wav
```

命令参数说明：

```text
--server-ip 为FunASR runtime-SDK服务部署机器ip，默认为本机ip（127.0.0.1），如果client与服务不在同一台服务器，需要改为部署机器ip
--port 10095 部署端口号
--wav-path 需要进行转写的音频文件，支持文件路径
```

### 视频demo

[点击此处]()















