# FunASR File Transcription Service Convenient Deployment Tutorial

FunASR provides offline file transcription services that can be conveniently deployed on local or cloud servers. The core of the service is based on the open-source runtime-SDK of FunASR. It integrates various related capabilities, such as voice endpoint detection (VAD) and Paraformer-large speech recognition (ASR), as well as punctuation recovery (PUNC), which have been open-sourced by the speech laboratory of DAMO Academy on the Modelscope community. With these capabilities, the service can transcribe audio accurately and efficiently under high concurrency.

## Installation and Start Service

Environment Preparation and Configuration（[docs](./aliyun_server_tutorial.md)）

### Downloading Tools and Deployment

Run the following command to perform a one-click deployment of the FunASR runtime-SDK service. Follow the prompts to complete the deployment and running of the service. Currently, only Linux environments are supported, and for other environments, please refer to the Advanced SDK Development Guide ([docs](./SDK_advanced_guide_offline.md)). 

[//]: # (Due to network restrictions, the download of the funasr-runtime-deploy.sh one-click deployment tool may not proceed smoothly. If the tool has not been downloaded and entered into the one-click deployment tool after several seconds, please terminate it with Ctrl + C and run the following command again.)

```shell
curl -O https://raw.githubusercontent.com/alibaba-damo-academy/FunASR-APP/main/TransAudio/funasr-runtime-deploy.sh; sudo bash funasr-runtime-deploy.sh install
# For the users in China, you could install with the command:
# curl -O https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/shell/funasr-runtime-deploy.sh; sudo bash funasr-runtime-deploy.sh install
```

#### Details of Configuration

##### Choosing FunASR Docker Image

We recommend selecting the "latest" tag to use our latest image, but you can also choose from our historical versions.

```text
[1/9]
  Please choose the Docker image.
    1) registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-latest
    2) registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.1.0
  Enter your choice: 1
  You have chosen the Docker image: registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-latest
```

##### Choosing ASR/VAD/PUNC Models

You can choose a model from ModelScope by name, or fill in the name of a model in ModelScope as <model_name>. The model will be automatically downloaded during Docker runtime. You can also select <model_path> to fill in the local model path on the host machine.

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

##### Enter the executable path of the FunASR service on the host machine

Enter the host path of the executable of the FunASR service. It will be automatically mounted and run in Docker at runtime. If left blank, the default path in Docker will be set to /workspace/FunASR/funasr/runtime/websocket/build/bin/funasr-wss-server.

```text
[3/9]
  Please enter the path to the excutor of the FunASR service on the localhost.
  If not set, the default /workspace/FunASR/funasr/runtime/websocket/build/bin/funasr-wss-server in Docker is used.
  Setting the path to the excutor of the FunASR service on the localhost: 
  Corresponding, the path of FunASR in Docker is /workspace/FunASR/funasr/runtime/websocket/build/bin/funasr-wss-server
```

##### Setting the port on the host machine for FunASR

Setting the port on the host machine for Docker. The default port is 10095. Please ensure that this port is available.

```text
[4/9]
  Please input the opened port in the host used for FunASR server.
  Default: 10095
  Setting the opened host port [1-65535]: 
  The port of the host is 10095
  The port in Docker for FunASR server is 10095
```


##### Setting the number of inference threads for the FunASR service

Setting the number of inference threads for the FunASR service. The default value is the number of cores on the host machine. The number of I/O threads for the service will also be automatically set to one-quarter of the number of inference threads.

```text
[5/9]
  Please input thread number for FunASR decoder.
  Default: 1
  Setting the number of decoder thread: 

  The number of decoder threads is 1
  The number of IO threads is 1
```

##### Displaying all set parameters for confirmation

Displaying the parameters set in the previous 6 steps. Confirming will save all parameters to /var/funasr/config and start Docker. Otherwise, users will be prompted to reset the parameters.

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

##### Checking the Docker service

Checking if Docker service is installed on the host machine. If not installed, installing and starting Docker

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

##### Downloading the FunASR Docker image

Downloading and updating the FunASR Docker image selected in step 1.1

```text
[8/9]
  Pull docker image(registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-latest)...
funasr-runtime-cpu-0.0.1: Pulling from funasr_repo/funasr
7608715873ec: Pull complete 
3e1014c56f38: Pull complete 

 ...
 ...
```

##### Starting the FunASR Docker

Starting the FunASR Docker and waiting for the model selected in step 1.2 to finish downloading and start the FunASR service

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

#### Starting the deployed FunASR service

If the computer is restarted or Docker is closed after one-click deployment, the following command can be used to start the FunASR service directly with the settings from the last one-click deployment.

```shell
sudo bash funasr-runtime-deploy.sh start
```

#### Shutting down the FunASR service

```shell
sudo bash funasr-runtime-deploy.sh stop
```

#### Restarting the FunASR service

Restarting the FunASR service with the settings from the last one-click deployment

```shell
sudo bash funasr-runtime-deploy.sh restart
```

#### Replacing the model and restarting the FunASR service

Replacing the currently used model and restarting the FunASR service. The model must be an ASR/VAD/PUNC model from ModelScope.

```shell
sudo bash scripts/funasr-runtime-deploy.sh update model <model ID in ModelScope>

e.g
sudo bash scripts/funasr-runtime-deploy.sh update model damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```

### How to test and use the offline file transcription service

After completing the FunASR service deployment on the server, you can test and use the offline file transcription service by following these steps. Currently, command line running is supported for Python, C++, and Java client versions, as well as an HTML web page version that can be directly experienced in the browser. For more client language support, please refer to the "FunASR Advanced Development Guide" documentation.
After the funasr-runtime-deploy.sh script finishes running, you can use the following command to automatically download the test samples to the funasr_samples directory in the current directory and run the program with the set parameters in an interactive manner:

```shell
sudo bash funasr-runtime-deploy.sh client
```

You can choose from the provided Python and Linux C++ sample programs. Taking the Python sample as an example:

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

If you want to directly run the client for testing, you can refer to the following simple instructions, taking the Python version as an example:
```shell
python3 wss_client_asr.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "../audio/asr_example.wav" --send_without_sleep --output_dir "./results"
```

Command parameter instructions: 

```text
--host: The IP address of the machine where the FunASR runtime-SDK service is deployed. The default is the local IP address (127.0.0.1). If the client and service are not on the same server, the IP address should be changed to that of the deployment machine.
--port 10095: The deployment port number.
--mode offline: Indicates offline file transcription.
--audio_in: The audio file(s) to be transcribed, which can be a file path or a file list (wav.scp).
--output_dir: The path to save the recognition results.
```

#### cpp-client

```shell
export LD_LIBRARY_PATH=/root/funasr_samples/cpp/libs:$LD_LIBRARY_PATH
/root/funasr_samples/cpp/funasr-wss-client --server-ip 127.0.0.1 --port 10095 --wav-path /root/funasr_samples/audio/asr_example.wav
```

Command parameter instructions: 

```text
--server-ip: The IP address of the machine where the FunASR runtime-SDK service is deployed. The default is the local IP address (127.0.0.1). If the client and service are not on the same server, the IP address should be changed to that of the deployment machine.
--port 10095: The deployment port number.
--wav-path: The audio file(s) to be transcribed, which can be a file path.
```

### Video demo

[demo]()















