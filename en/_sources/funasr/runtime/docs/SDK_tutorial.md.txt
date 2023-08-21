([简体中文](./SDK_tutorial_zh.md)|English)

# Highlights
**FunASR offline file transcription service has been upgraded to version 2.0, which integrates ffmpeg to support various audio and video inputs. It also supports custom hotword models and timestamp models. Feel free to deploy and experience it![Quick Start](#Quick Start)**

# FunASR Offline File Transcription Service

FunASR provides an offline file transcription service that can be easily deployed on a local or cloud server. The core is the FunASR open-source runtime-SDK. It integrates various capabilities such as speech endpoint detection (VAD) and Paraformer-large speech recognition (ASR) and punctuation restoration (PUNC) released by the speech laboratory of the Damo Academy in the Modelscope community. It has a complete speech recognition chain and can recognize audio or video of tens of hours into punctuated text. Moreover, it supports transcription for hundreds of simultaneous requests.

## Server Configuration

Users can choose appropriate server configurations based on their business needs. The recommended configurations are:
- Configuration 1: (X86, computing-type) 4-core vCPU, 8GB memory, and a single machine can support about 32 requests.
- Configuration 2: (X86, computing-type) 16-core vCPU, 32GB memory, and a single machine can support about 64 requests.
- Configuration 3: (X86, computing-type) 64-core vCPU, 128GB memory, and a single machine can support about 200 requests. 

Detailed performance [report](./benchmark_onnx_cpp.md)

Cloud service providers offer a 3-month free trial for new users. Application tutorial ([docs](./aliyun_server_tutorial.md)).

## Quick Start

### Server Startup

`Note`: The one-click deployment tool process includes installing Docker, downloading Docker images, and starting the service. If the user wants to start from the FunASR Docker image, please refer to the development guide ([docs](./SDK_advanced_guide_offline.md).

Download the deployment tool `funasr-runtime-deploy-offline-cpu-zh.sh`

```shell
curl -O https://raw.githubusercontent.com/alibaba-damo-academy/FunASR/main/funasr/runtime/deploy_tools/funasr-runtime-deploy-offline-cpu-en.sh;
# If there is a network problem, users in mainland China can use the following command:
# curl -O https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/shell/funasr-runtime-deploy-offline-cpu-en.sh;
```

Execute the deployment tool and press the Enter key at the prompt to complete the installation and deployment of the server. Currently, the convenient deployment tool only supports Linux environments. For other environments, please refer to the development guide ([docs](./SDK_advanced_guide_offline.md)).
```shell
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh install --workspace /root/funasr-runtime-resources
```
Note: If you need to deploy the timestamp model or hotword model, select the corresponding model in step 2 of the installation and deployment process, where 1 is the paraformer-large model, 2 is the paraformer-large timestamp model, and 3 is the paraformer-large hotword model.

### Client Testing and Usage

After running the above installation instructions, the client testing tool directory samples will be downloaded in the default installation directory /root/funasr-runtime-resources ([download click](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz)).
We take the Python language client as an example to explain that it supports multiple audio format inputs (such as .wav, .pcm, .mp3, etc.), video inputs (.mp4, etc.), and multiple file list wav.scp inputs. For other client versions, please refer to the [documentation](#Detailed-Description-of-Client-Usage).

```shell
python3 funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "../audio/asr_example.wav"
```

## Detailed Description of Client Usage

After completing the FunASR runtime-SDK service deployment on the server, you can test and use the offline file transcription service through the following steps. Currently, the following programming language client versions are supported:

- [Python](#python-client)
- [CPP](#cpp-client)
- [html](#html-client)
- [java](#java-client)

For more client version support, please refer to the [development guide](./SDK_advanced_guide_offline_zh.md).

### python-client
If you want to run the client directly for testing, you can refer to the following simple instructions, using the Python version as an example:

```shell
python3 funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "../audio/asr_example.wav"
```

Command parameter instructions:
```text
--host is the IP address of the FunASR runtime-SDK service deployment machine, which defaults to the local IP address (127.0.0.1). If the client and the service are not on the same server, it needs to be changed to the deployment machine IP address.
--port 10095 deployment port number
--mode offline represents offline file transcription
--audio_in is the audio file that needs to be transcribed, supporting file paths and file list wav.scp
--thread_num sets the number of concurrent sending threads, default is 1
--ssl sets whether to enable SSL certificate verification, default is 1 to enable, and 0 to disable
--hotword If am is hotword model, setting hotword: *.txt(one hotword perline) or hotwords seperate by space (could be: 阿里巴巴 达摩院)
```

### cpp-client

After entering the samples/cpp directory, you can test it with CPP. The command is as follows:
```shell
./funasr-wss-client --server-ip 127.0.0.1 --port 10095 --wav-path ../audio/asr_example.wav
```

Command parameter description:
```text
--server-ip specifies the IP address of the machine where the FunASR runtime-SDK service is deployed. The default value is the local IP address (127.0.0.1). If the client and the service are not on the same server, the IP address needs to be changed to the IP address of the deployment machine.
--port specifies the deployment port number as 10095.
--wav-path specifies the audio file to be transcribed, and supports file paths.
--thread_num sets the number of concurrent send threads, with a default value of 1.
--ssl sets whether to enable SSL certificate verification, with a default value of 1 for enabling and 0 for disabling.
--hotword If am is hotword model, setting hotword: *.txt(one hotword perline) or hotwords seperate by space (could be: 阿里巴巴 达摩院)
```

### html-client

To experience it directly, open `html/static/index.html` in your browser. You will see the following page, which supports microphone input and file upload.
<img src="images/html.png"  width="900"/>

### java-client

```shell
FunasrWsClient --host localhost --port 10095 --audio_in ./asr_example.wav --mode offline
```
For more details, please refer to the [docs](../java/readme.md)

## Server Usage Details

### Start the deployed FunASR service

If you have restarted the computer or shut down Docker after one-click deployment, you can start the FunASR service directly with the following command. The startup configuration is the same as the last one-click deployment.

```shell
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh start
```

### Stop the FunASR service

```shell
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh stop
```

### Release the FunASR service

Release the deployed FunASR service.
```shell
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh remove
```

### Restart the FunASR service

Restart the FunASR service with the same configuration as the last one-click deployment.
```shell
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh restart
```

### Replace the model and restart the FunASR service

Replace the currently used model, and restart the FunASR service. The model must be an ASR/VAD/PUNC model in ModelScope, or a finetuned model from ModelScope.

```shell
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh update [--asr_model | --vad_model | --punc_model] <model_id or local model path>

e.g
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh update --asr_model damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```

### Update parameters and restart the FunASR service

Update the configured parameters and restart the FunASR service to take effect. The parameters that can be updated include the host and Docker port numbers, as well as the number of inference and IO threads.

```shell
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh update [--host_port | --docker_port] <port number>
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh update [--decode_thread_num | --io_thread_num] <the number of threads>
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh update [--workspace] <workspace in local>
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh update [--ssl] <0: close SSL; 1: open SSL, default:1>

e.g
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh update --decode_thread_num 32
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh update --workspace /root/funasr-runtime-resources
```

### Set SSL

SSL verification is enabled by default. If you need to disable it, you can set it when starting.
```shell
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update --ssl 0
```



## Contact Us

If you encounter any problems during use, please join our user group for feedback.


|                                DingDing Group                                |                             Wechat                             |
|:----------------------------------------------------------------------------:|:--------------------------------------------------------------:|
| <div align="left"><img src="../../../docs/images/dingding.jpg" width="250"/> | <img src="../../../docs/images/wechat.png" width="232"/></div> |

















