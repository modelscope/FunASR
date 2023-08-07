([简体中文](./SDK_tutorial_online_zh.md)|English)

# FunASR-realtime-transcribe service

FunASR offers a real-time speech-to-text service that can be easily deployed locally or on cloud servers. The service integrates various capabilities developed by the speech laboratory of DAMO Academy on the ModelScope, including voice activity detection (VAD), Paraformer-large non-streaming automatic speech recognition (ASR), Paraformer-large streaming ASR, and punctuation prediction (PUNC). The software package supports realtime speech-to-text service as well as high-precision transcription text correction at the end of each sentence and outputs text with punctuation.

## Server Configurations

Users can choose appropriate server configurations based on their business needs. The recommended configurations are:
- Configuration 1: (X86, computing-type) 4-core vCPU, 8GB memory, and a single machine can support about 32 requests.
- Configuration 2: (X86, computing-type) 16-core vCPU, 32GB memory, and a single machine can support about 64 requests.
- Configuration 3: (X86, computing-type) 64-core vCPU, 128GB memory, and a single machine can support about 200 requests. 

Detailed performance [report](./benchmark_onnx_cpp.md)

Cloud service providers offer a 3-month free trial for new users. Application tutorial ([docs](./aliyun_server_tutorial.md)).

## Quick Start

### Server Startup

`Note`: The one-click deployment tool process includes installing Docker, downloading Docker images, and starting the service. If the user wants to start from the FunASR Docker image, please refer to the development guide ([docs](./SDK_advanced_guide_online.md).

Download the deployment tool `funasr-runtime-deploy-online-cpu-zh.sh`

```shell
curl -O https://raw.githubusercontent.com/alibaba-damo-academy/FunASR/main/funasr/runtime/deploy_tools/funasr-runtime-deploy-online-cpu-en.sh;
# If there is a network problem, users in mainland China can use the following command:
# curl -O https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/shell/funasr-runtime-deploy-online-cpu-en.sh;
```

Execute the deployment tool and press the Enter key at the prompt to complete the installation and deployment of the server. Currently, the convenient deployment tool only supports Linux environments. For other environments, please refer to the development guide ([docs](./SDK_advanced_guide_offline.md)).
```shell
sudo bash funasr-runtime-deploy-online-cpu-zh.sh install --workspace ./funasr-runtime-resources
```

### Client Testing and Usage

After running the above installation instructions, the client testing tool directory samples will be downloaded in the default installation directory ./funasr-runtime-resources ([download click](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz)).
We take the Python language client as an example to explain that it supports multiple audio format inputs (such as .wav, .pcm, .mp3, etc.), video inputs (.mp4, etc.), and multiple file list wav.scp inputs. For other client versions, please refer to the [documentation](#Detailed-Description-of-Client-Usage).

```shell
python3 funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode 2pass
```

## Detailed Description of Client Usage

After completing the FunASR runtime-SDK service deployment on the server, you can test and use the offline file transcription service through the following steps. Currently, the following programming language client versions are supported:

- [Python](#python-client)
- [CPP](#cpp-client)
- [html](#html-client)
- [java](#java-client)

For more client version support, please refer to the [websocket_protocol](./websocket_protocol_zh.md).

### python-client
If you want to run the client directly for testing, you can refer to the following simple instructions, using the Python version as an example:

```shell
python3 funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "../audio/asr_example.wav"
```

Command parameter instructions:
```text
--host is the IP address of the FunASR runtime-SDK service deployment machine, which defaults to the local IP address (127.0.0.1). If the client and the service are not on the same server, it needs to be changed to the deployment machine IP address.
--port 10095 deployment port number
--mode: `offline` indicates that the inference mode is one-sentence recognition; `online` indicates that the inference mode is real-time speech recognition; `2pass` indicates real-time speech recognition, and offline models are used for error correction at the end of each sentence.
--chunk_size: indicates the latency configuration of the streaming model. [5,10,5] indicates that the current audio is 600ms, with a lookback of 300ms and a lookahead of 300ms.
--audio_in is the audio file that needs to be transcribed, supporting file paths and file list wav.scp
--thread_num sets the number of concurrent sending threads, default is 1
--ssl sets whether to enable SSL certificate verification, default is 1 to enable, and 0 to disable
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
--mode: `offline` indicates that the inference mode is one-sentence recognition; `online` indicates that the inference mode is real-time speech recognition; `2pass` indicates real-time speech recognition, and offline models are used for error correction at the end of each sentence.
--chunk_size: indicates the latency configuration of the streaming model. [5,10,5] indicates that the current audio is 600ms, with a lookback of 300ms and a lookahead of 300ms.
--wav-path specifies the audio file to be transcribed, and supports file paths.
--thread_num sets the number of concurrent send threads, with a default value of 1.
--ssl sets whether to enable SSL certificate verification, with a default value of 1 for enabling and 0 for disabling.
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
sudo bash funasr-runtime-deploy-online-cpu-zh.sh start
```

### Set SSL

SSL verification is enabled by default. If you need to disable it, you can set it when starting.
```shell
sudo bash funasr-runtime-deploy-online-cpu-zh.sh --ssl 0
```

### Stop the FunASR service

```shell
sudo bash funasr-runtime-deploy-online-cpu-zh.sh stop
```

### Release the FunASR service

Release the deployed FunASR service.
```shell
sudo bash funasr-runtime-deploy-online-cpu-zh.sh remove
```

### Restart the FunASR service

Restart the FunASR service with the same configuration as the last one-click deployment.
```shell
sudo bash funasr-runtime-deploy-online-cpu-zh.sh restart
```

### Replace the model and restart the FunASR service

Replace the currently used model, and restart the FunASR service. The model must be an ASR/VAD/PUNC model in ModelScope, or a finetuned model from ModelScope.

```shell
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update [--asr_model | --vad_model | --punc_model] <model_id or local model path>

e.g
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update --asr_model damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```

### Update parameters and restart the FunASR service

Update the configured parameters and restart the FunASR service to take effect. The parameters that can be updated include the host and Docker port numbers, as well as the number of inference and IO threads.

```shell
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update [--host_port | --docker_port] <port number>
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update [--decode_thread_num | --io_thread_num] <the number of threads>
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update [--workspace] <workspace in local>
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update [--ssl] <0: close SSL; 1: open SSL, default:1>

e.g
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update --decode_thread_num 32
sudo bash funasr-runtime-deploy-online-cpu-zh.sh update --workspace ./funasr-runtime-resources
```



## Contact Us

If you encounter any problems during use, please join our user group for feedback.


|                                DingDing Group                                |                             Wechat                             |
|:----------------------------------------------------------------------------:|:--------------------------------------------------------------:|
| <div align="left"><img src="../../../docs/images/dingding.jpg" width="250"/> | <img src="../../../docs/images/wechat.png" width="232"/></div> |



















