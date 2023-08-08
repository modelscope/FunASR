 # Advanced Development Guide (File transcription service)
 
FunASR provides a Chinese online transcription service that can be deployed locally or on a cloud server with just one click. The core of the service is the FunASR runtime SDK, which has been open-sourced. FunASR-runtime combines various capabilities such as speech endpoint detection (VAD), offline large-scale speech recognition (ASR) using Paraformer-large, online large-scale speech recognition (ASR) using Paraformer-large, and punctuation detection (PUNC), which have all been open-sourced by the speech laboratory of DAMO Academy on the Modelscope community. 
This document serves as a development guide for the FunASR online transcription service. If you wish to quickly experience the online transcription service, please refer to the one-click deployment example for the FunASR online transcription service ([docs](./SDK_tutorial_online.md)).

## Installation of Docker

The following steps are for manually installing Docker and Docker images. If your Docker image has already been launched, you can ignore this step.

### Installation of Docker environment

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

More details could ref to [docs](https://alibaba-damo-academy.github.io/FunASR/en/installation/docker.html)

### Starting Docker

```shell
sudo systemctl start docker
```

### Pulling and launching images

Use the following command to pull and launch the Docker image for the FunASR runtime-SDK:

```shell
sudo docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.0

sudo docker run -p 10095:10095 -it --privileged=true -v /root:/workspace/models registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.0
```

Introduction to command parameters: 
```text
-p <host port>:<mapped docker port>: In the example, host machine (ECS) port 10095 is mapped to port 10095 in the Docker container. Make sure that port 10095 is open in the ECS security rules.

-v <host path>:<mounted Docker path>: In the example, the host machine path /root is mounted to the Docker path /workspace/models.

```


## Starting the server

Use the flollowing script to start the server ：
```shell
cd FunASR/funasr/runtime
./run_server_2pass.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx  \
  --online-model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx  \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx
```

More details about the script run_server_2pass.sh:

The FunASR-wss-server-2pass supports downloading models from Modelscope. You can set the model download address (--download-model-dir, default is /workspace/models) and the model ID (--model-dir, --vad-dir, --punc-dir). Here is an example:

```shell
cd /workspace/FunASR/funasr/runtime/websocket/build/bin
./funasr-wss-server-2pass  \
  --download-model-dir /workspace/models \
  --decoder-thread-num 32 \
  --io-thread-num  8 \
  --port 10095 \
 ```

Introduction to command parameters: 

```text
--download-model-dir: Model download address, download models from Modelscope by setting the model ID.
--model-dir: Modelscope model ID.
--quantize: True for quantized ASR model, False for non-quantized ASR model. Default is True.
--vad-dir: Modelscope model ID.
--vad-quant: True for quantized VAD model, False for non-quantized VAD model. Default is True.
--punc-dir: Modelscope model ID.
--punc-quant: True for quantized PUNC model, False for non-quantized PUNC model. Default is True.
--port: Port number that the server listens on. Default is 10095.
--decoder-thread-num: Number of inference threads that the server starts. Default is 8.
--io-thread-num: Number of IO threads that the server starts. Default is 1.
--certfile <string>: SSL certificate file. Default is ../../../ssl_key/server.crt.
--keyfile <string>: SSL key file. Default is ../../../ssl_key/server.key.
```

After executing the above command, the real-time speech recognition service will be started. If the model is specified as the model ID in ModelScope, the following model will be automatically downloaded from ModelScope:
[FSMN-VAD](https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx/summary)，
[Paraformer-lagre-online](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx/summary )
[Paraformer-lagre-offline](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx/summary)
[CT-Transformer-online](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx/summary)

### Exporting models from finetuned resources

If you want to deploy a finetuned model, you can follow these steps:
Rename the model you want to deploy after finetuning (for example, 10epoch.pb) to model.pb, and replace the original model.pb in Modelscope with this one. If the path of the replaced model is /path/to/finetune/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch, set the path to model-dir.


## Starting the client

After completing the deployment of FunASR online transcription service on the server, you can test and use the service by following these steps. Currently, FunASR-bin supports multiple ways to start the client. The following are command-line examples based on python-client, c++-client, and custom client Websocket communication protocol: 

### python-client
```shell
python funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode 2pass --audio_in "./data/wav.scp" --send_without_sleep --output_dir "./results"
```

Introduction to command parameters:

```text
--host: the IP address of the server. It can be set to 127.0.0.1 for local testing.
--port: the port number of the server listener.
--audio_in: the audio input. Input can be a path to a wav file or a wav.scp file (a Kaldi-formatted wav list in which each line includes a wav_id followed by a tab and a wav_path).
--output_dir: the path to the recognition result output.
--ssl: whether to use SSL encryption. The default is to use SSL.
--mode: offline, online, 2pass
```

### c++-client
```shell
. /funasr-wss-client-2pass --server-ip 127.0.0.1 --port 10095 --wav-path test.wav --thread-num 1 --is-ssl 1
```

Introduction to command parameters:

```text
--server-ip: the IP address of the server. It can be set to 127.0.0.1 for local testing.
--port: the port number of the server listener.
--wav-path: the audio input. Input can be a path to a wav file or a wav.scp file (a Kaldi-formatted wav list in which each line includes a wav_id followed by a tab and a wav_path).
--is-ssl: whether to use SSL encryption. The default is to use SSL.
--mode: offline, online, 2pass
--thread-num 1 

```
