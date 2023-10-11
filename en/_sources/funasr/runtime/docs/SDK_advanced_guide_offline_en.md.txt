 # Advanced Development Guide (File transcription service)
 
FunASR provides a English offline file transcription service that can be deployed locally or on a cloud server with just one click. The core of the service is the FunASR runtime SDK, which has been open-sourced. FunASR-runtime combines various capabilities such as speech endpoint detection (VAD), large-scale speech recognition (ASR) using Paraformer-large, and punctuation detection (PUNC), which have all been open-sourced by the speech laboratory of DAMO Academy on the Modelscope community. This enables accurate and efficient high-concurrency transcription of audio files.

This document serves as a development guide for the FunASR offline file transcription service. If you wish to quickly experience the offline file transcription service, please refer to the one-click deployment example for the FunASR offline file transcription service ([docs](./SDK_tutorial.md)).

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
sudo docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-en-cpu-0.1.0

sudo docker run -p 10095:10095 -it --privileged=true -v /root:/workspace/models registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-en-cpu-0.1.0
```

Introduction to command parameters: 
```text
-p <host port>:<mapped docker port>: In the example, host machine (ECS) port 10095 is mapped to port 10095 in the Docker container. Make sure that port 10095 is open in the ECS security rules.

-v <host path>:<mounted Docker path>: In the example, the host machine path /root is mounted to the Docker path /workspace/models.

```


## Starting the server

Use the flollowing script to start the server ：
```shell
nohup bash run_server.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large_asr_nat-en-16k-common-vocab10020-onnx  \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx > log.out 2>&1 &

# If you want to close ssl，please add：--certfile 0

```

### More details about the script run_server.sh:

The funasr-wss-server supports downloading models from Modelscope. You can set the model download address (--download-model-dir, default is /workspace/models) and the model ID (--model-dir, --vad-dir, --punc-dir). Here is an example:

```shell
cd /workspace/FunASR/funasr/runtime
nohup bash run_server.sh \
  --download-model-dir /workspace/models \
  --model-dir damo/speech_paraformer-large_asr_nat-en-16k-common-vocab10020-onnx \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx \
  --decoder-thread-num 32 \
  --io-thread-num  8 \
  --port 10095 \
  --certfile  ../../../ssl_key/server.crt \
  --keyfile ../../../ssl_key/server.key
 ```

Introduction to run_server.sh parameters: 

```text
--download-model-dir: Model download address, download models from Modelscope by setting the model ID.
--model-dir: Modelscope model ID.
--quantize: True for quantized ASR model, False for non-quantized ASR model. Default is True.
--vad-dir: Modelscope model ID.
--vad-quant: True for quantized VAD model, False for non-quantized VAD model. Default is True.
--punc-dir: Modelscope model ID.
--punc-quant: True for quantized PUNC model, False for non-quantized PUNC model. Default is True.
--itn-dir modelscope model ID
--port: Port number that the server listens on. Default is 10095.
--decoder-thread-num: Number of inference threads that the server starts. Default is 8.
--io-thread-num: Number of IO threads that the server starts. Default is 1.
--certfile <string>: SSL certificate file. Default is ../../../ssl_key/server.crt. If you want to close ssl，set 0
--keyfile <string>: SSL key file. Default is ../../../ssl_key/server.key. 
```

### Shutting Down the FunASR Service
```text
# Check the PID of the funasr-wss-server process
ps -x | grep funasr-wss-server
kill -9 PID
```

### Modifying Models and Other Parameters
To replace the currently used model or other parameters, you need to first shut down the FunASR service, make the necessary modifications to the parameters you want to replace, and then restart the FunASR service. The model should be either an ASR/VAD/PUNC model from ModelScope or a fine-tuned model obtained from ModelScope.
```text
# For example, to replace the ASR model with damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx, use the following parameter setting --model-dir
    --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx 
# Set the port number using --port
    --port <port number>
# Set the number of inference threads the server will start using --decoder-thread-num
    --decoder-thread-num <decoder thread num>
# Set the number of IO threads the server will start using --io-thread-num
    --io-thread-num <io thread num>
# Disable SSL certificate
    --certfile 0
```

After executing the above command, the real-time speech transcription service will be started. If the model is specified as a ModelScope model id, the following models will be automatically downloaded from ModelScope:
[FSMN-VAD](https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx/summary),
[Paraformer-lagre](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-en-16k-common-vocab10020-onnx/summary),
[CT-Transformer](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx/summary)

If you wish to deploy your fine-tuned model (e.g., 10epoch.pb), you need to manually rename the model to model.pb and replace the original model.pb in ModelScope. Then, specify the path as `model_dir`.

## Starting the client

After completing the deployment of FunASR offline file transcription service on the server, you can test and use the service by following these steps. Currently, FunASR-bin supports multiple ways to start the client. The following are command-line examples based on python-client, c++-client, and custom client Websocket communication protocol: 

### python-client
```shell
python funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "./data/wav.scp" --send_without_sleep --output_dir "./results"
```

Introduction to command parameters:

```text
--host: the IP address of the server. It can be set to 127.0.0.1 for local testing.
--port: the port number of the server listener.
--audio_in: the audio input. Input can be a path to a wav file or a wav.scp file (a Kaldi-formatted wav list in which each line includes a wav_id followed by a tab and a wav_path).
--output_dir: the path to the recognition result output.
--ssl: whether to use SSL encryption. The default is to use SSL.
--mode: offline mode.
--hotword: If am is hotword model, setting hotword: *.txt(one hotword perline) or hotwords seperate by space (could be: 阿里巴巴 达摩院)
--use_itn: whether to use itn, the default value is 1 for enabling and 0 for disabling.
```

### c++-client
```shell
. /funasr-wss-client --server-ip 127.0.0.1 --port 10095 --wav-path test.wav --thread-num 1 --is-ssl 1
```

Introduction to command parameters:

```text
--server-ip: the IP address of the server. It can be set to 127.0.0.1 for local testing.
--port: the port number of the server listener.
--wav-path: the audio input. Input can be a path to a wav file or a wav.scp file (a Kaldi-formatted wav list in which each line includes a wav_id followed by a tab and a wav_path).
--is-ssl: whether to use SSL encryption. The default is to use SSL.
--hotword: If am is hotword model, setting hotword: *.txt(one hotword perline) or hotwords seperate by space (could be: 阿里巴巴 达摩院)
--use-itn: whether to use itn, the default value is 1 for enabling and 0 for disabling.
```

### Custom client

If you want to define your own client, see the [Websocket communication protocol](./websocket_protocol.md)

## How to customize service deployment

The code for FunASR-runtime is open source. If the server and client cannot fully meet your needs, you can further develop them based on your own requirements:

### C++ client

https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/websocket

### Python client

https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/websocket

### C++ server

#### VAD
```c++
// The use of the VAD model consists of two steps: FsmnVadInit and FsmnVadInfer:
FUNASR_HANDLE vad_hanlde=FsmnVadInit(model_path, thread_num);
// Where: model_path contains "model-dir" and "quantize", thread_num is the ONNX thread count;
FUNASR_RESULT result=FsmnVadInfer(vad_hanlde, wav_file.c_str(), NULL, 16000);
// Where: vad_hanlde is the return value of FunOfflineInit, wav_file is the path to the audio file, and sampling_rate is the sampling rate (default 16k).
```

See the usage example for details [docs](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/onnxruntime/bin/funasr-onnx-offline-vad.cpp)

#### ASR
```text
// The use of the ASR model consists of two steps: FunOfflineInit and FunOfflineInfer:
FUNASR_HANDLE asr_hanlde=FunOfflineInit(model_path, thread_num);
// Where: model_path contains "model-dir" and "quantize", thread_num is the ONNX thread count;
FUNASR_RESULT result=FunOfflineInfer(asr_hanlde, wav_file.c_str(), RASR_NONE, NULL, 16000);
// Where: asr_hanlde is the return value of FunOfflineInit, wav_file is the path to the audio file, and sampling_rate is the sampling rate (default 16k).
```

See the usage example for details, [docs](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/onnxruntime/bin/funasr-onnx-offline.cpp)

#### PUNC
```text
// The use of the PUNC model consists of two steps: CTTransformerInit and CTTransformerInfer:
FUNASR_HANDLE punc_hanlde=CTTransformerInit(model_path, thread_num);
// Where: model_path contains "model-dir" and "quantize", thread_num is the ONNX thread count;
FUNASR_RESULT result=CTTransformerInfer(punc_hanlde, txt_str.c_str(), RASR_NONE, NULL);
// Where: punc_hanlde is the return value of CTTransformerInit, txt_str is the text
```
See the usage example for details, [docs](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/onnxruntime/bin/funasr-onnx-offline-punc.cpp)
