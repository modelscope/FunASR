# Real-time Speech Transcription Service Development Guide

FunASR provides a real-time speech transcription service that can be easily deployed on local or cloud servers, with the FunASR runtime-SDK as the core. It integrates the speech endpoint detection (VAD), Paraformer-large non-streaming speech recognition (ASR), Paraformer-large streaming speech recognition (ASR), punctuation (PUNC), and other related capabilities open-sourced by the speech laboratory of DAMO Academy on the Modelscope community. The software package can perform real-time speech-to-text transcription, and can also accurately transcribe text at the end of sentences for high-precision output. The output text contains punctuation and supports high-concurrency multi-channel requests.

## Quick Start
### Pull Docker Image

Use the following command to pull and start the FunASR software package docker image:

```shell
sudo docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.2
mkdir -p ./funasr-runtime-resources/models
sudo docker run -p 10095:10095 -it --privileged=true -v $PWD/funasr-runtime-resources/models:/workspace/models registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.2
```
If you do not have Docker installed, please refer to [Docker Installation](https://alibaba-damo-academy.github.io/FunASR/en/installation/docker.html)

### Launching the Server

After Docker is launched, start the funasr-wss-server-2pass service program:
```shell
cd FunASR/funasr/runtime
nohup bash run_server_2pass.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx  \
  --online-model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx  \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx \
  --itn-dir thuduj12/fst_itn_zh > log.out 2>&1 &

# If you want to close ssl，please add：--certfile 0
```
For a more detailed description of server parameters, please refer to Server Introduction
### Client Testing and Usage

Download the client testing tool directory `samples`:
```shell
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz
```
For illustration, we will use the Python language client, which supports audio formats (.wav, .pcm) and a multi-file list wav.scp input.

```shell
python3 wss_client_asr.py --host "127.0.0.1" --port 10095 --mode 2pass
```

------------------

## Client Usage Details

After completing the FunASR service deployment on the server, you can test and use the offline file transcription service by following these steps. Currently, the following programming language client versions are supported:

- [Python](./SDK_tutorial_online.md#python-client)
- [CPP](./SDK_tutorial_online.md#cpp-client)
- [Html](./SDK_tutorial_online.md#html-client)
- [Java](./SDK_tutorial_online.md#java-client)
- [C\#](./SDK_tutorial_online.md#c\#)

For more detailed usage, please click on the links above. For more client version support, please refer to [WebSocket/GRPC Protocol](./websocket_protocol_zh.md).


## Server Introduction

Use the flollowing script to start the server ：
```shell
cd /workspace/FunASR/funasr/runtime
nohup bash run_server_2pass.sh \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx \
  --online-model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx \
  --itn-dir thuduj12/fst_itn_zh \
  --decoder-thread-num 32 \
  --io-thread-num  8 \
  --port 10095 \
  --certfile  ../../../ssl_key/server.crt \
  --keyfile ../../../ssl_key/server.key > log.out 2>&1 &

# If you want to close ssl，please add：--certfile 0
# If you want to deploy the timestamp or hotword model, please set --model-dir to the corresponding model:
# speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx（timestamp）
# damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx（hotword）

```

### More details about the script run_server_2pass.sh:
```text
--download-model-dir: Model download address, download models from Modelscope by setting the model ID.
--model-dir: Modelscope model ID.
--online-model-dir modelscope model ID
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
# Check the PID of the funasr-wss-server-2pass process
ps -x | grep funasr-wss-server-2pass
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
[FSMN-VAD model](https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx/summary),
[Paraformer-lagre online](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx/summary),
[Paraformer-lagre](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx/summary),
[CT-Transformer](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx/summary),
[FST-ITN](https://www.modelscope.cn/models/thuduj12/fst_itn_zh/summary)

If you wish to deploy your fine-tuned model (e.g., 10epoch.pb), you need to manually rename the model to model.pb and replace the original model.pb in ModelScope. Then, specify the path as `model_dir`.
