# Real-time Speech Transcription Service Development Guide

FunASR provides a real-time speech transcription service that can be easily deployed on local or cloud servers, with the FunASR runtime-SDK as the core. It integrates the speech endpoint detection (VAD), Paraformer-large non-streaming speech recognition (ASR), Paraformer-large streaming speech recognition (ASR), punctuation (PUNC), and other related capabilities open-sourced by the speech laboratory of DAMO Academy on the Modelscope community. The software package can perform real-time speech-to-text transcription, and can also accurately transcribe text at the end of sentences for high-precision output. The output text contains punctuation and supports high-concurrency multi-channel requests.

## Quick Start
### Pull Docker Image

Use the following command to pull and start the FunASR software package docker image:

```shell
sudo docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.1
mkdir -p ./funasr-runtime-resources/models
sudo docker run -p 10095:10095 -it --privileged=true -v ./funasr-runtime-resources/models:/workspace/models registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.1
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
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx > log.out 2>&1 &
```
For a more detailed description of server parameters, please refer to [Server Introduction]()
### Client Testing and Usage

Download the client testing tool directory `samples`:
```shell
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz
```
For illustration, we will use the Python language client, which supports audio formats (.wav, .pcm) and a multi-file list wav.scp input. For other client versions, please refer to the [documentation]().

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

## Server Introduction:

funasr-wss-server-2pass supports downloading models from Modelscope or starting from a local directory path, as shown below:
```shell
cd /workspace/FunASR/funasr/runtime/websocket/build/bin
./funasr-wss-server-2pass  \
  --decoder-thread-num 32 \
  --io-thread-num  8 \
  --port 10095 
 ```

Command parameter introduction:
```text
--download-model-dir Model download address, download models from Modelscope by setting model id
--model-dir modelscope model ID
--online-model-dir modelscope model ID
--quantize True for quantized ASR models, False for non-quantized ASR models, default is True
--vad-dir modelscope model ID
--vad-quant True for quantized VAD models, False for non-quantized VAD models, default is True
--punc-dir modelscope model ID
--punc-quant True for quantized PUNC models, False for non-quantized PUNC models, default is True
--port Port number that the server should listen on, default is 10095
--decoder-thread-num The number of inference threads the server should start, default is 8
--io-thread-num The number of IO threads the server should start, default is 1
--certfile SSL certificate file, the default is: ../../../ssl_key/server.crt, set to "" to disable
--keyfile SSL key file, the default is: ../../../ssl_key/server.key, set to "" to disable
```

After executing the above command, the real-time speech transcription service will be started. If the model is specified as a ModelScope model id, the following models will be automatically downloaded from ModelScope:
[FSMN-VAD model](https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx/summary)，
[Paraformer-lagre online](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx/summary )
[Paraformer-lagre](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx/summary)
[CT-Transformer](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx/summary)

If you wish to deploy your fine-tuned model (e.g., 10epoch.pb), you need to manually rename the model to model.pb and replace the original model.pb in ModelScope. Then, specify the path as `model_dir`.
