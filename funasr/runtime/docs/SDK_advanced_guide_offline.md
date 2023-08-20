 # Advanced Development Guide (File transcription service)
 
FunASR provides a Chinese offline file transcription service that can be deployed locally or on a cloud server with just one click. The core of the service is the FunASR runtime SDK, which has been open-sourced. FunASR-runtime combines various capabilities such as speech endpoint detection (VAD), large-scale speech recognition (ASR) using Paraformer-large, and punctuation detection (PUNC), which have all been open-sourced by the speech laboratory of DAMO Academy on the Modelscope community. This enables accurate and efficient high-concurrency transcription of audio files.

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
sudo docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-latest

sudo docker run -p 10095:10095 -it --privileged=true -v /root:/workspace/models registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-latest
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
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx  \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx > log.out 2>&1 &

# If you want to close ssl，please add：--certfile 0
# If you want to deploy the timestamp or hotword model, please set --model-dir to the corresponding model:
# speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx（timestamp）
# damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx（hotword）
```

More details about the script run_server.sh:

The FunASR-wss-server supports downloading models from Modelscope. You can set the model download address (--download-model-dir, default is /workspace/models) and the model ID (--model-dir, --vad-dir, --punc-dir). Here is an example:

```shell
cd /workspace/FunASR/funasr/runtime/websocket/build/bin
./funasr-wss-server  \
  --download-model-dir /workspace/models \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx \
  --decoder-thread-num 32 \
  --io-thread-num  8 \
  --port 10095 \
  --certfile  ../../../ssl_key/server.crt \
  --keyfile ../../../ssl_key/server.key
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

The FunASR-wss-server also supports loading models from a local path (see Preparing Model Resources for detailed instructions on preparing local model resources). Here is an example:

```shell
cd /workspace/FunASR/funasr/runtime/websocket/build/bin
./funasr-wss-server  \
  --model-dir /workspace/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx \
  --vad-dir /workspace/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --punc-dir /workspace/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx \
  --decoder-thread-num 32 \
  --io-thread-num  8 \
  --port 10095 \
  --certfile  ../../../ssl_key/server.crt \
  --keyfile ../../../ssl_key/server.key
 ```


## Preparing Model Resources

If you choose to download models from Modelscope through the FunASR-wss-server, you can skip this step. The vad, asr, and punc model resources in the offline file transcription service of FunASR are all from Modelscope. The model addresses are shown in the table below:

| Model | Modelscope url                                                                                                   |
|-------|------------------------------------------------------------------------------------------------------------------|
| VAD   | https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary |
| ASR   | https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary                           |
| PUNC  | https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary               |

The offline file transcription service deploys quantized ONNX models. Below are instructions on how to export ONNX models and their quantization. You can choose to export ONNX models from Modelscope, local files, or finetuned resources: 

### Exporting ONNX models from Modelscope

Download the corresponding model with the given model name from the Modelscope website, and then export the quantized ONNX model

```shell
python -m funasr.export.export_model \
--export-dir ./export \
--type onnx \
--quantize True \
--model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
--model-name damo/speech_fsmn_vad_zh-cn-16k-common-pytorch \
--model-name damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
```

Introduction to command parameters:

```text
--model-name: The name of the model on Modelscope, for example: damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
--export-dir: The export directory of ONNX model.
--type: Model type, currently supports ONNX and torch.
--quantize: Quantize the int8 model.
```

### Exporting ONNX models from local files

Set the model name to the local path of the model, and export the quantized ONNX model:

```shell
python -m funasr.export.export_model --model-name /workspace/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type onnx --quantize True
```


### Exporting models from finetuned resources

If you want to deploy a finetuned model, you can follow these steps:
Rename the model you want to deploy after finetuning (for example, 10epoch.pb) to model.pb, and replace the original model.pb in Modelscope with this one. If the path of the replaced model is /path/to/finetune/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch, use the following command to convert the finetuned model to an ONNX model: 

```shell
python -m funasr.export.export_model --model-name /path/to/finetune/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type onnx --quantize True
```

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
```

### c++-client
```shell
. /funasr-wss-client --server-ip 127.0.0.1 --port 10095 --wav-path test.wav --thread-num 1 --is-ssl 1
```

Introduction to command parameters:

```text
--host: the IP address of the server. It can be set to 127.0.0.1 for local testing.
--port: the port number of the server listener.
--audio_in: the audio input. Input can be a path to a wav file or a wav.scp file (a Kaldi-formatted wav list in which each line includes a wav_id followed by a tab and a wav_path).
--output_dir: the path to the recognition result output.
--ssl: whether to use SSL encryption. The default is to use SSL.
--mode: offline mode.
```

### Custom client

If you want to define your own client, the Websocket communication protocol is as follows:

```text
# First communication
{"mode": "offline", "wav_name": wav_name, "is_speaking": True, "hotwords": "hotword1|hotword2"}
# Send wav data
Bytes data
# Send end flag
{"is_speaking": False}
```

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
