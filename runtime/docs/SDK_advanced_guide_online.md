# Real-time Speech Transcription Service Development Guide
([简体中文](SDK_advanced_guide_online_zh.md)|English)

[//]: # (FunASR provides a real-time speech transcription service that can be easily deployed on local or cloud servers, with the FunASR runtime-SDK as the core. It integrates the speech endpoint detection &#40;VAD&#41;, Paraformer-large non-streaming speech recognition &#40;ASR&#41;, Paraformer-large streaming speech recognition &#40;ASR&#41;, punctuation &#40;PUNC&#41;, and other related capabilities open-sourced by the speech laboratory of DAMO Academy on the Modelscope community. The software package can perform real-time speech-to-text transcription, and can also accurately transcribe text at the end of sentences for high-precision output. The output text contains punctuation and supports high-concurrency multi-channel requests.)
FunASR Real-time Speech Recognition Software Package integrates real-time versions of speech endpoint detection model, speech recognition model, punctuation prediction model, and so on. By using multiple models collaboratively, it can perform real-time speech-to-text conversion, as well as high-precision transcription correction at the end of a sentence, with punctuation included in the output text. It supports multiple concurrent requests. Depending on the user's scenarios, it supports three service modes: real-time speech recognition service (online), non-real-time single-sentence transcription (offline), and real-time and non-real-time integrated collaboration (2pass). The software package provides client libraries in various programming languages such as HTML, Python, C++, Java, and C#, allowing users to use and further develop the software.

<img src="images/online_structure.png"  width="900"/>

| TIME       | INFO                                                                                | IMAGE VERSION                       | IMAGE ID     |
|------------|-------------------------------------------------------------------------------------|-------------------------------------|--------------|
| 2024.05.15 | Adapting to FunASR 1.0 model structure | funasr-runtime-sdk-online-cpu-0.1.10 | 1c2adfcff84d |
| 2024.03.05 | docker image supports ARM64 platform, update modelscope | funasr-runtime-sdk-online-cpu-0.1.9 | 4a875e08c7a2 |
| 2024.01.25 | Optimization of the client-side | funasr-runtime-sdk-online-cpu-0.1.7  | 2aa23805572e      |
| 2024.01.03 | The 2pass-offline mode supports Ngram language model decoding and WFST hotwords, while also addressing known crash issues and memory leak problems | funasr-runtime-sdk-online-cpu-0.1.6  | f99925110d27      |
| 2023.11.09 | fix bug: without online results                                                     | funasr-runtime-sdk-online-cpu-0.1.5 | b16584b6d38b      |
| 2023.11.08 | supporting server-side loading of hotwords, adaptation to runtime structure changes | funasr-runtime-sdk-online-cpu-0.1.4 | 691974017c38 |
| 2023.09.19 | supporting hotwords, timestamps, and ITN model in 2pass mode                        | funasr-runtime-sdk-online-cpu-0.1.2 | 7222c5319bcf |
| 2023.08.11 | addressing some known bugs (including server crashes)                               | funasr-runtime-sdk-online-cpu-0.1.1 | bdbdd0b27dee |
| 2023.08.07 | 1.0 released                                                                        | funasr-runtime-sdk-online-cpu-0.1.0 | bdbdd0b27dee |

## Quick Start
### Docker install
If you have already installed Docker, ignore this step!
```shell
curl -O https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/shell/install_docker.sh;
sudo bash install_docker.sh
```
If you do not have Docker installed, please refer to [Docker Installation](https://alibaba-damo-academy.github.io/FunASR/en/installation/docker.html)

### Pull Docker Image
Use the following command to pull and start the FunASR software package docker image:
```shell
sudo docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.10
mkdir -p ./funasr-runtime-resources/models
sudo docker run -p 10096:10095 -it --privileged=true -v $PWD/funasr-runtime-resources/models:/workspace/models registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.10
```

### Launching the Server

After Docker is launched, start the funasr-wss-server-2pass service program:
```shell
cd FunASR/runtime
nohup bash run_server_2pass.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx  \
  --online-model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx  \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx \
  --lm-dir damo/speech_ngram_lm_zh-cn-ai-wesp-fst \
  --itn-dir thuduj12/fst_itn_zh > log.txt 2>&1 &

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
python3 funasr_wss_client.py --host "127.0.0.1" --port 10096 --mode 2pass
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
cd /workspace/FunASR/runtime
nohup bash run_server_2pass.sh \
  --model-dir damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx \
  --online-model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx \
  --lm-dir damo/speech_ngram_lm_zh-cn-ai-wesp-fst \
  --itn-dir thuduj12/fst_itn_zh \
  --certfile  ../../../ssl_key/server.crt \
  --keyfile ../../../ssl_key/server.key \
  --hotword ../../hotwords.txt > log.txt 2>&1 &

# If you want to close ssl，please add：--certfile 0
# If you want to deploy the timestamp or nn hotword model, please set --model-dir to the corresponding model:
# speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx（timestamp）
# damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx（hotword）

```

### More details about the script run_server_2pass.sh:
```text
--download-model-dir: Model download address, download models from Modelscope by setting the model ID.
--model-dir: modelscope model ID or local model path.
--online-model-dir modelscope model ID
--vad-dir: modelscope model ID or local model path.
--punc-dir: modelscope model ID or local model path.
--lm-dir modelscope model ID or local model path.
--itn-dir modelscope model ID or local model path.
--port: Port number that the server listens on. Default is 10095.
--decoder-thread-num: The number of thread pools on the server side that can handle concurrent requests.
                      The script will automatically configure parameters decoder-thread-num and io-thread-num based on the server's thread count.
--io-thread-num: Number of IO threads that the server starts.
--model-thread-num: The number of internal threads for each recognition route to control the parallelism of the ONNX model. 
        The default value is 1. It is recommended that decoder-thread-num * model-thread-num equals the total number of threads.
--certfile <string>: SSL certificate file. Default is ../../../ssl_key/server.crt. If you want to close ssl，set 0
--keyfile <string>: SSL key file. Default is ../../../ssl_key/server.key. 
--hotword: Hotword file path, one line for each hotword(e.g.:阿里巴巴 20), if the client provides hot words, then combined with the hot words provided by the client.
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
