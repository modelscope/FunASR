([简体中文](./quick_start_zh.md)|English)

# Quick Start

You can use FunASR in the following ways:

- Service Deployment SDK
- Industrial model egs
- Academic model egs

## Service Deployment SDK

### Python version Example
Supports real-time streaming speech recognition, uses non-streaming models for error correction, and outputs text with punctuation. Currently, only single client is supported. For multi-concurrency, please refer to the C++ version service deployment SDK below.

#### Server Deployment

```shell
cd runtime/python/websocket
python funasr_wss_server.py --port 10095
```

#### Client Testing

```shell
python funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode 2pass --chunk_size "5,10,5"
```

For more examples, please refer to [docs](../runtime/python/websocket/README.md).

### Service Deployment Software

Both high-precision, high-efficiency, and high-concurrency file transcription, as well as low-latency real-time speech recognition, are supported. It also supports Docker deployment and multiple concurrent requests.

##### Docker Installation (optional)
###### If you have already installed Docker, skip this step.

```shell
curl -O https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/shell/install_docker.sh;
sudo bash install_docker.sh
```

##### Real-time Speech Recognition Service Deployment

###### Docker Image Download and Launch
Use the following command to pull and launch the FunASR software package Docker image（[Get the latest image version](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_online.md)）：

```shell
sudo docker pull \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.11
mkdir -p ./funasr-runtime-resources/models
sudo docker run -p 10096:10095 -it --privileged=true \
  -v $PWD/funasr-runtime-resources/models:/workspace/models \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.11
```

###### Server Start

After Docker is started, start the funasr-wss-server-2pass service program:

```shell
cd FunASR/runtime
nohup bash run_server_2pass.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx  \
  --online-model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx  \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx \
  --itn-dir thuduj12/fst_itn_zh \
  --hotword /workspace/models/hotwords.txt > log.txt 2>&1 &

# If you want to disable SSL, add the parameter: --certfile 0
# If you want to deploy with a timestamp or nn hotword model, please set --model-dir to the corresponding model:
#   damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx (timestamp)
#   damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx (nn hotword)
# If you want to load hotwords on the server side, please configure the hotwords in the host file ./funasr-runtime-resources/models/hotwords.txt (docker mapping address is /workspace/models/hotwords.txt):
#   One hotword per line, format (hotword weight): Alibaba 20
```

###### Client Testing
Testing [samples](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz)

```shell
python3 funasr_wss_client.py --host "127.0.0.1" --port 10096 --mode 2pass
```
For more examples, please refer to [docs](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_online.md)


#### File Transcription Service, Mandarin (CPU)

###### Docker Image Download and Launch
Use the following command to pull and launch the FunASR software package Docker image（[Get the latest image version](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_offline.md)）：

```shell
sudo docker pull \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.4.6
mkdir -p ./funasr-runtime-resources/models
sudo docker run -p 10095:10095 -it --privileged=true \
  -v $PWD/funasr-runtime-resources/models:/workspace/models \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.4.6
```

###### Server Start

After Docker is started, start the funasr-wss-server service program:

```shell
cd FunASR/runtime
nohup bash run_server.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx  \
  --punc-dir damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx \
  --lm-dir damo/speech_ngram_lm_zh-cn-ai-wesp-fst \
  --itn-dir thuduj12/fst_itn_zh \
  --hotword /workspace/models/hotwords.txt > log.txt 2>&1 &

# If you want to disable SSL, add the parameter: --certfile 0
# If you want to use timestamp or nn hotword models for deployment, please set --model-dir to the corresponding model:
#   damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx (timestamp)
#   damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx (nn hotword)
# If you want to load hotwords on the server side, please configure the hotwords in the host machine file ./funasr-runtime-resources/models/hotwords.txt (docker mapping address is /workspace/models/hotwords.txt):
#   One hotword per line, format (hotword weight): Alibaba 20
```

##### Client Testing

Testing [samples](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz)
```shell
python3 funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "../audio/asr_example.wav"
```

For more examples, please refer to [docs](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_offline.md)

