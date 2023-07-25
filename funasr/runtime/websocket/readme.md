([简体中文](./readme_zh.md)|English)

# Service with websocket-cpp


## Quick Start
### Docker Image start

Pull and start the FunASR runtime-SDK Docker image using the following command:

```shell
sudo docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.1.0

sudo docker run -p 10095:10095 -it --privileged=true -v /root:/workspace/models registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.1.0
```

If you have not installed Docker, please refer to [Docker Installation](https://alibaba-damo-academy.github.io/FunASR/en/installation/docker.html).

### Server Start

After Docker is started, start the funasr-wss-server service program:

```shell
cd FunASR/funasr/runtime
./run_server.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx  \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx
```
For detailed server parameters, please refer to [Server Parameter Introduction](#Server Parameter Introduction).

### Client Testing and Usage

Download the client test tool directory samples:

```shell
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz
```

We take the Python language client as an example to explain. It supports various audio formats (.wav, .pcm, .mp3, etc.), video input (.mp4, etc.), and multi-file list wav.scp input. For other versions of clients, please refer to the document ([click here](#Detailed Usage of Clients)). For customized service deployment, please refer to [How to Customize Service Deployment](#How to Customize Service Deployment).

```shell
python3 wss_client_asr.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "../audio/asr_example.wav"
```

## Detailed Steps

### Dependencies Download and Install

The third-party libraries have been pre-installed in Docker. If not using Docker, please download and install them manually ([Download and Install Third-Party Libraries](requirements_install.md)).


### Build runtime

```shell
git clone https://github.com/alibaba-damo-academy/FunASR.git && cd FunASR/funasr/runtime/websocket
mkdir build && cd build
cmake  -DCMAKE_BUILD_TYPE=release .. -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.14.0 -DFFMPEG_DIR=/path/to/ffmpeg-N-111383-g20b8688092-linux64-gpl-shared
make
```


### Start Service Deployment

#### API-reference：
```text
--download-model-dir Model download address, download the model from Modelscope by setting the model ID. If starting from a local model, this parameter can be left out.
--model-dir ASR model ID in Modelscope or the absolute path of local model
--quantize True for quantized ASR model, False for non-quantized ASR model. Default is True.
--vad-dir VAD model ID in Modelscope or the absolute path of local model
--vad-quant True for quantized VAD model, False for non-quantized VAD model. Default is True.
--punc-dir PUNC model ID in Modelscope or the absolute path of local model
--punc-quant True for quantized PUNC model, False for non-quantized PUNC model. Default is True.
--port Port number for the server to listen on. Default is 10095.
--decoder-thread-num Number of inference threads started by the server. Default is 8.
--io-thread-num Number of IO threads started by the server. Default is 1.
--certfile SSL certificate file. Default is: ../../../ssl_key/server.crt.
--keyfile SSL key file. Default is: ../../../ssl_key/server.key.
```

#### Example of Starting from Modelscope
```shell
./funasr-wss-server  \
  --download-model-dir /workspace/models \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx
```

Note: In the above example, `model-dir`，`vad-dir`，`punc-dir` are the model names in Modelscope, downloaded directly from Modelscope and exported as quantized onnx. If starting from a local model, please change the parameter to the absolute path of the local model.


#### Example of Starting from Local Model

##### Export the Model

```shell
python -m funasr.export.export_model \
--export-dir ./export \
--type onnx \
--quantize True \
--model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
--model-name damo/speech_fsmn_vad_zh-cn-16k-common-pytorch \
--model-name damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
```

Export Detailed Introduction（[docs](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export)）

##### Start the Service
```shell
./funasr-wss-server  \
  --download-model-dir /workspace/models \
  --model-dir ./exportdamo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx \
  --vad-dir ./exportdamo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --punc-dir ./export/damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx
```

### Client Usage


Download the client test tool directory [samples](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz)

```shell
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz
```

After deploying the FunASR service on the server, you can test and use the offline file transcription service through the following steps. Currently, the following programming language client is supported:

- [Python](#python-client)
- [CPP](#cpp-client)
- [html](#Html-client)
- [Java](#Java-client)

#### python-client

If you want to run the client directly for testing, you can refer to the following simple instructions, taking the Python version as an example:
```shell
python3 wss_client_asr.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "../audio/asr_example.wav" --output_dir "./results"
```

API-reference
```text
--host: IP address of the machine where FunASR runtime-SDK service is deployed. The default value is the IP address of the local machine (127.0.0.1). If the client and service are not on the same server, it needs to be changed to the IP address of the deployment machine.
--port: The port number of the deployed service is 10095.
--mode: "offline" means offline file transcription.
--audio_in: The audio file that needs to be transcribed, which supports file path and file list (wav.scp).
--output_dir: The path to save the recognition result.
```

### cpp-client

After entering the directory samples/cpp, you can test it with CPP, as follows:

```shell
./funasr-wss-client --server-ip 127.0.0.1 --port 10095 --wav-path ../audio/asr_example.wav
```

API-reference:

```text
--server-ip: The IP address of the machine where FunASR runtime-SDK service is deployed. The default value is the IP address of the local machine (127.0.0.1). If the client and service are not on the same server, it needs to be changed to the IP address of the deployment machine.
--port: The port number of the deployed service is 10095.
--wav-path: The audio file that needs to be transcribed, which supports file path.
```

### Html-client

Open `html/static/index.html` in the browser, and you can see the following page, which supports microphone input and file upload for direct experience.

<img src="images/html.png"  width="900"/>

### Java-client

```shell
FunasrWsClient --host localhost --port 10095 --audio_in ./asr_example.wav --mode offline
```
For more details, please refer to the [documentation](../java/readme.md) 


## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [zhaoming](https://github.com/zhaomingwork/FunASR/tree/add-offline-websocket-srv/funasr/runtime/websocket) for contributing the websocket(cpp-api).


