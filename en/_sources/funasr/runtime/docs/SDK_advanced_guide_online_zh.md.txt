# FunASR实时语音听写服务开发指南

FunASR提供可便捷本地或者云端服务器部署的实时语音听写服务，内核为FunASR已开源runtime-SDK。
集成了达摩院语音实验室在Modelscope社区开源的语音端点检测(VAD)、Paraformer-large非流式语音识别(ASR)、Paraformer-large流式语音识别(ASR)、标点(PUNC) 等相关能力。软件包既可以实时地进行语音转文字，而且能够在说话句尾用高精度的转写文字修正输出，输出文字带有标点，支持高并发多路请求

本文档为FunASR实时转写服务开发指南。如果您想快速体验实时语音听写服务，可参考[快速上手](#快速上手)。

## 快速上手
### 镜像启动

通过下述命令拉取并启动FunASR软件包的docker镜像：

```shell
sudo docker pull \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.2
mkdir -p ./funasr-runtime-resources/models
sudo docker run -p 10095:10095 -it --privileged=true \
  -v $PWD/funasr-runtime-resources/models:/workspace/models \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.2
```
如果您没有安装docker，可参考[Docker安装](https://alibaba-damo-academy.github.io/FunASR/en/installation/docker_zh.html)

### 服务端启动

docker启动之后，启动 funasr-wss-server-2pass服务程序：
```shell
cd FunASR/funasr/runtime
nohup bash run_server_2pass.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx  \
  --online-model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx  \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx \
  --itn-dir thuduj12/fst_itn_zh > log.out 2>&1 &

# 如果您想关闭ssl，增加参数：--certfile 0
# 如果您想使用时间戳或者热词模型进行部署，请设置--model-dir为对应模型：
# damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx（时间戳）
# 或者 damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx（热词）
```
服务端详细参数介绍可参考[服务端用法详解](#服务端用法详解)
### 客户端测试与使用

下载客户端测试工具目录samples
```shell
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz
```
我们以Python语言客户端为例，进行说明，支持音频格式（.wav, .pcm），以及多文件列表wav.scp输入，其他版本客户端请参考文档（[点击此处](#客户端用法详解)）。
```shell
python3 wss_client_asr.py --host "127.0.0.1" --port 10095 --mode 2pass
```

------------------

## 客户端用法详解

在服务器上完成FunASR服务部署以后，可以通过如下的步骤来测试和使用离线文件转写服务。
目前分别支持以下几种编程语言客户端

- [Python](./SDK_tutorial_online_zh.md#python-client)
- [CPP](./SDK_tutorial_online_zh.md#cpp-client)
- [html网页版本](./SDK_tutorial_online_zh.md#html-client)
- [Java](./SDK_tutorial_online_zh.md#java-client)
- [c\#](./SDK_tutorial_online_zh.md#c\#)

详细用法可以点击进入查看。更多版本客户端支持请参考[websocket/grpc协议](./websocket_protocol_zh.md)

## 服务端用法详解：

### 启动FunASR服务
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
 ```
**run_server_2pass.sh命令参数介绍**
```text
--download-model-dir 模型下载地址，通过设置model ID从Modelscope下载模型
--model-dir  modelscope model ID
--online-model-dir  modelscope model ID
--quantize  True为量化ASR模型，False为非量化ASR模型，默认是True
--vad-dir  modelscope model ID
--vad-quant   True为量化VAD模型，False为非量化VAD模型，默认是True
--punc-dir  modelscope model ID
--punc-quant   True为量化PUNC模型，False为非量化PUNC模型，默认是True
--itn-dir modelscope model ID
--port  服务端监听的端口号，默认为 10095
--decoder-thread-num  服务端启动的推理线程数，默认为 8
--io-thread-num  服务端启动的IO线程数，默认为 1
--certfile  ssl的证书文件，默认为：../../../ssl_key/server.crt，如果需要关闭ssl，参数设置为0
--keyfile   ssl的密钥文件，默认为：../../../ssl_key/server.key
```

### 关闭FunASR服务
```text
# 查看 funasr-wss-server-2pass 对应的PID
ps -x | grep funasr-wss-server-2pass
kill -9 PID
```

### 修改模型及其他参数
替换正在使用的模型或者其他参数，需先关闭FunASR服务，修改需要替换的参数，并重新启动FunASR服务。其中模型需为ModelScope中的ASR/VAD/PUNC模型，或者从ModelScope中模型finetune后的模型。
```text
# 例如替换ASR模型为 damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx，则如下设置参数 --model-dir
    --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx 
# 设置端口号 --port
    --port <port number>
# 设置服务端启动的推理线程数 --decoder-thread-num
    --decoder-thread-num <decoder thread num>
# 设置服务端启动的IO线程数 --io-thread-num
    --io-thread-num <io thread num>
# 关闭SSL证书 
    --certfile 0
```

执行上述指令后，启动实时语音听写服务。如果模型指定为ModelScope中model id，会自动从MoldeScope中下载如下模型：
[FSMN-VAD模型](https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx/summary)，
[Paraformer-lagre实时模型](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx/summary )
[Paraformer-lagre非实时模型](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx/summary)
[CT-Transformer标点预测模型](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx/summary)
[基于FST的中文ITN](https://www.modelscope.cn/models/thuduj12/fst_itn_zh/summary)

如果，您希望部署您finetune后的模型（例如10epoch.pb），需要手动将模型重命名为model.pb，并将原modelscope中模型model.pb替换掉，将路径指定为`model_dir`即可。

