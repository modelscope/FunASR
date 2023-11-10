(简体中文|[English](./quick_start.md))

<a name="快速开始"></a>
## 快速开始

您可以通过如下几种方式使用FunASR功能:

- 服务部署社区软件包
- 工业模型egs
- 学术模型egs

### 服务部署社区软件包

#### python版本示例

支持实时流式语音识别，并且会用非流式模型进行纠错，输出文本带有标点。目前只支持单个client，如需多并发请参考下方c++版本服务部署SDK

##### 服务端部署
```shell
cd funasr/runtime/python/websocket
python funasr_wss_server.py --port 10095
```

##### 客户端测试
```shell
python funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode 2pass --chunk_size "5,10,5"
#python funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode 2pass --chunk_size "8,8,4" --audio_in "./data/wav.scp"
```
更多例子可以参考（[点击此处](../runtime/python/websocket/README.md)）

<a name="cpp版本示例"></a>
#### 服务部署软件包

既可以进行高精度、高效率与高并发的文件转写，也可以进行低延时的实时语音听写。支持Docker化部署，多路请求。

##### 准备工作：docker安装（可选）
###### 如果您已安装docker，忽略本步骤

```shell
curl -O https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/shell/install_docker.sh;
sudo bash install_docker.sh
```

##### 实时语音听写服务部署

###### docker镜像下载与启动
通过下述命令拉取并启动FunASR软件包docker镜像（[获取最新镜像版本](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_online_zh.md)）：

```shell
sudo docker pull \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.5
mkdir -p ./funasr-runtime-resources/models
sudo docker run -p 10096:10095 -it --privileged=true \
  -v $PWD/funasr-runtime-resources/models:/workspace/models \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.5
```

###### 服务端启动
docker启动之后，启动 funasr-wss-server-2pass服务程序：
```shell
cd FunASR/runtime
nohup bash run_server_2pass.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx  \
  --online-model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx  \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx \
  --itn-dir thuduj12/fst_itn_zh \
  --hotword /workspace/models/hotwords.txt > log.out 2>&1 &

# 如果您想关闭ssl，增加参数：--certfile 0
# 如果您想使用时间戳或者nn热词模型进行部署，请设置--model-dir为对应模型：
#   damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx（时间戳）
#   damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx（nn热词）
# 如果您想在服务端加载热词，请在宿主机文件./funasr-runtime-resources/models/hotwords.txt配置热词（docker映射地址为/workspace/models/hotwords.txt）:
#   每行一个热词，格式(热词 权重)：阿里巴巴 20
```

##### 客户端测试与使用
客户端测试（[samples](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz)）

```shell
python3 funasr_wss_client.py --host "127.0.0.1" --port 10096 --mode 2pass
```
更多例子参考（[点击此处](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_online_zh.md)）

##### 离线文件转写服务部署

###### 镜像启动

通过下述命令拉取并启动FunASR软件包docker镜像（[获取最新镜像版本](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_offline_zh.md)）：

```shell
sudo docker pull \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.3.0
mkdir -p ./funasr-runtime-resources/models
sudo docker run -p 10095:10095 -it --privileged=true \
  -v $PWD/funasr-runtime-resources/models:/workspace/models \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.3.0
```

###### 服务端启动

docker启动之后，启动 funasr-wss-server服务程序：
```shell
cd FunASR/runtime
nohup bash run_server.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx  \
  --punc-dir damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx \
  --lm-dir damo/speech_ngram_lm_zh-cn-ai-wesp-fst \
  --itn-dir thuduj12/fst_itn_zh \
  --hotword /workspace/models/hotwords.txt > log.out 2>&1 &

# 如果您想关闭ssl，增加参数：--certfile 0
# 如果您想使用时间戳或者nn热词模型进行部署，请设置--model-dir为对应模型：
#   damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx（时间戳）
#   damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx（nn热词）
# 如果您想在服务端加载热词，请在宿主机文件./funasr-runtime-resources/models/hotwords.txt配置热词（docker映射地址为/workspace/models/hotwords.txt）:
#   每行一个热词，格式(热词 权重)：阿里巴巴 20
```

###### 客户端测试
客户端测试（[samples](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz)）
```shell
python3 funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "../audio/asr_example.wav"
```
更多例子参考（[点击此处](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_offline_zh.md)）



### 工业模型egs

如果您希望使用ModelScope中预训练好的工业模型，进行推理或者微调训练，您可以参考下面指令：


```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
)

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
print(rec_result)
# {'text': '欢迎大家来体验达摩院推出的语音识别模型'}
```

更多例子可以参考（[点击此处](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_pipeline/quick_start.html)）


### 学术模型egs

如果您希望从头开始训练，通常为学术模型，您可以通过下面的指令启动训练与推理：

```shell
cd egs/aishell/paraformer
. ./run.sh --CUDA_VISIBLE_DEVICES="0,1" --gpu_num=2
```

更多例子可以参考（[点击此处](https://alibaba-damo-academy.github.io/FunASR/en/academic_recipe/asr_recipe.html)）