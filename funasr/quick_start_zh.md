(简体中文|[English](./quick_start.md))

<a name="快速开始"></a>
## 快速开始

您可以通过如下几种方式使用FunASR功能:

- 服务部署SDK
- 工业模型egs
- 学术模型egs

### 服务部署SDK

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
更多例子可以参考（[点击此处](https://alibaba-damo-academy.github.io/FunASR/en/runtime/websocket_python.html#id2)）

<a name="cpp版本示例"></a>
#### c++版本示例

目前已支持离线文件转写服务（CPU），支持上百路并发请求

##### 服务端部署
可以用个下面指令，一键部署完成部署
```shell
curl -O https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/shell/funasr-runtime-deploy-offline-cpu-zh.sh
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh install --workspace ./funasr-runtime-resources
```

##### 客户端测试
客户端测试（[samples](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz)）
```shell
python3 funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "../audio/asr_example.wav"
```
更多例子参考（[点击此处](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/docs/SDK_tutorial_zh.md)）


### 工业模型egs

如果您希望使用ModelScope中预训练好的工业模型，进行推理或者微调训练，您可以参考下面指令：


```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
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