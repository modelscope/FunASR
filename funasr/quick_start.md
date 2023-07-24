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
cd funasr/runtime/python/websocket
python funasr_wss_server.py --port 10095
```

#### Client Testing

```shell
python funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode 2pass --chunk_size "5,10,5"
```

For more examples, please refer to [docs](https://alibaba-damo-academy.github.io/FunASR/en/runtime/websocket_python.html#id2).

### C++ version Example

Currently, offline file transcription service (CPU) is supported, and concurrent requests of hundreds of channels are supported.

#### Server Deployment

You can use the following command to complete the deployment with one click:

```shell
curl -O https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/shell/funasr-runtime-deploy-offline-cpu-zh.sh
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh install --workspace ./funasr-runtime-resources
```

#### Client Testing

```shell
python3 funasr_wss_client.py --host "127.0.0.1" --port 10095 --mode offline --audio_in "../audio/asr_example.wav"
```

For more examples, please refer to [docs](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/docs/SDK_tutorial_zh.md)


## Industrial Model Egs

If you want to use the pre-trained industrial models in ModelScope for inference or fine-tuning training, you can refer to the following command:

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

More examples could be found in [docs](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_pipeline/quick_start.html)

## Academic model egs

If you want to train from scratch, usually for academic models, you can start training and inference with the following command:

```shell
cd egs/aishell/paraformer
. ./run.sh --CUDA_VISIBLE_DEVICES="0,1" --gpu_num=2
```
More examples could be found in [docs](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_pipeline/quick_start.html)