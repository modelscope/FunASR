# Speech Recognition

## Inference

### Quick start
#### Paraformer model
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
)

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
print(rec_result)
```

#### Inference with you data

#### Inference with multi-threads on CPU

#### Inference with multi GPU

## Finetune with pipeline

### Quick start

### Finetune with your data

## Inference with your finetuned model

