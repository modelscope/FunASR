from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/damo/speech_paraformer_asr-en-16k-vocab4199-pytorch',
    model_revision="v1.0.1",
)
audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_en.wav'
rec_result = inference_pipeline(audio_in=audio_in)
print(rec_result)
