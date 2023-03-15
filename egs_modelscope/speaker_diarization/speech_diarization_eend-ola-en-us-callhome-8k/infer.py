from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_diar_pipline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_diarization_eend-ola-en-us-callhome-8k',
    model_revision="v1.0.0",
)
results = inference_diar_pipline(audio_in=["https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/record2.wav"])
print(results)