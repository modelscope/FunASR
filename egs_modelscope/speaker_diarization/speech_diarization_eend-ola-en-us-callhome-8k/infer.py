from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_diar_pipline = pipeline(
    task=Tasks.speaker_diarization,
    model='damo/speech_diarization_eend-ola-en-us-callhome-8k',
)
results = inference_diar_pipline(audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/record.wav")