from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_UniASR_asr_2pass-tr-16k-common-vocab1582-pytorch',
    model_revision='v1.0.0')

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_tr.wav')
print(rec_result)