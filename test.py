from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

'''
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    timestamp_model='damo/speech_timestamp_prediction-v1-16k-offline',
    timestamp_model_revision='v1.0.3',
    )

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
print(rec_result)
'''

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    vad_model_revision="v1.1.8",
    punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    punc_model_revision="v1.1.6")

rec_result = inference_pipeline(audio_in='/Users/shixian/Downloads/test.wav')
print(rec_result)
