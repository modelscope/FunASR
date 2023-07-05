from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

param_dict = dict()
param_dict['hotword'] = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/hotword.txt"
param_dict['clas_scale'] = 1.00  # 1.50 # set it larger if you want high recall (sacrifice general accuracy)
# 13% relative recall raise over internal hotword test set (45%->51%)
# CER might raise when utterance contains no hotword

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model="damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
    param_dict=param_dict)

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_hotword.wav')
print(rec_result)
