from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

decoding_mode="normal" #fast, normal, offline
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-online',
    param_dict={"decoding_model": decoding_mode}
)

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
print(rec_result)
