from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

if __name__ == '__main__':
    audio_in = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav'
    output_dir = None
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model="damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab8358-tensorflow1-online",
        output_dir=output_dir,
    )
    rec_result = inference_pipeline(audio_in=audio_in)
    print(rec_result)

