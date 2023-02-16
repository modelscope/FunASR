from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


if __name__ == '__main__':
    param_dict = dict()
    param_dict['hotword'] = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/hotword.txt"

    audio_in = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_hotword.wav"
    output_dir = None
    batch_size = 1

    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model="damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
        output_dir=output_dir,
        batch_size=batch_size,
        param_dict=param_dict)

    rec_result = inference_pipeline(audio_in=audio_in)
    print(rec_result)
