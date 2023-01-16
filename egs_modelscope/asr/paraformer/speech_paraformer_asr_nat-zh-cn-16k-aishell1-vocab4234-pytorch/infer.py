from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

if __name__ == '__main__':
    audio_in = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav'
    output_dir = None
    inference_pipline = pipeline(
        task=Tasks.auto_speech_recognition,
        model="damo/speech_paraformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch",
        output_dir=output_dir,
        batch_size=32,
    )
    rec_result = inference_pipline(audio_in=audio_in)
    print(rec_result)

