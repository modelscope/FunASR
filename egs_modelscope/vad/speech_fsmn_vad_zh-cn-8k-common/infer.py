from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

if __name__ == '__main__':
    audio_in = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example_8k.wav'
    output_dir = None
    inference_pipline = pipeline(
        task=Tasks.voice_activity_detection,
        model="damo/speech_fsmn_vad_zh-cn-8k-common",
        model_revision='v1.1.1',
        output_dir='./output_dir',
        batch_size=1,
    )
    segments_result = inference_pipline(audio_in=audio_in)
    print(segments_result)
