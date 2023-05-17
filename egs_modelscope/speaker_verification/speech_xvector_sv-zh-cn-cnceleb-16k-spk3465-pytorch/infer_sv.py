from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

if __name__ == '__main__':
    inference_sv_pipline = pipeline(
        task=Tasks.speaker_verification,
        model='damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch'
    )

    # the same speaker
    rec_result = inference_sv_pipline(audio_in=(
        'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_enroll.wav',
        'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_same.wav'))
    print("Similarity", rec_result["scores"])

    # different speaker
    rec_result = inference_sv_pipline(audio_in=(
        'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_enroll.wav',
        'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_different.wav'))

    print("Similarity", rec_result["scores"])
