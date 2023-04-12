from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

if __name__ == "__main__":
    audio_in = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_en.wav"
    output_dir = "./results"
    inference_pipline = pipeline(
        task=Tasks.auto_speech_recognition,
        model="damo/speech_conformer_asr-en-16k-vocab4199-pytorch",
        output_dir=output_dir,
    )
    rec_result = inference_pipline(audio_in=audio_in)
    print(rec_result)
