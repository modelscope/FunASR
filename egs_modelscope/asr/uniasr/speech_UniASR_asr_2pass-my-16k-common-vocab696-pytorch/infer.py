from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

if __name__ == "__main__":
    audio_in = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_my.wav"
    output_dir = "./results"
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model="damo/speech_UniASR_asr_2pass-my-16k-common-vocab696-pytorch",
        output_dir=output_dir,
    )
    rec_result = inference_pipeline(audio_in=audio_in, param_dict={"decoding_model":"offline"})
    print(rec_result)
