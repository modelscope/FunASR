from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

if __name__ == "__main__":
    audio_in = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_vi.wav"
    output_dir = "./results"
    inference_pipline = pipeline(
        task=Tasks.auto_speech_recognition,
        model="damo/speech_UniASR_asr_2pass-vi-16k-common-vocab1001-pytorch-online",
        output_dir=output_dir,
    )
    rec_result = inference_pipline(audio_in=audio_in, param_dict={"decoding_model":"normal"})
    print(rec_result)
