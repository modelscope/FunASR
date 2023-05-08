from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

if __name__ == "__main__":
    audio_in = "https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/asr_example.wav"
    output_dir = "./results"
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model="damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch",
        output_dir=output_dir,
    )
    rec_result = inference_pipeline(audio_in=audio_in)
    print(rec_result)
