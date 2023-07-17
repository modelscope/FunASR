from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

if __name__ == '__main__':
    audio_in = '/Users/shixian/Downloads/1640s.wav'
    output_dir = "./results"
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
        punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
        output_dir=output_dir,
        mode="paraformer_vad_clipvideo",
    )
    rec_result = inference_pipeline(audio_in=audio_in, batch_size_token=5000)
    print(rec_result)