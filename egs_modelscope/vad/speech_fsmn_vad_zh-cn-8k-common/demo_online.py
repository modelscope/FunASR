from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
import logging
logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)
import soundfile

if __name__ == '__main__':
    output_dir = None
    inference_pipeline = pipeline(
        task=Tasks.voice_activity_detection,
        model="damo/speech_fsmn_vad_zh-cn-8k-common",
        model_revision=None,
        output_dir=output_dir,
        batch_size=1,
        mode='online',
    )
    speech, sample_rate = soundfile.read("./vad_example_8k.wav")
    speech_length = speech.shape[0]
    
    sample_offset = 0
    
    step = 80 * 10
    param_dict = {'in_cache': dict(), 'max_end_sil': 800}
    for sample_offset in range(0, speech_length, min(step, speech_length - sample_offset)):
        if sample_offset + step >= speech_length - 1:
            step = speech_length - sample_offset
            is_final = True
        else:
            is_final = False
        param_dict['is_final'] = is_final
        segments_result = inference_pipeline(audio_in=speech[sample_offset: sample_offset + step],
                                            param_dict=param_dict)
        print(segments_result)

