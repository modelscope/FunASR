import os
import logging
import torch
import soundfile

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)

os.environ["MODELSCOPE_CACHE"] = "./"
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online',
    model_revision='v1.0.2')

model_dir = os.path.join(os.environ["MODELSCOPE_CACHE"], "damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online")
speech, sample_rate = soundfile.read(os.path.join(model_dir, "example/asr_example.wav"))
speech_length = speech.shape[0]

sample_offset = 0
step = 4800  #300ms
param_dict = {"cache": dict(), "is_final": False}
final_result = ""

for sample_offset in range(0, speech_length, min(step, speech_length - sample_offset)):
    if sample_offset + step >= speech_length - 1:
        step = speech_length - sample_offset
        param_dict["is_final"] = True
    rec_result = inference_pipeline(audio_in=speech[sample_offset: sample_offset + step],
                                    param_dict=param_dict)
    if len(rec_result) != 0 and rec_result['text'] != "sil" and rec_result['text'] != "waiting_for_more_voice":
        final_result += rec_result['text']
    print(rec_result)
print(final_result)
