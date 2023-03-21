import torch
import torchaudio
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from modelscope.utils.logger import get_logger
import logging
logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online',
    model_revision='v1.0.2')

waveform, sample_rate = torchaudio.load("waihu.wav")
speech_length = waveform.shape[1]
speech = waveform[0]

cache_en = {"start_idx": 0, "pad_left": 0, "stride": 10, "pad_right": 5, "cif_hidden": None, "cif_alphas": None}
cache_de = {"decode_fsmn": None}
cache = {"encoder": cache_en, "decoder": cache_de}
param_dict = {}
param_dict["cache"] = cache

first_chunk = True
speech_buffer = speech
speech_cache = []
final_result = ""

while len(speech_buffer) >= 960:
    if first_chunk:
        if len(speech_buffer) >= 14400:
            rec_result = inference_pipeline(audio_in=speech_buffer[0:14400], param_dict=param_dict)
            speech_buffer = speech_buffer[4800:]
        else:
            cache_en["stride"] = len(speech_buffer) // 960
            cache_en["pad_right"] = 0
            rec_result = inference_pipeline(audio_in=speech_buffer, param_dict=param_dict)
            speech_buffer = []
        cache_en["start_idx"] = -5
        first_chunk = False
    else:
        cache_en["start_idx"] += 10
        if len(speech_buffer) >= 4800:
            cache_en["pad_left"] = 5
            rec_result = inference_pipeline(audio_in=speech_buffer[:19200], param_dict=param_dict)
            speech_buffer = speech_buffer[9600:]
        else:
            cache_en["stride"] = len(speech_buffer) // 960 
            cache_en["pad_right"] = 0
            rec_result = inference_pipeline(audio_in=speech_buffer, param_dict=param_dict)
            speech_buffer = []
    if len(rec_result) !=0 and rec_result['text'] != "sil":
        final_result += rec_result['text']
    print(rec_result)
print(final_result)
