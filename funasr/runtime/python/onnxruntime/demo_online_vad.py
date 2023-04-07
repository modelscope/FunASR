import soundfile
from funasr_onnx.vad_online_bin import Fsmn_vad


model_dir = "/mnt/ailsa.zly/tfbase/espnet_work/FunASR_dev_gzf/export/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
wav_path = "/mnt/ailsa.zly/tfbase/espnet_work/FunASR_dev_gzf/egs_modelscope/vad/speech_fsmn_vad_zh-cn-16k-common/vad_example_16k.wav"
model = Fsmn_vad(model_dir)


##online vad
speech, sample_rate = soundfile.read(wav_path)
speech_length = speech.shape[0]
#
sample_offset = 0
step = 1600
param_dict = {'in_cache': []}
for sample_offset in range(0, speech_length, min(step, speech_length - sample_offset)):
    if sample_offset + step >= speech_length - 1:
        step = speech_length - sample_offset
        is_final = True
    else:
        is_final = False
    param_dict['is_final'] = is_final
    segments_result = model(audio_in=speech[sample_offset: sample_offset + step],
                            param_dict=param_dict)
    if segments_result:
        print(segments_result)

