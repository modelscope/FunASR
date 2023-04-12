import soundfile
from funasr_onnx import Fsmn_vad


model_dir = "/Users/zhifu/Downloads/speech_fsmn_vad_zh-cn-16k-common-pytorch"
wav_path = "/Users/zhifu/Downloads/speech_fsmn_vad_zh-cn-16k-common-pytorch/example/vad_example.wav"
model = Fsmn_vad(model_dir)

#offline vad
# result = model(wav_path)
# print(result)

#online vad
speech, sample_rate = soundfile.read(wav_path)
speech_length = speech.shape[0]

sample_offset = 0
step = 160 * 10
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
    print(segments_result)

