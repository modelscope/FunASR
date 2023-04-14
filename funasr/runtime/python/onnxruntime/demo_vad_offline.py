import soundfile
from funasr_onnx import Fsmn_vad


model_dir = "/mnt/ailsa.zly/tfbase/espnet_work/FunASR_dev_zly/export/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
wav_path = "/mnt/ailsa.zly/tfbase/espnet_work/FunASR_dev_zly/egs_modelscope/vad/speech_fsmn_vad_zh-cn-16k-common/vad_example_16k.wav"
model = Fsmn_vad(model_dir)

#offline vad
result = model(wav_path)
print(result)
