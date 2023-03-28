
from funasr_onnx import Fsmn_vad


model_dir = "/Users/zhifu/Downloads/speech_fsmn_vad_zh-cn-16k-common-pytorch"

model = Fsmn_vad(model_dir)

wav_path = "/Users/zhifu/Downloads/speech_fsmn_vad_zh-cn-16k-common-pytorch/example/vad_example.wav"

result = model(wav_path)
print(result)