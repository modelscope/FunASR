from funasr_onnx import ContextualParaformer
from pathlib import Path

model_dir = "damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404" # your export dir
model = ContextualParaformer(model_dir, batch_size=1)

wav_path = ['/Users/shixian/Downloads/sac_test.wav']

hotwords = '随机热词 各种热词 魔搭 阿里巴巴 仏'

result = model(wav_path, hotwords)
print(result)
