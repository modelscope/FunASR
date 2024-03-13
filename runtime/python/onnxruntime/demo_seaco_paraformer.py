from funasr_onnx import SeacoParaformer
from pathlib import Path

model_dir = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model = SeacoParaformer(model_dir, batch_size=1)

wav_path = ['/Users/shixian/Downloads/sac_test.wav']

hotwords = '随机热词 各种热词 魔搭 阿里巴巴 仏'

result = model(wav_path, hotwords)
print(result)
