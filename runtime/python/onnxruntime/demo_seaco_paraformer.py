from funasr_onnx import SeacoParaformer
from pathlib import Path

model_dir = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model = SeacoParaformer(model_dir, batch_size=1)

wav_path = ["{}/.cache/modelscope/hub/{}/example/asr_example.wav".format(Path.home(), model_dir)]

hotwords = "你的热词 魔搭"

result = model(wav_path, hotwords)
print(result)
