from funasr_onnx import ContextualParaformer
from pathlib import Path

model_dir = "damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"
model = ContextualParaformer(model_dir, batch_size=1)

wav_path = ["{}/.cache/modelscope/hub/{}/example/asr_example.wav".format(Path.home(), model_dir)]

hotwords = "你的热词 魔搭"

result = model(wav_path, hotwords)
print(result)
