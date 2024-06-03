from pathlib import Path
from funasr_torch import Paraformer

model_dir = "damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"
model = Paraformer(model_dir, batch_size=1)  # cpu
# model = Paraformer(model_dir, batch_size=1, device_id=0)  # gpu

wav_path = "{}/.cache/modelscope/hub/{}/example/asr_example.wav".format(Path.home(), model_dir)
hotwords = "你的热词 魔搭"

result = model(wav_path, hotwords)
print(result)
