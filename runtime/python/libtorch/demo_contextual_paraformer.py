import torch
from pathlib import Path
from funasr_torch.paraformer_bin import ContextualParaformer

model_dir = "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"
device_id = 0 if torch.cuda.is_available() else -1
model = ContextualParaformer(model_dir, batch_size=1, device_id=device_id)  # gpu

wav_path = "{}/.cache/modelscope/hub/{}/example/asr_example.wav".format(Path.home(), model_dir)
hotwords = "你的热词 魔搭"

result = model(wav_path, hotwords)
print(result)
