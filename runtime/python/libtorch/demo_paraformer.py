from pathlib import Path
from funasr_torch.paraformer_bin import Paraformer

model_dir = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model = Paraformer(model_dir, batch_size=1)  # cpu
# model = Paraformer(model_dir, batch_size=1, device_id=0)  # gpu

wav_path = "{}/.cache/modelscope/hub/{}/example/asr_example.wav".format(Path.home(), model_dir)

result = model(wav_path)
print(result)
