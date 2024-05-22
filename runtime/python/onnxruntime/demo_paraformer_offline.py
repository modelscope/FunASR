from funasr_onnx import Paraformer
from pathlib import Path

model_dir = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
# model_dir = "damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model = Paraformer(model_dir, batch_size=1, quantize=False)
# model = Paraformer(model_dir, batch_size=1, device_id=0)  # gpu

# when using paraformer-large-vad-punc model, you can set plot_timestamp_to="./xx.png" to get figure of alignment besides timestamps
# model = Paraformer(model_dir, batch_size=1, plot_timestamp_to="test.png")

wav_path = ["{}/.cache/modelscope/hub/{}/example/asr_example.wav".format(Path.home(), model_dir)]

result = model(wav_path)
print(result)
