from funasr_onnx import ContextualParaformer
from pathlib import Path

model_dir = "./export/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"
model = ContextualParaformer(model_dir, batch_size=1)

wav_path = ['{}/.cache/modelscope/hub/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/example/asr_example.wav'.format(Path.home())]
hotwords = '随机热词 各种热词 魔搭 阿里巴巴 仏'

result = model(wav_path, hotwords)
print(result)
