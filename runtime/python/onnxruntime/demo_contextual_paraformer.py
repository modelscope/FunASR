from funasr_onnx import ContextualParaformer
from pathlib import Path

# export the contextual model with following command first (one line)
# python -m funasr.export.export_model 
# --model-name damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404 
# --export-dir ./export --type onnx --quantize false

model_dir = "./export/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"
model = ContextualParaformer(model_dir, batch_size=1)

wav_path = ['{}/.cache/modelscope/hub/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/example/asr_example.wav'.format(Path.home())]

hotwords = '随机热词 各种热词 魔搭 阿里巴巴'

result = model(wav_path, hotwords)
print(result)
