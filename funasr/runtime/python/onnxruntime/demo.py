
from funasr_onnx import Paraformer

#model_dir = "/Users/shixian/code/funasr/export/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
#model_dir = "/Users/shixian/code/funasr/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model_dir = "/Users/shixian/code/funasr/export/damo/speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch"

# if you use paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch, you should set pred_bias=0
# plot_timestamp_to works only when using speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
model = Paraformer(model_dir, batch_size=2, plot_timestamp_to="./", pred_bias=0) 

wav_path = "/Users/shixian/code/funasr/export/damo/speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch/example/asr_example.wav"

result = model(wav_path)
print(result)