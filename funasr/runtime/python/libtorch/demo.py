from funasr_torch import Paraformer

#model_dir = "/Users/shixian/code/funasr/export/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model_dir = "/Users/shixian/code/funasr/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model = Paraformer(model_dir, batch_size=2)

# when using paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch, you should set pred_bias=0
# plot_timestamp_to works only when using speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
# model_dir = "/Users/shixian/code/funasr/export/damo/speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch"
# model = Paraformer(model_dir, batch_size=2, pred_bias=0)

# when using paraformer-large-vad-punc model, you can set plot_timestamp_to="./xx.png" to get figure of alignment besides timestamps
# model_dir = "/Users/shixian/code/funasr/export/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
# model = Paraformer(model_dir, batch_size=1)
# model = Paraformer(model_dir, batch_size=1, plot_timestamp_to="test.png")

wav_path = "YourPath/xx.wav"

result = model(wav_path)
print(result)