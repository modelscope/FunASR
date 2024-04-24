import soundfile
from funasr_onnx.paraformer_online_bin import Paraformer
from pathlib import Path

model_dir = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
wav_path = ["{}/.cache/modelscope/hub/{}/example/asr_example.wav".format(Path.home(), model_dir)]

chunk_size = [5, 10, 5]
model = Paraformer(
    model_dir, batch_size=1, quantize=True, chunk_size=chunk_size, intra_op_num_threads=4
)  # only support batch_size = 1

##online asr
speech, sample_rate = soundfile.read(wav_path)
speech_length = speech.shape[0]
sample_offset = 0
step = chunk_size[1] * 960
param_dict = {"cache": dict()}
final_result = ""
for sample_offset in range(0, speech_length, min(step, speech_length - sample_offset)):
    if sample_offset + step >= speech_length - 1:
        step = speech_length - sample_offset
        is_final = True
    else:
        is_final = False
    param_dict["is_final"] = is_final
    rec_result = model(audio_in=speech[sample_offset : sample_offset + step], param_dict=param_dict)
    if len(rec_result) > 0:
        final_result += rec_result[0]["preds"][0]
    print(rec_result)
print(final_result)
