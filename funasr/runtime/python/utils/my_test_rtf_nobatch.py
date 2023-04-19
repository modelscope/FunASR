
import time
import sys
import librosa
backend=sys.argv[1]
model_dir=sys.argv[2]
wav_file=sys.argv[3]

from funasr_torch import Paraformer
if backend == "onnxruntime":
    from rapid_paraformer import Paraformer
if 'blade' in model_dir:
    import torch_blade

# """
import torch
# tf32 = False
tf32 = True
torch.backends.cuda.matmul.allow_tf32 = tf32
torch.backends.cudnn.allow_tf32 = tf32
print(torch.backends.cuda.matmul.allow_tf32)
print(torch.backends.cudnn.allow_tf32)
# """

# model = Paraformer(model_dir, batch_size=1, device_id="-1")
model = Paraformer(model_dir, batch_size=1, device_id=0)  # gpu

wav_file_f = open(wav_file, 'r')
wav_files = wav_file_f.readlines()

# warm-up
print('TIME(warm-up): {}'.format(time.time()))
total = 0.0
num = 100
# wav_path = wav_files[0].split("\t")[1].strip() if "\t" in wav_files[0] else wav_files[0].split(" ")[1].strip()
# wav_path = '../../../export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav'
# NOTE(xw) warmup true data
for i in range(num):
    wav_path_i = wav_files[i % len(wav_files)]
    wav_path = wav_path_i.split("\t")[1].strip() if "\t" in wav_path_i else wav_path_i.split(" ")[1].strip()
    beg_time = time.time()
    result = model(wav_path)
    end_time = time.time()
    duration = end_time-beg_time
    total += duration
    print(result)
    print("num: {}, time, {}, avg: {}, rtf: {}".format(len(wav_path), duration, total/(i+1), (total/(i+1))/5.53))

# infer time
print('TIME(infer): {}'.format(time.time()))
beg_time = time.time()
for i, wav_path_i in enumerate(wav_files):
    wav_path = wav_path_i.split("\t")[1].strip() if "\t" in wav_path_i else wav_path_i.split(" ")[1].strip()
    # tic = time.time()
    result = model(wav_path)
    # print(time.time() - tic)
end_time = time.time()
duration = (end_time-beg_time)*1000
print("total_time_comput_ms: {}".format(int(duration)))

print('TIME(finish): {}'.format(time.time()))
duration_time = 0.0
for i, wav_path_i in enumerate(wav_files):
    wav_path = wav_path_i.split("\t")[1].strip() if "\t" in wav_path_i else wav_path_i.split(" ")[1].strip()
    waveform, _ = librosa.load(wav_path, sr=16000)
    duration_time += len(waveform)/16.0
print("total_time_wav_ms: {}".format(int(duration_time)))

print("total_rtf: {:.5}".format(duration/duration_time))
