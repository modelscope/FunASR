
import time
import sys
import librosa
backend = sys.argv[1]
model_dir = sys.argv[2]
wav_file = sys.argv[3]
batch_size = int(sys.argv[4])
res_file = sys.argv[5] if len(sys.argv) > 5 else None

from funasr_torch import Paraformer
if backend == "onnxruntime":
    from funasr_onnx import Paraformer
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

# model = Paraformer(model_dir, batch_size=batch_size, device_id="-1")
model = Paraformer(model_dir, batch_size=batch_size, device_id=0)  # gpu

wav_file_f = open(wav_file, 'r')
wav_files = wav_file_f.readlines()

def strip(x):
    sep = '\t' if '\t' in x else ' '
    return x.split(sep)[1].strip()

wav_files = [strip(i) for i in wav_files]

# warm-up
print('TIME(warm-up): {}'.format(time.time()))
num = 100
warm_wav_files = [wav_files[i % len(wav_files)] for i in range(0, num * batch_size)]
beg_time = time.time()
result = model(warm_wav_files)
end_time = time.time()
duration = end_time-beg_time
# print(result)
print("num: {}, batch_size: {}, time, {}, avg: {}".format(len(warm_wav_files), batch_size, duration, duration/len(warm_wav_files)))

# infer time
print('TIME(infer): {}'.format(time.time()))
beg_time = time.time()
result = model(wav_files)
end_time = time.time()
duration = (end_time-beg_time)*1000
print("total_time_comput_ms: {}".format(int(duration)))

if res_file:
    with open(res_file, 'w') as f:
        for wav_file, res in zip(wav_files, result):
            name = wav_file.split('/')[-1][:-4]
            pred = res['preds'][0]
            f.write('{} {}\n'.format(name, pred))

print('TIME(finish): {}'.format(time.time()))
duration_time = 0.0
for wav_path in wav_files:
    waveform, _ = librosa.load(wav_path, sr=16000)
    duration_time += len(waveform)/16.0
print("total_time_wav_ms: {}".format(int(duration_time)))

print("total_rtf: {:.5}".format(duration/duration_time))
