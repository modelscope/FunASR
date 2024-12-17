import time
import sys
import librosa
from funasr.utils.type_utils import str2bool

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--backend", type=str, default="onnx", help='["onnx", "torch"]')
parser.add_argument("--wav_file", type=str, default=None, help="amp fallback number")
parser.add_argument("--quantize", type=str2bool, default=False, help="quantized model")
parser.add_argument(
    "--intra_op_num_threads", type=int, default=1, help="intra_op_num_threads for onnx"
)
args = parser.parse_args()


from funasr.runtime.python.libtorch.funasr_torch import Paraformer

if args.backend == "onnx":
    from funasr.runtime.python.onnxruntime.funasr_onnx import Paraformer

model = Paraformer(
    args.model_dir,
    batch_size=1,
    quantize=args.quantize,
    intra_op_num_threads=args.intra_op_num_threads,
)

wav_file_f = open(args.wav_file, "r")
wav_files = wav_file_f.readlines()

# warm-up
total = 0.0
num = 30
wav_path = (
    wav_files[0].split("\t")[1].strip()
    if "\t" in wav_files[0]
    else wav_files[0].split(" ")[1].strip()
)
for i in range(num):
    beg_time = time.time()
    result = model(wav_path)
    end_time = time.time()
    duration = end_time - beg_time
    total += duration
    print(result)
    print(
        "num: {}, time, {}, avg: {}, rtf: {}".format(
            len(wav_path), duration, total / (i + 1), (total / (i + 1)) / 5.53
        )
    )

# infer time
beg_time = time.time()
for i, wav_path_i in enumerate(wav_files):
    wav_path = (
        wav_path_i.split("\t")[1].strip()
        if "\t" in wav_path_i
        else wav_path_i.split(" ")[1].strip()
    )
    result = model(wav_path)
end_time = time.time()
duration = (end_time - beg_time) * 1000
print("total_time_comput_ms: {}".format(int(duration)))

duration_time = 0.0
for i, wav_path_i in enumerate(wav_files):
    wav_path = (
        wav_path_i.split("\t")[1].strip()
        if "\t" in wav_path_i
        else wav_path_i.split(" ")[1].strip()
    )
    waveform, _ = librosa.load(wav_path, sr=16000)
    duration_time += len(waveform) / 16.0
print("total_time_wav_ms: {}".format(int(duration_time)))

print("total_rtf: {:.5}".format(duration / duration_time))
