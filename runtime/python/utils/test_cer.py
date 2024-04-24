import os
import time
import sys
import librosa
from funasr.utils.types import str2bool

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--backend", type=str, default="onnx", help='["onnx", "torch"]')
parser.add_argument("--wav_file", type=str, default=None, help="amp fallback number")
parser.add_argument("--quantize", type=str2bool, default=False, help="quantized model")
parser.add_argument(
    "--intra_op_num_threads", type=int, default=1, help="intra_op_num_threads for onnx"
)
parser.add_argument("--output_dir", type=str, default=None, help="amp fallback number")
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

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if os.name == "nt":  # Windows
    newline = "\r\n"
else:  # Linux Mac
    newline = "\n"
text_f = open(os.path.join(output_dir, "text"), "w", newline=newline)
token_f = open(os.path.join(output_dir, "token"), "w", newline=newline)

for i, wav_path_i in enumerate(wav_files):
    wav_name, wav_path = wav_path_i.strip().split()
    result = model(wav_path)
    text_i = "{} {}\n".format(wav_name, result[0]["preds"][0])
    token_i = "{} {}\n".format(wav_name, result[0]["preds"][1])
    text_f.write(text_i)
    text_f.flush()
    token_f.write(token_i)
    token_f.flush()
text_f.close()
token_f.close()
