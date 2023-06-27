from pathlib import Path
import os
import argparse
from funasr.utils.types import str2bool

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--export-dir', type=str, required=True)
parser.add_argument('--type', type=str, default='onnx', help='["onnx", "torch"]')
parser.add_argument('--device', type=str, default='cpu', help='["cpu", "cuda"]')
parser.add_argument('--quantize', type=str2bool, default=False, help='export quantized model')
parser.add_argument('--fallback-num', type=int, default=0, help='amp fallback number')
parser.add_argument('--audio_in', type=str, default=None, help='["wav", "wav.scp"]')
parser.add_argument('--calib_num', type=int, default=200, help='calib max num')
args = parser.parse_args()

model_dir = args.model_name
if not Path(args.model_name).exists():
	from modelscope.hub.snapshot_download import snapshot_download
	try:
		model_dir = snapshot_download(args.model_name, cache_dir=args.export_dir)
	except:
		raise "model_dir must be model_name in modelscope or local path downloaded from modelscope, but is {}".format \
			(model_dir)

model_file = os.path.join(model_dir, 'model.onnx')
if args.quantize:
	model_file = os.path.join(model_dir, 'model_quant.onnx')
if not os.path.exists(model_file):
	print(".onnx is not exist, begin to export onnx")
	from funasr.export.export_model import ModelExport
	export_model = ModelExport(
		cache_dir=args.export_dir,
		onnx=True,
		device="cpu",
		quant=args.quantize,
	)
	export_model.export(model_dir)