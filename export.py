# method2, inference from local path
from funasr import AutoModel
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

model = AutoModel(
    model="/raid/t3cv/wangch/WORK_SAPCE/ASR/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
)

res = model.export(type="onnx", quantize=False, opset_version=13, device='cuda')  # fp32 onnx-gpu
# res = model.export(type="onnx_fp16", quantize=False, opset_version=13, device='cuda')  # fp16 onnx-gpu