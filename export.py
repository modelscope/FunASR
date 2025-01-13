# method2, inference from local path
from funasr import AutoModel

model = AutoModel(
    model="iic/emotion2vec_base",
    hub="ms"
)

res = model.export(type="onnx", quantize=False, opset_version=13, device='cpu')  # fp32 onnx-gpu
# res = model.export(type="onnx_fp16", quantize=False, opset_version=13, device='cuda')  # fp16 onnx-gpu
