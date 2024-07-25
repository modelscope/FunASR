# from funasr import AutoModel
# model = AutoModel(model="/workspace/models/damo/speech_campplus_sv_zh-cn_16k-common")
# res = model.export(quantize=True)



# # pip3 install -U funasr-onnx
# from funasr_onnx import Paraformer
# model_dir = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
# model = Paraformer(model_dir, batch_size=1, quantize=True)

# wav_path = ['~/.cache/modelscope/hub/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav']

# result = model(wav_path)
# print(result)


# from funasr import AutoModel
# import torch
# import onnx
# from onnxruntime.quantization import QuantType, quantize_dynamic
# import os 
# model = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common",
#                   model_revision="v2.0.2",
#                   device="cpu"
#                   )

# print("load torch Done!")
# dummy_input = torch.rand(1, 553,80)
# quant=True
# model=model.model
# model_script = model #torch.jit.trace(model)
# model_path = "camplus.onnx"
# input_names=["input"]
# output_names=["output"]
# verbose=True
# torch.onnx.export(
#     model_script,
#     dummy_input,
#     model_path,
#     verbose=verbose,
#     opset_version=14,
#     input_names=input_names,
#     output_names=output_names,
#     dynamic_axes={'input': [0, 1]}
# )
# print("export Done!")
# #quant
# if quant:
#     quant_model_path = "camplus_quant.onnx"
#     if not os.path.exists(quant_model_path):
#         onnx_model = onnx.load(model_path)
#         quantize_dynamic(
#             model_input=model_path,
#             model_output=quant_model_path, 
#             weight_type=QuantType.QUInt8,
#             )
# print("quant Done!")