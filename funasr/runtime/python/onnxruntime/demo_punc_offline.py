from funasr_onnx import TargetDelayTransformer

model_dir = "/disk1/mengzhe.cmz/workspace/FunASR/funasr/export/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
model = TargetDelayTransformer(model_dir)

text_in = "我们都是木头人不会讲话不会动"

result = model(text_in)
print(result)
