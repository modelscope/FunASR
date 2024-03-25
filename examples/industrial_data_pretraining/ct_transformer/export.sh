# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# method1, inference from model hub
export HYDRA_FULL_ERROR=1


model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"


python -m funasr.bin.export \
++model=${model} \
++model_revision=${model_revision} \
++type="onnx" \
++quantize=false \
++device="cpu"


# method2, inference from local path
model="/Users/zhifu/.cache/modelscope/hub/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

python -m funasr.bin.export \
++model=${model} \
++type="onnx" \
++quantize=false \
++device="cpu"