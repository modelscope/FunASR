# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# method1, inference from model hub
export HYDRA_FULL_ERROR=1

model="iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online"

python -m funasr.bin.export \
++model=${model} \
++type="onnx" \
++quantize=false \
++device="cpu"


## method2, inference from local path
#model="/Users/zhifu/.cache/modelscope/hub/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online"
#
#python -m funasr.bin.export \
#++model=${model} \
#++type="onnx" \
#++quantize=false \
#++device="cpu" \
#++debug=false
