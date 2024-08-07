# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# method2, inference from local model

# for more input type, please ref to readme.md
input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"

output_dir="./outputs/debug"

workspace=`pwd`

# download model
local_path_root=${workspace}/modelscope_models
mkdir -p ${local_path_root}
local_path=${local_path_root}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
git lfs clone https://www.modelscope.cn/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git ${local_path}

device="cuda:0" # "cuda:0" for gpu0, "cuda:1" for gpu1, "cpu"

tokens="${local_path}/tokens.json"
cmvn_file="${local_path}/am.mvn"

config="config.yaml"
init_param="${local_path}/model.pt"

python -m funasr.bin.inference \
--config-path "${local_path}" \
--config-name "${config}" \
++init_param="${init_param}" \
++tokenizer_conf.token_list="${tokens}" \
++frontend_conf.cmvn_file="${cmvn_file}" \
++input="${input}" \
++output_dir="${output_dir}" \
++device="${device}" \




