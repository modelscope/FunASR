# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# To install requirements: pip3 install -U openai-whisper

# method2, inference from local model

# for more input type, please ref to readme.md
input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"

output_dir="./outputs/debug"

workspace=`pwd`

# download model
local_path_root=${workspace}/modelscope_models
mkdir -p ${local_path_root}
#Whisper-large-v2
#local_path=${local_path_root}/speech_whisper-large_asr_multilingual
#git clone https://www.modelscope.cn/iic/speech_whisper-large_asr_multilingual.git ${local_path}
#init_param="${local_path}/large-v2.pt"
#Whisper-large-v3
local_path=${local_path_root}/Whisper-large-v3
git clone https://www.modelscope.cn/iic/Whisper-large-v3.git ${local_path}
init_param="${local_path}/large-v3.pt"

device="cuda:0" # "cuda:0" for gpu0, "cuda:1" for gpu1, "cpu"

config="config.yaml"


python -m funasr.bin.inference \
--config-path "${local_path}" \
--config-name "${config}" \
++init_param="${init_param}" \
++input="${input}" \
++output_dir="${output_dir}" \
++device="${device}" \




