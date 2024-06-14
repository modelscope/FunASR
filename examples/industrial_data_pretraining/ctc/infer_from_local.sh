# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# method2, inference from local model

# for more input type, please ref to readme.md
model_dir=$1
input_file=$2
output_dir=$3

# download model
device="cuda:0" # "cuda:0" for gpu0, "cuda:1" for gpu1, "cpu"

tokens="${model_dir}/tokens.json"
cmvn_file="${model_dir}/am.mvn"

config="config.yaml"
init_param="${model_dir}/model.pt"

mkdir -p ${output_dir}

python -m funasr.bin.inference \
--config-path "${model_dir}" \
--config-name "${config}" \
++init_param="${init_param}" \
++tokenizer_conf.token_list="${tokens}" \
++frontend_conf.cmvn_file="${cmvn_file}" \
++input="${input_file}" \
++output_dir="${output_dir}" \
++device="${device}" \

