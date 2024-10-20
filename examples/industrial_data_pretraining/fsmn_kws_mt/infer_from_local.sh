# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# method2, inference from local model

# for more input type, please ref to readme.md
input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testset/kws_xiaoyunxiaoyun.wav"

output_dir="./outputs/debug"

workspace=`pwd`

# download model
local_path_root=${workspace}/modelscope_models
mkdir -p ${local_path_root}
local_path=${local_path_root}/speech_charctc_kws_phone-xiaoyun_mt
git clone https://www.modelscope.cn/iic/speech_charctc_kws_phone-xiaoyun_mt.git ${local_path}

device="cuda:0" # "cuda:0" for gpu0, "cuda:1" for gpu1, "cpu"

config="inference_fsmn_4e_l10r2_280_200_fdim40_t2602_t4.yaml"
tokens="${local_path}/funasr/tokens_2602.txt"
tokens2="${local_path}/funasr/tokens_xiaoyun.txt"
seg_dict="${local_path}/funasr/lexicon.txt"
init_param="${local_path}/funasr/finetune_fsmn_4e_l10r2_280_200_fdim40_t2602_t4_xiaoyun_xiaoyun.pt"
cmvn_file="${local_path}/funasr/am.mvn.dim40_l4r4"

keywords=(小云小云)
keywords_string=$(IFS=,; echo "${keywords[*]}")
echo "keywords: $keywords_string"

python -m funasr.bin.inference \
--config-path "${local_path}/funasr" \
--config-name "${config}" \
++init_param="${init_param}" \
++frontend_conf.cmvn_file="${cmvn_file}" \
++token_lists='['''${tokens}''', '''${tokens2}''']' \
++seg_dicts='['''${seg_dict}''', '''${seg_dict}''']' \
++input="${input}" \
++output_dir="${output_dir}" \
++device="${device}" \
++keywords="\"$keywords_string"\"
