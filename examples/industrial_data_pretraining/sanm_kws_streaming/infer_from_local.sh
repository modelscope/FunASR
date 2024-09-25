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
local_path=${local_path_root}/speech_sanm_kws_phone-xiaoyun-commands-online
git clone https://www.modelscope.cn/iic/speech_sanm_kws_phone-xiaoyun-commands-online.git ${local_path}

device="cpu" # "cuda:0" for gpu0, "cuda:1" for gpu1, "cpu"

config="inference_sanm_6e_320_256_fdim40_t2602_online.yaml"
tokens="${local_path}/tokens_2602.txt"
seg_dict="${local_path}/lexicon.txt"
init_param="${local_path}/finetune_sanm_6e_320_256_fdim40_t2602_online_xiaoyun_commands.pt"
cmvn_file="${local_path}/am.mvn.dim40_l3r3"

keywords=(小云小云)
keywords_string=$(IFS=,; echo "${keywords[*]}")
echo "keywords: $keywords_string"

echo "inference sanm streaming with chunk_size=[4, 8, 4]"
python -m funasr.bin.inference \
--config-path "${local_path}/" \
--config-name "${config}" \
++init_param="${init_param}" \
++frontend_conf.cmvn_file="${cmvn_file}" \
++tokenizer_conf.token_list="${tokens}" \
++tokenizer_conf.seg_dict="${seg_dict}" \
++input="${input}" \
++output_dir="${output_dir}" \
++chunk_size='[4, 8, 4]' \
++encoder_chunk_look_back=0 \
++decoder_chunk_look_back=0 \
++device="${device}" \
++keywords="\"$keywords_string"\"


echo "inference sanm streaming with chunk_size=[5, 10, 5]"
python -m funasr.bin.inference \
--config-path "${local_path}/" \
--config-name "${config}" \
++init_param="${init_param}" \
++frontend_conf.cmvn_file="${cmvn_file}" \
++tokenizer_conf.token_list="${tokens}" \
++tokenizer_conf.seg_dict="${seg_dict}" \
++input="${input}" \
++output_dir="${output_dir}" \
++chunk_size='[5, 10, 5]' \
++encoder_chunk_look_back=0 \
++decoder_chunk_look_back=0 \
++device="${device}" \
++keywords="\"$keywords_string"\"
