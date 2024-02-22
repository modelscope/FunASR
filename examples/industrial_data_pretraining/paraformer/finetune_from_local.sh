# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# method2, finetune from local model

workspace=`pwd`

# download model
local_path_root=${workspace}/modelscope_models
mkdir -p ${local_path_root}
local_path=${local_path_root}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
git clone https://www.modelscope.cn/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git ${local_path}


# which gpu to train or finetune
export CUDA_VISIBLE_DEVICES="0,1"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# data dir, which contains: train.json, val.json
data_dir="/Users/zhifu/funasr1.0/data/list"

## generate jsonl from wav.scp and text.txt
#python -m funasr.datasets.audio_datasets.scp2jsonl \
#++scp_file_list='["/Users/zhifu/funasr1.0/test_local/wav.scp", "/Users/zhifu/funasr1.0/test_local/text.txt"]' \
#++data_type_list='["source", "target"]' \
#++jsonl_file_out=/Users/zhifu/funasr1.0/test_local/audio_datasets.jsonl

train_data="${data_dir}/train.jsonl"
val_data="${data_dir}/val.jsonl"

tokens="${local_path}/tokens.json"
cmvn_file="${local_path}/am.mvn"

# exp output dir
output_dir="/Users/zhifu/exp"
log_file="${output_dir}/log.txt"

config="config.yaml"

init_param="${local_path}/model.pt"

mkdir -p ${output_dir}
echo "log_file: ${log_file}"

torchrun \
--nnodes 1 \
--nproc_per_node ${gpu_num} \
../../../funasr/bin/train.py \
--config-path "${local_path}" \
--config-name "${config}" \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++tokenizer_conf.token_list="${tokens}" \
++frontend_conf.cmvn_file="${cmvn_file}" \
++dataset_conf.batch_size=32 \
++dataset_conf.batch_type="example" \
++dataset_conf.num_workers=4 \
++train_conf.max_epoch=20 \
++optim_conf.lr=0.0002 \
++init_param="${init_param}" \
++output_dir="${output_dir}" &> ${log_file}
