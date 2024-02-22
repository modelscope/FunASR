# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)


# which gpu to train or finetune
export CUDA_VISIBLE_DEVICES="0,1"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# data dir, which contains: train.json, val.json, tokens.jsonl/tokens.txt, am.mvn
data_dir="/Users/zhifu/funasr1.0/data/list"

## generate jsonl from wav.scp and text.txt
#python -m funasr.datasets.audio_datasets.scp2jsonl \
#++scp_file_list='["/Users/zhifu/funasr1.0/test_local/wav.scp", "/Users/zhifu/funasr1.0/test_local/text.txt"]' \
#++data_type_list='["source", "target"]' \
#++jsonl_file_out=/Users/zhifu/funasr1.0/test_local/audio_datasets.jsonl

train_data="${data_dir}/train.jsonl"
val_data="${data_dir}/val.jsonl"
tokens="${data_dir}/tokens.json"
cmvn_file="${data_dir}/am.mvn"

# exp output dir
output_dir="/Users/zhifu/exp"
log_file="${output_dir}/log.txt"

workspace=`pwd`
config="paraformer_conformer_12e_6d_2048_256.yaml"

init_param="${output_dir}/model.pt"

mkdir -p ${output_dir}
echo "log_file: ${log_file}"

torchrun \
--nnodes 1 \
--nproc_per_node ${gpu_num} \
../../../funasr/bin/train.py \
--config-path "${workspace}/conf" \
--config-name "${config}" \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++tokenizer_conf.token_list="${tokens}" \
++frontend_conf.cmvn_file="${cmvn_file}" \
++dataset_conf.batch_size=32 \
++dataset_conf.batch_type="example" \
++dataset_conf.num_workers=4 \
++train_conf.max_epoch=150 \
++optim_conf.lr=0.0002 \
++init_param="${init_param}" \
++output_dir="${output_dir}" &> ${log_file}
