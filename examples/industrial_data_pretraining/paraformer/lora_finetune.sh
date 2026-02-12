#!/usr/bin/env bash
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

set -euo pipefail

workspace=$(pwd)

# which gpu to train or finetune
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
gpu_num=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F "," '{print NF}')

# data dir, which contains train.jsonl/val.jsonl
# NOTE: update these paths to your dataset jsonl files.
data_dir="${workspace}/data/list"
train_data="${data_dir}/train.jsonl"
val_data="${data_dir}/val.jsonl"

# config
config_path="${workspace}/examples/industrial_data_pretraining/paraformer/conf"
config_name="paraformer_lora.yaml"

# exp output dir
output_dir="${workspace}/examples/industrial_data_pretraining/paraformer/outputs_lora"
log_file="${output_dir}/log.txt"

mkdir -p "${output_dir}"

DISTRIBUTED_ARGS="
    --nnodes ${WORLD_SIZE:-1} \
    --nproc_per_node ${gpu_num} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-26669}
"

echo "log_file: ${log_file}"

torchrun ${DISTRIBUTED_ARGS} \
  funasr/bin/train_ds.py \
  --config-path "${config_path}" \
  --config-name "${config_name}" \
  ++train_data_set_list="${train_data}" \
  ++valid_data_set_list="${val_data}" \
  ++output_dir="${output_dir}" \
  &> "${log_file}"
