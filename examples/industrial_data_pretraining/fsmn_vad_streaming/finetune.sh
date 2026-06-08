#!/bin/bash
# Copyright FunASR (https://github.com/modelscope/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
#
# Fine-tune FSMN-VAD model on custom data.
# Usage: bash finetune.sh

workspace=`pwd`

export CUDA_VISIBLE_DEVICES="0"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# Model: choose 16k or 8k version
model_name_or_model_dir="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
# For 8k audio: model_name_or_model_dir="iic/speech_fsmn_vad_zh-cn-8k-common"

# Data directory
data_dir="../../../data/list"

train_data="${data_dir}/train.jsonl"
val_data="${data_dir}/val.jsonl"

# Generate train.jsonl from wav.scp and vad.txt
# wav.scp format:  utt_id /path/to/audio.wav
# vad.txt format:  utt_id [[start_ms, end_ms], [start_ms, end_ms], ...]
scp2jsonl \
++scp_file_list='["../../../data/list/train_wav.scp", "../../../data/list/train_vad.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="${train_data}"

scp2jsonl \
++scp_file_list='["../../../data/list/val_wav.scp", "../../../data/list/val_vad.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="${val_data}"

output_dir="./outputs"
log_file="${output_dir}/log.txt"
mkdir -p ${output_dir}

torchrun \
    --nnodes 1 \
    --nproc_per_node $gpu_num \
../../../funasr/bin/train_ds.py \
++model="${model_name_or_model_dir}" \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++dataset="AudioDataset" \
++dataset_conf.index_ds="IndexDSJsonl" \
++dataset_conf.data_split_num=1 \
++dataset_conf.batch_sampler="BatchSampler" \
++dataset_conf.batch_size=6000  \
++dataset_conf.sort_size=1024 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=4 \
++train_conf.max_epoch=20 \
++train_conf.log_interval=50 \
++train_conf.resume=true \
++train_conf.validate_interval=1000 \
++train_conf.save_checkpoint_interval=1000 \
++train_conf.keep_nbest_models=5 \
++train_conf.avg_nbest_model=3 \
++train_conf.use_deepspeed=false \
++optim_conf.lr=0.00005 \
++output_dir="${output_dir}" &> ${log_file}
