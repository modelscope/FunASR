# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)


# which gpu to train or finetune
export CUDA_VISIBLE_DEVICES="0"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# data dir, which contains: train.json, val.json, tokens.jsonl/tokens.txt, am.mvn
#data_dir="/Users/zhifu/funasr1.0/data/list"

## generate jsonl from wav.scp and text.txt
#python -m funasr.datasets.audio_datasets.scp2jsonl \
#++scp_file_list='["/Users/zhifu/funasr1.0/test_local/wav.scp", "/Users/zhifu/funasr1.0/test_local/text.txt"]' \
#++data_type_list='["source", "target"]' \
#++jsonl_file_out=/Users/zhifu/funasr1.0/test_local/audio_datasets.jsonl

train_data="/nfs/maziyang.mzy/data/librispeech/librispeech_train_960h.jsonl"
val_data="/nfs/maziyang.mzy/data/librispeech/librispeech_dev_other_filtered.jsonl"

# exp output dir
output_dir="/nfs/zhifu.gzf/ckpt/exp/llm_asr_whisper_vicuna_exp1"
log_file="${output_dir}/log.txt"

workspace=`pwd`
config="whisper_vicuna_linear.yaml"

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
++dataset_conf.batch_size=4 \
++dataset_conf.num_workers=4 \
++train_conf.max_epoch=15 \
++optim_conf.lr=0.0001 \
++init_param="${init_param}" \
++output_dir="${output_dir}" &> ${log_file} &
