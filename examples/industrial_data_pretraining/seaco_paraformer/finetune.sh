# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)


# which gpu to train or finetune
export CUDA_VISIBLE_DEVICES="0,1"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# model_name from model_hub, or model_dir in local path

## option 1, download model automatically
model_name_or_model_dir="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

## option 2, download model by git
#local_path_root=${workspace}/modelscope_models
#mkdir -p ${local_path_root}/${model_name_or_model_dir}
#git clone https://www.modelscope.cn/${model_name_or_model_dir}.git ${local_path_root}/${model_name_or_model_dir}
#model_name_or_model_dir=${local_path_root}/${model_name_or_model_dir}


# data dir, which contains: train.json, val.json
data_dir="../../../data/list"

train_data="${data_dir}/train.jsonl"
val_data="${data_dir}/val.jsonl"

# generate train.jsonl and val.jsonl from wav.scp and text.txt
scp2jsonl \
++scp_file_list='["../../../data/list/train_wav.scp", "../../../data/list/train_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="${train_data}"

scp2jsonl \
++scp_file_list='["../../../data/list/val_wav.scp", "../../../data/list/val_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="${val_data}"


# exp output dir
output_dir="./outputs"
log_file="${output_dir}/log.txt"


mkdir -p ${output_dir}
echo "log_file: ${log_file}"

torchrun \
--nnodes 1 \
--nproc_per_node ${gpu_num} \
../../../funasr/bin/train.py \
++model="${model_name_or_model_dir}" \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++dataset_conf.batch_size=20000 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=4 \
++train_conf.max_epoch=50 \
++train_conf.log_interval=1 \
++train_conf.resume=false \
++train_conf.validate_interval=2000 \
++train_conf.save_checkpoint_interval=2000 \
++train_conf.avg_keep_nbest_models_type='loss' \
++train_conf.keep_nbest_models=20 \
++optim_conf.lr=0.0002 \
++output_dir="${output_dir}" &> ${log_file}