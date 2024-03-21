# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# method2, finetune from local model

workspace=`pwd`

echo "current path: ${workspace}" # /xxxx/funasr/examples/industrial_data_pretraining/paraformer

# download model
local_path_root=${workspace}/modelscope_models
mkdir -p ${local_path_root}
local_path=${local_path_root}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
git clone https://www.modelscope.cn/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git ${local_path}


# which gpu to train or finetune
export CUDA_VISIBLE_DEVICES="0,1"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# data dir, which contains: train.json, val.json
data_dir="../../../data/list"

train_data="${data_dir}/train.jsonl"
val_data="${data_dir}/val.jsonl"


# generate train.jsonl and val.jsonl from wav.scp and text.txt
python -m funasr.datasets.audio_datasets.scp2jsonl \
++scp_file_list='["../../../data/list/train_wav.scp", "../../../data/list/train_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="${train_data}"

python -m funasr.datasets.audio_datasets.scp2jsonl \
++scp_file_list='["../../../data/list/val_wav.scp", "../../../data/list/val_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="${val_data}"


tokens="${local_path}/tokens.json"
cmvn_file="${local_path}/am.mvn"

# output dir
output_dir="./outputs"
log_file="${output_dir}/log.txt"

config_name="config.yaml"

init_param="${local_path}/model.pt"

mkdir -p ${output_dir}
echo "log_file: ${log_file}"

torchrun \
--nnodes 1 \
--nproc_per_node ${gpu_num} \
../../../funasr/bin/train.py \
--config-path "${local_path}" \
--config-name "${config_name}" \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++dataset_conf.batch_size=20000 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=4 \
++train_conf.max_epoch=50 \
++train_conf.log_interval=10 \
++train_conf.resume=false \
++train_conf.validate_interval=15 \
++train_conf.save_checkpoint_interval=15 \
++train_conf.keep_nbest_models=50 \
++optim_conf.lr=0.0002 \
++init_param="${init_param}" \
++tokenizer_conf.token_list="${tokens}" \
++frontend_conf.cmvn_file="${cmvn_file}" \
++output_dir="${output_dir}" &> ${log_file}
