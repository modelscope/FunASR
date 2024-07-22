#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES="0,1,2,3"

# general configuration
feats_dir="../DATA" #feature output dictionary
exp_dir=`pwd`
lang=zh
token_type=char
# feature configuration
nj=32

inference_device="cuda" #"cpu"
inference_checkpoint="model.pt.avg10"
inference_scp="wav.scp"
inference_batch_size=1

# exp tag
tag="WD-LoRA-FT2"
workspace=`pwd`

master_port=12345

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=WD/train
valid_set=WD/dev

config=paraformer_conformer_12e_6d_2048_256.yaml
model_dir="baseline_$(basename "${config}" .yaml)_${lang}_${token_type}_${tag}"
token_list=/ssd/zhuang/code/FunASR/examples/aishell/paraformer/exp/speech_paraformer_asr_nat-aishell1-pytorch/tokens.txt
cmvn_file=/ssd/zhuang/code/FunASR/examples/aishell/paraformer/exp/speech_paraformer_asr_nat-aishell1-pytorch/am.mvn

# ASR Training Stage
echo "stage 1: ASR Training"

mkdir -p ${exp_dir}/exp/${model_dir}
current_time=$(date "+%Y-%m-%d_%H-%M")
log_file="${exp_dir}/exp/${model_dir}/train.log.txt.${current_time}"
echo "log_file: ${log_file}"

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
torchrun \
--nnodes 1 \
--nproc_per_node ${gpu_num} \
--master_port ${master_port} \
../../../funasr/bin/train.py \
--config-path "${workspace}/conf" \
--config-name "${config}" \
++init_param=/ssd/zhuang/code/FunASR/examples/aishell/paraformer/exp/speech_paraformer_asr_nat-aishell1-pytorch/model.pb \
++train_data_set_list="${feats_dir}/data/${train_set}/audio_datasets.jsonl" \
++valid_data_set_list="${feats_dir}/data/${valid_set}/audio_datasets.jsonl" \
++tokenizer_conf.token_list="${token_list}" \
++frontend_conf.cmvn_file="${cmvn_file}" \
++use_lora=true \
++lora_details=/ssd/zhuang/code/FunASR/examples/kespeech/paraformer/conf_lora/config.json \
++lora_bias=lora_only \
++output_dir="${exp_dir}/exp/${model_dir}" &> ${log_file}

