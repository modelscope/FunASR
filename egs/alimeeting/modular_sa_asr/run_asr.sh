#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
CUDA_VISIBLE_DEVICES="4,5,6,7"
gpu_num=4
count=1
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
finetune=true
# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=2
train_cmd=utils/run.pl
infer_cmd=utils/run.pl

# general configuration
feats_dir="data" #feature output dictionary
exp_dir="."
lang=zh
token_type=char
type=sound
scp=wav.scp
speed_perturb="1.0"
stage=0
stop_stage=1

# feature configuration
feats_dim=80
nj=64

# exp tag
tag="finetune"

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=Train_Ali_far_wpegss
valid_set=Test_Ali_far_wpegss
test_sets="${DATA_NAME}_wpegss"

asr_config=conf/train_paraformer.yaml
model_dir="$(basename "${asr_config}" .yaml)_${lang}_${token_type}_${tag}"
pretrain_model_dir=./speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
inference_config=$pretrain_model_dir/decoding.yaml


token_list=$pretrain_model_dir/tokens.txt

# you can set gpu num for decoding here
gpuid_list=$CUDA_VISIBLE_DEVICES  # set gpus for decoding, the same as training stage by default
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')

if ${gpu_inference}; then
    inference_nj=$[${ngpu}*${njob}]
    _ngpu=1
else
    inference_nj=$njob
    _ngpu=0
fi

if ${finetune}; then
    inference_asr_model=./checkpoint/valid.acc.ave.pb
    finetune_tag="_finetune"
else
    inference_asr_model=$pretrain_model_dir/model.pb
    finetune_tag=""
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    if [ -L ./utils ]; then
        unlink ./utils
        ln -s ../../aishell/transformer/utils
    else
        ln -s ../../aishell/transformer/utils
    fi
fi

# Download Model
world_size=$gpu_num  # run on one machine
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Download Model"
    if [ ! -d $pretrain_model_dir ]; then
        git clone https://www.modelscope.cn/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git
    fi
fi

# ASR Training Stage
world_size=$gpu_num  # run on one machine
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    
    echo "stage 2: ASR Training"
    python  -m torch.distributed.launch \
     --nproc_per_node $gpu_num local/finetune.py

fi

# Testing Stage
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Inference"
    for dset in ${test_sets}; do
        _dir="$pretrain_model_dir/decode_${dset}${finetune_tag}"
        _logdir="${_dir}/logdir"
        if [ -d ${_dir} ]; then
            echo "${_dir} is already exists. if you want to decode again, please delete this dir first."
            exit 0
        fi
        mkdir -p "${_logdir}"
        _data="./data/${dset}"
        key_file=${_data}/${scp}
        num_scp_file="$(<${key_file} wc -l)"
        _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
        split_scps=
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}
        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi
        ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
            python -m funasr.bin.asr_inference_launch \
                --batch_size 1 \
                --ngpu "${_ngpu}" \
                --njob ${njob} \
                --gpuid_list ${gpuid_list} \
                --data_path_and_name_and_type "${_data}/${scp},speech,${type}" \
                --cmvn_file $pretrain_model_dir/am.mvn \
                --key_file "${_logdir}"/keys.JOB.scp \
                --asr_train_config $pretrain_model_dir/config.yaml \
                --asr_model_file $inference_asr_model \
                --output_dir "${_logdir}"/output.JOB \
                --mode paraformer \
                ${_opts}

        for f in token token_int score text; do
            if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                done | sort -k1 >"${_dir}/${f}"
            fi
        done
        python local/merge_spk_text.py ${_dir}/text ${_data}/utt2spk
        python local/compute_cpcer.py ${_data}/text_merge ${_dir}/text_merge
        echo "cpCER is saved at ${_dir}/text_cpcer"
    done
fi

