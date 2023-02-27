#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
CUDA_VISIBLE_DEVICES="6,7"
gpu_num=2
count=1
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=5
train_cmd=utils/run.pl
infer_cmd=utils/run.pl

# general configuration
feats_dir="." #feature output dictionary
exp_dir="."
lang=zh
dumpdir=dump/raw
feats_type=raw
token_type=char
scp=wav.scp
type=kaldi_ark
stage=3
stop_stage=4

# feature configuration
feats_dim=
sample_frequency=16000
nj=32
speed_perturb=

# exp tag
tag="exp1"

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="dev test"

asr_config=conf/train_asr_conformer.yaml
model_dir="baseline_$(basename "${asr_config}" .yaml)_${feats_type}_${lang}_${token_type}_${tag}"

inference_config=conf/decode_asr_transformer.yaml
inference_asr_model=valid.acc.ave_10best.pth

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

feat_train_dir=${feats_dir}/${dumpdir}/train; mkdir -p ${feat_train_dir}
feat_dev_dir=${feats_dir}/${dumpdir}/dev; mkdir -p ${feat_dev_dir}
feat_test_dir=${feats_dir}/${dumpdir}/test; mkdir -p ${feat_test_dir}

# Training Stage
world_size=$gpu_num  # run on one machine
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Training"
    mkdir -p ${exp_dir}/exp/${model_dir}
    mkdir -p ${exp_dir}/exp/${model_dir}/log
    INIT_FILE=${exp_dir}/exp/${model_dir}/ddp_init
    if [ -f $INIT_FILE ];then
        rm -f $INIT_FILE
    fi 
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    for ((i = 0; i < $gpu_num; ++i)); do
        {
            rank=$i
            local_rank=$i
            gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
            asr_train.py \
                --gpu_id $gpu_id \
                --use_preprocessor true \
                --token_type char \
                --token_list $token_list \
                --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${train_set}/${scp},speech,${type} \
                --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${train_set}/text,text,text \
                --train_shape_file ${feats_dir}/asr_stats_fbank_zh_char/${train_set}/speech_shape \
                --train_shape_file ${feats_dir}/asr_stats_fbank_zh_char/${train_set}/text_shape.char \
                --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${valid_set}/${scp},speech,${type} \
                --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${valid_set}/text,text,text \
                --valid_shape_file ${feats_dir}/asr_stats_fbank_zh_char/${valid_set}/speech_shape \
                --valid_shape_file ${feats_dir}/asr_stats_fbank_zh_char/${valid_set}/text_shape.char  \
                --resume true \
                --output_dir ${exp_dir}/exp/${model_dir} \
                --config $asr_config \
                --input_size $feats_dim \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --multiprocessing_distributed true \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${exp_dir}/exp/${model_dir}/log/train.log.$i 2>&1
        } &
        done
        wait
fi

# Testing Stage
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Inference"
    for dset in ${test_sets}; do
        asr_exp=${exp_dir}/exp/${model_dir}
        inference_tag="$(basename "${inference_config}" .yaml)"
        _dir="${asr_exp}/${inference_tag}/${inference_asr_model}/${dset}"
        _logdir="${_dir}/logdir"
        if [ -d ${_dir} ]; then
            echo "${_dir} is already exists. if you want to decode again, please delete this dir first."
            exit 0
        fi
        mkdir -p "${_logdir}"
        _data="${feats_dir}/${dumpdir}/${dset}"
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
        ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1: "${_nj}" "${_logdir}"/asr_inference.JOB.log \
            python -m funasr.bin.asr_inference_launch \
                --batch_size 1 \
                --ngpu "${_ngpu}" \
                --njob ${njob} \
                --gpuid_list ${gpuid_list} \
                --data_path_and_name_and_type "${_data}/${scp},speech,${type}" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --asr_train_config "${asr_exp}"/config.yaml \
                --asr_model_file "${asr_exp}"/"${inference_asr_model}" \
                --output_dir "${_logdir}"/output.JOB \
                --mode asr \
                ${_opts}

        for f in token token_int score text; do
            if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                done | sort -k1 >"${_dir}/${f}"
            fi
        done
        python utils/proce_text.py ${_dir}/text ${_dir}/text.proc
        python utils/proce_text.py ${_data}/text ${_data}/text.proc
        python utils/compute_wer.py ${_data}/text.proc ${_dir}/text.proc ${_dir}/text.cer
        tail -n 3 ${_dir}/text.cer > ${_dir}/text.cer.txt
        cat ${_dir}/text.cer.txt
    done
fi

