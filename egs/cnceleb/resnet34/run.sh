#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
gpu_devices="6,7"
gpu_num=2
count=1
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=1
train_cmd=utils/run.pl
infer_cmd=utils/run.pl

# general configuration
feats_dir="." #feature output dictionary
exp_dir="."
lang=zh
dumpdir=dump/fbank
feats_type=fbank
token_type=spk
scp=feats.scp
type=kaldi_ark
stage=0
stop_stage=4

# feature configuration
feats_dim=80
sample_frequency=16000
nj=32
speed_perturb="0.9,1.0,1.1"

# data
data_cnceleb=

# exp tag
tag=""
inference_tag="basic"
inference_sv_model=sv.pb
sv_config=sv.yaml

. utils/parse_options.sh || exit 1;

model_dir="baseline_$(basename "${sv_config}" .yaml)_${feats_type}_${lang}_${token_type}_${tag}"

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="eval_enroll eval_test"

# you can set gpu num for decoding here
gpuid_list=$gpu_devices  # set gpus for decoding, the same as training stage by default
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')

if ${gpu_inference}; then
    inference_nj=$[${ngpu}*${njob}]
    _ngpu=1
else
    inference_nj=$njob
    _ngpu=0
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    # Data preparation
    bash local/make_cnceleb1.sh ${data_cnceleb}/CN-Celeb1 ${feats_dir}/data
    # bash local/make_cnceleb2.sh ${data_cnceleb}/CN-Celeb2 ${feats_dir}/data/cnceleb2_train
    grep speech ${feats_dir}/data/eval_test/trials/trials.lst > ${feats_dir}/data/eval_test/trials/trials.lst.speech
    # local/combine_data.sh ${feats_dir}/data/train ${feats_dir}/data/cnceleb1_train ${feats_dir}/data/cnceleb2_train
fi

# Testing Stage
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Inference"
    for dset in ${test_sets}; do
        echo "extracting embedding for ${dset}"
        asr_exp=${exp_dir}/exp/${model_dir}
        _dir="${asr_exp}/${inference_tag}/${inference_sv_model}/${dset}"
        _logdir="${_dir}/logdir"
        if [ -d ${_dir} ]; then
            echo "${_dir} is already exists. if you want to decode again, please delete this dir first."
            exit 0
        fi
        mkdir -p "${_logdir}"
        _data="data/${dset}"
        key_file=${_data}/wav.scp
        num_scp_file="$(<${key_file} wc -l)"
        _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
        split_scps=
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/sv_inference.JOB.log \
            python -m funasr.bin.sv_inference \
              --gpuid_list ${gpu_devices} \
              --ngpu ${_ngpu} \
              --key_file "${_logdir}"/keys.JOB.scp \
              --data_path_and_name_and_type data/${dset}/wav.scp,speech,sound  \
              --allow_variable_data_keys true \
              --sv_train_config ${sv_config} \
              --sv_model_file ${inference_sv_model} \
              --output_dir "${_logdir}"/output.JOB \
              --num_workers 1

        for f in xvector.ark; do
            if [ -f "${_logdir}/output.1/${f}" ]; then
                for i in $(seq "${_nj}"); do
                    echo "${_logdir}/output.${i}/${f}"
                done > "${_dir}/${f}.flist"
            fi
        done
    done

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

    # compute eer and minDCF results
    echo "stage 2: computing EER and minDCF"
    asr_exp=${exp_dir}/exp/${model_dir}
    _dir="${asr_exp}/${inference_tag}/${inference_sv_model}"
    mkdir -p ${_dir}/score
    cp ${_dir}/eval_enroll/xvector.ark.flist ${_dir}/score/spk2xvec.flist
    cp ${_dir}/eval_test/xvector.ark.flist ${_dir}/score/utt2xvec.flist
    cp ${feats_dir}/data/eval_test/trials/trials.lst.speech ${_dir}/score/trials
    python sid/calc_trial_scores.py --no_pbar ${_dir}/score ${_dir}/score/trials ${_dir}/score/trials.cos
    python sid/compute_eer.py ${_dir}/score/trials ${_dir}/score/trials.cos
    python sid/compute_min_dcf.py --c-miss 10 --p-target 0.01 \
      ${_dir}/score/trials.cos ${_dir}/score/trials 2>/dev/null
fi