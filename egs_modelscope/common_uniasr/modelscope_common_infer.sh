#!/usr/bin/env bash

set -e
set -u
set -o pipefail

model_name=speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-online  # pre-trained model, download from modelscope
model_revision="v1.0.0"     # please do not modify the model revision
data_dir=  # wav list, ${data_dir}/wav.scp
exp_dir="exp"
gpuid_list="0,1"
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')
njob=4
gpu_inference=true
decode_cmd=utils/run.pl

. utils/parse_options.sh

if ${gpu_inference}; then
    inference_nj=$[${ngpu}*${njob}]
    _ngpu=1
else
    inference_nj=${njob}
    _ngpu=0
fi

# LM configs
use_lm=false
beam_size=1
lm_weight=0.0

python modelscope_utils/download_model.py \
          --model_name ${model_name} --model_revision ${model_revision}

if [ -d ${exp_dir} ]; then
    echo "${exp_dir} is already exists. if you want to decode again, please delete ${exp_dir} first."
    exit 1
else
    mkdir -p ${exp_dir}/${model_name}
    cp ${HOME}/.cache/modelscope/hub/damo/${model_name}/* ${exp_dir}/${model_name}/. -r
    _dir=${exp_dir}/decode_asr
    _logdir=${_dir}/logdir
    mkdir -p "${_dir}"
    mkdir -p "${_logdir}"
fi

for n in $(seq "${inference_nj}"); do
    split_scps+=" ${_logdir}/keys.${n}.scp"
done
# shellcheck disable=SC2086
utils/split_scp.pl "${data_dir}/wav.scp" ${split_scps}

if "${use_lm}"; then
    cp ${exp_dir}/${model_name}/decoding.yaml ${exp_dir}/${model_name}/decoding.yaml.back
    sed -i "s#beam_size: [0-9]*#beam_size: `echo $beam_size`#g" ${exp_dir}/${model_name}/decoding.yaml
    sed -i "s#lm_weight: 0.[0-9]*#lm_weight: `echo $lm_weight`#g" ${exp_dir}/${model_name}/decoding.yaml
fi

echo "Decoding started... log: '${_logdir}/asr_inference.*.log'"
# shellcheck disable=SC2086
${decode_cmd} --max-jobs-run "${inference_nj}" JOB=1:"${inference_nj}" "${_logdir}"/asr_inference.JOB.log \
    python -m funasr.bin.modelscope_infer \
          --local_model_path ${exp_dir}/${model_name} \
          --wav_list ${_logdir}/keys.JOB.scp \
          --output_file ${_logdir}/text.JOB \
          --gpuid_list ${gpuid_list} \
          --njob ${njob} \
          --ngpu ${_ngpu} \

    for i in $(seq ${inference_nj}); do
        cat ${_logdir}/text.${i}
    done | sort -k1 >${_dir}/text

if "${use_lm}"; then
    mv ${exp_dir}/${model_name}/decoding.yaml.back ${exp_dir}/${model_name}/decoding.yaml
fi
