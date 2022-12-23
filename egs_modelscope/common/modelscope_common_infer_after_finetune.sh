#!/usr/bin/env bash

set -e
set -u
set -o pipefail

pretrained_model_name=speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch  # pre-trained model, download from modelscope
data_dir=  # wav list, ${data_dir}/wav.scp
finetune_model_name=  # fine-tuning model name
finetune_exp_dir=  # fine-tuning model experiment result path
gpuid_list="0"
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')
njob=1
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

if [ ! -d ${HOME}/.cache/modelscope/hub/damo/${pretrained_model_name} ]; then
    echo "${HOME}/.cache/modelscope/hub/damo/${pretrained_model_name} must exist."
    exit 1
else
    exp_dir=${finetune_exp_dir}/${finetune_model_name}.modelscope
    mkdir -p $exp_dir
    cp ${finetune_exp_dir}/${finetune_model_name} ${exp_dir}/${finetune_model_name}.modelscope
    cp ${HOME}/.cache/modelscope/hub/damo/${pretrained_model_name}/* ${exp_dir}/. -r
fi

_dir=${exp_dir}/decode_asr
_logdir=${_dir}/logdir
if [ -d ${_dir} ]; then
    echo "${_dir} is already exists. if you want to decode again, please delete ${_dir} first."
else
    mkdir -p "${_dir}"
    mkdir -p "${_logdir}"
fi

for n in $(seq "${inference_nj}"); do
    split_scps+=" ${_logdir}/keys.${n}.scp"
done
# shellcheck disable=SC2086
utils/split_scp.pl "${data_dir}/wav.scp" ${split_scps}

echo "Decoding started... log: '${_logdir}/asr_inference.*.log'"
# shellcheck disable=SC2086
${decode_cmd} --max-jobs-run "${inference_nj}" JOB=1:"${inference_nj}" "${_logdir}"/asr_inference.JOB.log \
    python -m funasr.bin.modelscope_infer \
          --local_model_path ${exp_dir} \
          --wav_list ${_logdir}/keys.JOB.scp \
          --output_file ${_logdir}/text.JOB \
          --gpuid_list ${gpuid_list} \
          --njob ${njob} \
          --ngpu ${_ngpu} \

    for i in $(seq ${inference_nj}); do
        cat ${_logdir}/text.${i}
    done | sort -k1 >${_dir}/text
