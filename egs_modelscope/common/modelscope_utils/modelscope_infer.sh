#!/usr/bin/env bash

set -e
set -u
set -o pipefail

data_dir=
exp_dir=
model_name=
model_revision=
inference_nj=32
gpuid_list="0,1,2,3"
njob=32
gpu_inference=true

test_sets="dev test"
decode_cmd=utils/run.pl

# LM configs
use_lm=false
beam_size=1
lm_weight=0.0

. utils/parse_options.sh

if ${gpu_inference}; then
    _ngpu=1
else
    _ngpu=0
fi

# download model from modelscope
python modelscope_utils/download_model.py \
          --model_name ${model_name} --model_revision ${model_revision}

modelscope_dir=${HOME}/.cache/modelscope/hub/damo/${model_name}


for dset in ${test_sets}; do
    _dir=${exp_dir}/${model_name}/decode_asr/${dset}
    _logdir=${_dir}/logdir
    _data=${data_dir}/${dset}
    if [ -d ${_dir} ]; then
        echo "${_dir} is already exists. if you want to decode again, please delete ${_dir} first."
        exit 1
    else
        mkdir -p "${_dir}"
        mkdir -p "${_logdir}"
    fi

    if "${use_lm}"; then
        cp ${modelscope_dir}/decoding.yaml ${modelscope_dir}/decoding.yaml.back
        sed -i "s#beam_size: [0-9]*#beam_size: `echo $beam_size`#g" ${modelscope_dir}/decoding.yaml
        sed -i "s#lm_weight: 0.[0-9]*#lm_weight: `echo $lm_weight`#g" ${modelscope_dir}/decoding.yaml
    fi

    split_scps=
    for n in $(seq "${inference_nj}"); do
        split_scps+=" ${_logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${data_dir}/${dset}/wav.scp" ${split_scps}

    echo "Decoding started... log: '${_logdir}/asr_inference.*.log'"
    # shellcheck disable=SC2086
    ${decode_cmd} --max-jobs-run "${inference_nj}" JOB=1:"${inference_nj}" "${_logdir}"/asr_inference.JOB.log \
        python -m funasr.bin.modelscope_infer \
              --model_name ${model_name} \
              --model_revision ${model_revision} \
              --wav_list ${_logdir}/keys.JOB.scp \
              --output_file ${_logdir}/text.JOB \
              --gpuid_list ${gpuid_list} \
              --njob ${njob} \
              --ngpu ${_ngpu} \

        for i in $(seq ${inference_nj}); do
            cat ${_logdir}/text.${i}
        done | sort -k1 >${_dir}/text

        python utils/proce_text.py ${_dir}/text ${_dir}/text.proc
        python utils/proce_text.py ${_data}/text ${_data}/text.proc
        python utils/compute_wer.py ${_data}/text.proc ${_dir}/text.proc ${_dir}/text.cer
        tail -n 3 ${_dir}/text.cer > ${_dir}/text.cer.txt
        cat ${_dir}/text.cer.txt
done

if "${use_lm}"; then
    mv ${modelscope_dir}/decoding.yaml.back ${modelscope_dir}/decoding.yaml
fi
