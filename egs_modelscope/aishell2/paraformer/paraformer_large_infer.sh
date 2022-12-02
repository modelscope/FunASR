#!/usr/bin/env bash

set -e
set -u
set -o pipefail

ori_data=
data_dir=
exp_dir=
model_name=speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
model_revision="v1.0.3"     # please do not modify the model revision
inference_nj=32
gpuid_list="0,1" # set gpus, e.g., gpuid_list="0,1"
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')
njob=4  # the number of jobs for each gpu
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding

if ${gpu_inference}; then
    inference_nj=$[${ngpu}*${njob}]
else
    inference_nj=$njob
fi

# LM configs
use_lm=false
beam_size=1
lm_weight=0.0

test_sets="dev_ios test_android test_ios test_mic"

. utils/parse_options.sh

for x in Android iOS Mic; do
    local/aishell2_data_prep.sh ${ori_data}/${x}/dev ${data_dir}/aishell2/local/dev_${x,,} ${data_dir}/aishell2/dev_${x,,} || exit 1;
    local/aishell2_data_prep.sh ${ori_data}/${x}/test ${data_dir}/aishell2/local/test_${x,,} ${data_dir}/aishell2/test_${x,,} || exit 1;
done
for x in dev_android dev_ios dev_mic test_android test_ios test_mic; do
    mv ${data_dir}/aishell2/${x}/text ${data_dir}/aishell2/${x}/text.org
    paste -d " " <(cut -f 1 ${data_dir}/aishell2/${x}/text.org) <(cut -f 2- ${data_dir}/aishell2/${x}/text.org \
        | tr 'A-Z' 'a-z' | tr -d " ") \
       > ${data_dir}/aishell2/${x}/text
    rm ${data_dir}/aishell2/${x}/text.org
done

mkdir -p ${exp_dir}/aishell2

modelscope_utils/modelscope_infer.sh \
        --data_dir ${data_dir}/aishell2 \
        --exp_dir ${exp_dir}/aishell2 \
        --test_sets "${test_sets}" \
        --model_name ${model_name} \
        --model_revision ${model_revision} \
        --inference_nj ${inference_nj} \
        --gpuid_list ${gpuid_list} \
        --njob ${njob} \
        --gpu_inference ${gpu_inference} \
        --use_lm ${use_lm} \
        --beam_size ${beam_size} \
        --lm_weight ${lm_weight}
