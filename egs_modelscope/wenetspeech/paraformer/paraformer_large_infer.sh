#!/usr/bin/env bash

set -e
set -u
set -o pipefail

ori_data=
data_dir=
exp_dir=
model_name=speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
model_revision="v1.0.4"     # please do not modify the model revision
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

test_sets="dev test_meeting test_net"

. utils/parse_options.sh

for tset_name in ${test_sets}; do
    test_dir=${data_dir}/wenetspeech/${tset_name}
    mkdir -p ${test_dir} 
    find ${ori_data}/${tset_name} -iname "*.wav" > ${test_dir}/wav.flist
    sed -e 's/\.wav//' ${test_dir}/wav.flist | awk -F '/' '{print $NF}' > ${test_dir}/utt.list
    paste -d' ' ${test_dir}/utt.list ${test_dir}/wav.flist > ${test_dir}/wav.scp
    cp ${ori_data}/${tset_name}/trans.txt ${test_dir}/text
    sed -i "s/\t/ /g" ${test_dir}/text
done

mkdir -p ${exp_dir}/wenetspeech

modelscope_utils/modelscope_infer.sh \
        --data_dir ${data_dir}/wenetspeech \
        --exp_dir ${exp_dir}/wenetspeech \
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

