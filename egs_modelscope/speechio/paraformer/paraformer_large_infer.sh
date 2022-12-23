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
gpuid_list="0" # set gpus, e.g., gpuid_list="0,1"
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')
njob=1  # the number of jobs for each gpu
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

test_sets="SPEECHIO_ASR_ZH00001 SPEECHIO_ASR_ZH00002 SPEECHIO_ASR_ZH00003 SPEECHIO_ASR_ZH00004 SPEECHIO_ASR_ZH00005 SPEECHIO_ASR_ZH00006 SPEECHIO_ASR_ZH00007 SPEECHIO_ASR_ZH00008 SPEECHIO_ASR_ZH00009 SPEECHIO_ASR_ZH00010 SPEECHIO_ASR_ZH00011 SPEECHIO_ASR_ZH00012 SPEECHIO_ASR_ZH00013 SPEECHIO_ASR_ZH00014 SPEECHIO_ASR_ZH00015"

. utils/parse_options.sh

for tset_name in ${test_sets}; do
    test_dir=${data_dir}/speechio/${tset_name}
    mkdir -p ${test_dir}
    find ${ori_data}/${tset_name} -iname "*.wav" > ${test_dir}/wav.flist
    sed -e 's/\.wav//' ${test_dir}/wav.flist | awk -F '/' '{print $NF}' > ${test_dir}/utt.list
    paste -d' ' ${test_dir}/utt.list ${test_dir}/wav.flist > ${test_dir}/wav.scp
    cp ${ori_data}/${tset_name}/trans.txt ${test_dir}/text
    sed -i "s/\t/ /g" ${test_dir}/text
done

mkdir -p ${exp_dir}/speechio

modelscope_utils/modelscope_infer.sh \
        --data_dir ${data_dir}/speechio \
        --exp_dir ${exp_dir}/speechio \
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

#  SpeechIO TIOBE textnorm
for tset_name in ${test_sets}; do
    echo "$0 --> Normalizing REF text ..."
    ./utils/textnorm_zh.py \
        --has_key --to_upper \
        ${ori_data}/${tset_name}/trans.txt \
        ${data_dir}/speechio/${tset_name}/ref.txt
    
    cp ${exp_dir}/speechio/${model_name}/decode_asr/${tset_name}/text.proc ${exp_dir}/speechio/${model_name}/decode_asr/${tset_name}/raw_rec.txt
    sed -i "s#</s>##g" ${exp_dir}/speechio/${model_name}/decode_asr/${tset_name}/raw_rec.txt 
    echo "$0 --> Normalizing HYP text ..."
    ./utils/textnorm_zh.py \
        --has_key --to_upper \
        ${exp_dir}/speechio/${model_name}/decode_asr/${tset_name}/raw_rec.txt \
        ${exp_dir}/speechio/${model_name}/decode_asr/${tset_name}/rec.txt
    grep -v $'\t$' ${exp_dir}/speechio/${model_name}/decode_asr/${tset_name}/rec.txt > ${exp_dir}/speechio/${model_name}/decode_asr/${tset_name}/rec_non_empty.txt

    echo "$0 --> computing WER/CER and alignment ..."
    ./utils/error_rate_zh \
        --tokenizer char \
        --ref ${data_dir}/speechio/${tset_name}/ref.txt \
        --hyp ${exp_dir}/speechio/${model_name}/decode_asr/${tset_name}/rec_non_empty.txt \
        ${exp_dir}/speechio/${model_name}/decode_asr/${tset_name}/DETAILS.txt | tee ${exp_dir}/speechio/${model_name}/decode_asr/${tset_name}/RESULTS.txt
done

