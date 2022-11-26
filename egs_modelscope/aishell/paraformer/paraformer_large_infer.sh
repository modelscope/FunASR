#!/usr/bin/env bash

set -e
set -u
set -o pipefail

ori_data=
data_dir=
exp_dir=
model_name=speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
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

test_sets="dev test"

. utils/parse_options.sh

aishell_audio_dir=$ori_data/data_aishell/wav
aishell_text=$ori_data/data_aishell/transcript/aishell_transcript_v0.8.txt
dev_dir=${data_dir}/aishell/dev
test_dir=${data_dir}/aishell/test
tmp_dir=${data_dir}/aishell/tmp

mkdir -p ${dev_dir}
mkdir -p ${test_dir}
mkdir -p ${tmp_dir}

find $aishell_audio_dir -iname "*.wav" > $tmp_dir/wav.flist
grep -i "wav/dev" $tmp_dir/wav.flist > $dev_dir/wav.flist || exit 1;
grep -i "wav/test" $tmp_dir/wav.flist > $test_dir/wav.flist || exit 1;

rm -r $tmp_dir

for dir in $dev_dir $test_dir; do
    sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
    paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
    utils/filter_scp.pl -f 1 $dir/utt.list $aishell_text > $dir/transcripts.txt
    awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
    utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
    sort -u $dir/transcripts.txt > $dir/text
done

mkdir -p ${exp_dir}/aishell

modelscope_utils/modelscope_infer.sh \
        --data_dir ${data_dir}/aishell \
        --exp_dir ${exp_dir}/aishell \
        --test_sets "${test_sets}" \
        --model_name ${model_name} \
        --inference_nj ${inference_nj} \
        --gpuid_list ${gpuid_list} \
        --njob ${njob} \
        --gpu_inference ${gpu_inference} \
        --use_lm ${use_lm} \
        --beam_size ${beam_size} \
        --lm_weight ${lm_weight}
