#!/usr/bin/env bash

set -e
set -u
set -o pipefail

stage=1
stop_stage=3
model="NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950"
data_dir="./data/test"
output_dir="./results_pl_gpu"
batch_size=1
gpu_inference=true    # whether to perform gpu decoding
gpuid_list="3,4"    # set gpus, e.g., gpuid_list="0,1"
njob=4    # the number of jobs for CPU decoding, if gpu_inference=false, use CPU decoding, please set njob

. utils/parse_options.sh || exit 1;

if ${gpu_inference} == "true"; then
    nj=$(echo $gpuid_list | awk -F "," '{print NF}')
else
    nj=$njob
    batch_size=1
    gpuid_list=""
    for JOB in $(seq ${nj}); do
        gpuid_list=$gpuid_list"-1,"
    done
fi

mkdir -p $output_dir/split
split_scps=""
for JOB in $(seq ${nj}); do
    split_scps="$split_scps $output_dir/split/wav.$JOB.scp"
done
perl utils/split_scp.pl ${data_dir}/wav.scp ${split_scps}

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ];then
    echo "Decoding ..."
    gpuid_list_array=(${gpuid_list//,/ })
    ./utils/run.pl JOB=1:${nj} ${output_dir}/log/infer.JOB.log \
    python infer.py \
       --model ${model} \
       --audio_in ${output_dir}/split/wav.JOB.scp \
       --output_dir ${output_dir}/output.JOB \
       --batch_size ${batch_size} \
       --gpuid ${gpuid_list_array[JOB-1]}

    mkdir -p ${output_dir}/1best_recog
    for f in token score text; do
        if [ -f "${output_dir}/output.1/1best_recog/${f}" ]; then
          for i in $(seq "${nj}"); do
              cat "${output_dir}/output.${i}/1best_recog/${f}"
          done | sort -k1 >"${output_dir}/1best_recog/${f}"
        fi
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ];then
    echo "Computing WER ..."
    cp ${output_dir}/1best_recog/token ${output_dir}/1best_recog/text.proc
    cp ${data_dir}/text ${output_dir}/1best_recog/text.ref
    sed -e 's/src//g' ${output_dir}/1best_recog/text.proc | sed -e 's/ \+/ /g' > ${output_dir}/1best_recog/text_nosp.proc
    sed -e 's/src//g' ${output_dir}/1best_recog/text.ref | sed -e 's/ \+/ /g' > ${output_dir}/1best_recog/text_nosp.ref

    python utils/compute_wer.py ${output_dir}/1best_recog/text.ref ${output_dir}/1best_recog/text.proc ${output_dir}/1best_recog/text.sp.cer
    tail -n 3 ${output_dir}/1best_recog/text.sp.cer
    python utils/compute_wer.py ${output_dir}/1best_recog/text_nosp.ref ${output_dir}/1best_recog/text_nosp.proc ${output_dir}/1best_recog/text.nosp.cer
    tail -n 3 ${output_dir}/1best_recog/text.nosp.cer
fi

