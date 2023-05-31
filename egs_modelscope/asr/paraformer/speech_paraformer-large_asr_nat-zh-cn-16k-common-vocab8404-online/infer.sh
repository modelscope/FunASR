#!/usr/bin/env bash

set -e
set -u
set -o pipefail

stage=1
stop_stage=2
model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
data_dir="./data/test"
output_dir="./results"
batch_size=64
gpu_inference=true    # whether to perform gpu decoding
gpuid_list="0,1"    # set gpus, e.g., gpuid_list="0,1"
njob=64    # the number of jobs for CPU decoding, if gpu_inference=false, use CPU decoding, please set njob
checkpoint_dir=
checkpoint_name="valid.cer_ctc.ave.pb"

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

if [ -n "${checkpoint_dir}" ]; then
  python utils/prepare_checkpoint.py ${model} ${checkpoint_dir} ${checkpoint_name}
  model=${checkpoint_dir}/${model}
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ];then
    echo "Decoding ..."
    gpuid_list_array=(${gpuid_list//,/ })
    for JOB in $(seq ${nj}); do
        {
        id=$((JOB-1))
        gpuid=${gpuid_list_array[$id]}
        mkdir -p ${output_dir}/output.$JOB
        python infer.py \
            --model ${model} \
            --audio_in ${output_dir}/split/wav.$JOB.scp \
            --output_dir ${output_dir}/output.$JOB \
            --batch_size ${batch_size} \
            --gpuid ${gpuid}
            --mode "paraformer_fake_streaming"
        }&
    done
    wait

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
    cp ${output_dir}/1best_recog/text ${output_dir}/1best_recog/text.proc
    cp ${data_dir}/text ${output_dir}/1best_recog/text.ref
    python utils/compute_wer.py ${output_dir}/1best_recog/text.ref ${output_dir}/1best_recog/text.proc ${output_dir}/1best_recog/text.cer
    tail -n 3 ${output_dir}/1best_recog/text.cer
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ];then
    echo "SpeechIO TIOBE textnorm"
    echo "$0 --> Normalizing REF text ..."
    ./utils/textnorm_zh.py \
        --has_key --to_upper \
        ${data_dir}/text \
        ${output_dir}/1best_recog/ref.txt

    echo "$0 --> Normalizing HYP text ..."
    ./utils/textnorm_zh.py \
        --has_key --to_upper \
        ${output_dir}/1best_recog/text.proc \
        ${output_dir}/1best_recog/rec.txt
    grep -v $'\t$' ${output_dir}/1best_recog/rec.txt > ${output_dir}/1best_recog/rec_non_empty.txt

    echo "$0 --> computing WER/CER and alignment ..."
    ./utils/error_rate_zh \
        --tokenizer char \
        --ref ${output_dir}/1best_recog/ref.txt \
        --hyp ${output_dir}/1best_recog/rec_non_empty.txt \
        ${output_dir}/1best_recog/DETAILS.txt | tee ${output_dir}/1best_recog/RESULTS.txt
    rm -rf ${output_dir}/1best_recog/rec.txt ${output_dir}/1best_recog/rec_non_empty.txt
fi

