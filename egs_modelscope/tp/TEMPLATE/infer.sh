#!/usr/bin/env bash

set -e
set -u
set -o pipefail

stage=1
stop_stage=2
model="damo/speech_timestamp_prediction-v1-16k-offline"
data_dir="./data/test"
output_dir="./results"
batch_size=1
gpu_inference=true    # whether to perform gpu decoding
gpuid_list="0,1"    # set gpus, e.g., gpuid_list="0,1"
njob=4    # the number of jobs for CPU decoding, if gpu_inference=false, use CPU decoding, please set njob
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
split_texts=""
for JOB in $(seq ${nj}); do
    split_scps="$split_scps $output_dir/split/wav.$JOB.scp"
    split_texts="$split_texts $output_dir/split/text.$JOB.scp"
done
perl utils/split_scp.pl ${data_dir}/wav.scp ${split_scps}
perl utils/split_scp.pl ${data_dir}/text.scp ${split_texts}

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
            --text_in ${output_dir}/split/text.$JOB.scp \
            --output_dir ${output_dir}/output.$JOB \
            --batch_size ${batch_size} \
            --gpuid ${gpuid}
        }&
    done
    wait

    mkdir -p ${output_dir}/timestamp_prediction
    for f in tp_sync tp_time; do
        if [ -f "${output_dir}/output.1/timestamp_prediction/${f}" ]; then
          for i in $(seq "${nj}"); do
              cat "${output_dir}/output.${i}/timestamp_prediction/${f}"
          done | sort -k1 >"${output_dir}/timestamp_prediction/${f}"
        fi
    done
fi

