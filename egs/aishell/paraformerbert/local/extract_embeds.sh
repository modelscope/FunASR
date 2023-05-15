#!/usr/bin/env bash

stage=1
stop_stage=3

bert_model_root="../../huggingface_models"
bert_model_name="bert-base-chinese"
raw_dataset_path="../DATA"
model_path=${bert_model_root}/${bert_model_name}

. utils/parse_options.sh || exit 1;

nj=32

for data_set in train dev test;do
    scp=$raw_dataset_path/dump/fbank/${data_set}/text
    local_scp_dir_raw=${raw_dataset_path}/${data_set}
    local_scp_dir=$local_scp_dir_raw/split$nj
    local_records_dir=$local_scp_dir_raw/ark

    mkdir -p $local_records_dir
    mkdir -p $local_scp_dir

    split_scps=""
    for JOB in $(seq ${nj}); do
        split_scps="$split_scps $local_scp_dir/data.$JOB.text"
    done

    utils/split_scp.pl $scp ${split_scps}


    for num in {0..7};do
        tmp=`expr $num \* 4`

        if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
            for idx in {1..4}; do
                JOB=`expr $tmp + $idx`
                echo "proces jobid=$JOB"
                {
                    beg=0
                    gpu=`expr $beg + $idx`
                    echo ${local_scp_dir}/log.${JOB}
                    python utils/extract_embeds.py $local_scp_dir/data.$JOB.text ${local_records_dir}/embeds.${JOB}.ark ${local_records_dir}/embeds.${JOB}.scp ${local_records_dir}/embeds.${JOB}.shape ${gpu} ${model_path} &> ${local_scp_dir}/log.${JOB}
            } &
            done
            wait
        fi
    done

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        for JOB in $(seq ${nj}); do
            cat ${local_records_dir}/embeds.${JOB}.scp || exit 1;
        done > ${local_scp_dir_raw}/embeds.scp

        for JOB in $(seq ${nj}); do
            cat ${local_records_dir}/embeds.${JOB}.shape || exit 1;
        done > ${local_scp_dir_raw}/embeds.shape
    fi
done

echo "embeds is in: ${local_scp_dir_raw}"
echo "success"