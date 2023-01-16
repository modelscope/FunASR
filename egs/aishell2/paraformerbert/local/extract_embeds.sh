#!/usr/bin/env bash

stage=1
stop_stage=3

bert_model_root="../../huggingface_models"
bert_model_name="bert-base-chinese"
#bert_model_name="chinese-roberta-wwm-ext"
#bert_model_name="mengzi-bert-base"
raw_dataset_path="../DATA"
model_path=${bert_model_root}/${bert_model_name}

. utils/parse_options.sh || exit 1;

nj=100

for data_set in train dev_ios test_ios;do
    scp=$raw_dataset_path/dump/fbank/${data_set}/text
    local_scp_dir_raw=$raw_dataset_path/embeds/$bert_model_name/${data_set}
    local_scp_dir=$local_scp_dir_raw/split$nj
    local_records_dir=$local_scp_dir_raw/ark

    mkdir -p $local_records_dir
    mkdir -p $local_scp_dir

    split_scps=""
    for JOB in $(seq ${nj}); do
        split_scps="$split_scps $local_scp_dir/data.$JOB.text"
    done

    utils/split_scp.pl $scp ${split_scps}


    for num in {0..24};do
        tmp=`expr $num \* 4`

        if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
            for idx in {1..4}; do
                JOB=`expr $tmp + $idx`
                echo "proces jobid=$JOB"
                {
                    beg=0
                    gpu=`expr $beg + $idx`
                    echo $local_scp_dir_raw/log/log.${JOB}
                    python tools/extract_embeds.py $local_scp_dir/text.$JOB.txt ${local_records_dir}/embeds.${JOB}.ark ${local_records_dir}/embeds.${JOB}.scp ${local_records_dir}/embeds.${JOB}.shape ${gpu} ${model_path} &> $local_scp_dir_raw/log/log.${JOB}
            } &
            done
            wait
        fi

        if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
            for idx in {1..4}; do
                JOB=`expr $tmp + $idx`
                echo "upload jobid=$JOB"
                {
                    hadoop  fs -put -f ${local_records_dir}/embeds.${JOB}.ark ${odps_des_feature_dir}/embeds.${JOB}.ark
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