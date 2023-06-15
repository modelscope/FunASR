#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
gpu_num=8
count=1

train_cmd=utils/run.pl

# general configuration
feats_dir="../DATA" #feature output dictionary
exp_dir="."
lang=zh
token_type=char
speed_perturb="0.9 1.0 1.1"
dataset_type=large
stage=0
stop_stage=3

# feature configuration
feats_dim=80
nj=64

# data
tr_dir=
dev_tst_dir=

# exp tag
tag="exp1"

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev_ios

asr_config=conf/train_pretrain_transformer.yaml
model_dir="baseline_$(basename "${asr_config}" .yaml) _${lang}_${token_type}_${tag}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    # For training set
    local/prepare_data.sh ${tr_dir} ${feats_dir}/data/local/train ${feats_dir}/data/train || exit 1;
    # # For dev and test set
    for x in iOS; do
        local/prepare_data.sh ${dev_tst_dir}/${x}/dev ${feats_dir}/data/local/dev_${x,,} ${feats_dir}/data/dev_${x,,} || exit 1;
        local/prepare_data.sh ${dev_tst_dir}/${x}/test ${feats_dir}/data/local/test_${x,,} ${feats_dir}/data/test_${x,,} || exit 1;
    done
    # Normalize text to capital letters
    for x in train dev_ios test_ios; do
        mv ${feats_dir}/data/${x}/text ${feats_dir}/data/${x}/text.org
        paste -d " " <(cut -f 1 ${feats_dir}/data/${x}/text.org) <(cut -f 2- ${feats_dir}/data/${x}/text.org \
             | tr 'A-Z' 'a-z' | tr -d " ") \
            > ${feats_dir}/data/${x}/text
        utils/text2token.py -n 1 -s 1 ${feats_dir}/data/${x}/text > ${feats_dir}/data/${x}/text.org
        mv ${feats_dir}/data/${x}/text.org ${feats_dir}/data/${x}/text
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature and CMVN Generation"
    utils/compute_cmvn.sh --fbankdir ${feats_dir}/data/${train_set} --cmd "$train_cmd" --nj $nj --feats_dim ${feats_dim} --config_file "$asr_config" --scale 1.0
fi

token_list=${feats_dir}/data/${lang}_token_list/char/tokens.txt
echo "dictionary: ${token_list}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary Preparation"
    mkdir -p ${feats_dir}/data/${lang}_token_list/char/

    echo "make a dictionary"
    echo "<blank>" > ${token_list}
    echo "<s>" >> ${token_list}
    echo "</s>" >> ${token_list}
    utils/text2token.py -s 1 -n 1 --space "" ${feats_dir}/data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0}' >> ${token_list}
    echo "<unk>" >> ${token_list}
 fi

# Training Stage
world_size=$gpu_num  # run on one machine
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Training"
    mkdir -p ${exp_dir}/exp/${model_dir}
    mkdir -p ${exp_dir}/exp/${model_dir}/log
    INIT_FILE=${exp_dir}/exp/${model_dir}/ddp_init
    if [ -f $INIT_FILE ];then
        rm -f $INIT_FILE
    fi
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    for ((i = 0; i < $gpu_num; ++i)); do
        {
            rank=$i
            local_rank=$i
            gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
            train.py \
                --task_name pretrain \
                --gpu_id $gpu_id \
                --use_preprocessor true \
                --data_dir ${feats_dir}/data \
                --train_set ${train_set} \
                --valid_set ${valid_set} \
                --data_file_names "wav.scp" \
                --cmvn_file ${feats_dir}/data/${train_set}/cmvn/am.mvn \
                --speed_perturb ${speed_perturb} \
                --dataset_type $dataset_type \
                --resume true \
                --output_dir ${exp_dir}/exp/${model_dir} \
                --config $asr_config \
                --input_size $feats_dim \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --multiprocessing_distributed true \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${exp_dir}/exp/${model_dir}/log/train.log.$i 2>&1
        } &
      done
      wait
fi