#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
gpu_num=8
count=1

train_cmd=tools/run.pl

# general configuration
feats_dir="../DATA" #feature output dictionary
exp_dir="."
lang=zh
dumpdir=dump/fbank
feats_type=fbank
token_type=char
dataset_type=large
stage=0
stop_stage=4

# feature configuration
feats_dim=80
sample_frequency=16000
nj=100
speed_perturb="0.9,1.0,1.1"

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

asr_config=conf/train_asr_paraformer_conformer_20e_1280_320_6d_1280_320.yaml
model_dir="baseline_$(basename "${asr_config}" .yaml)_${feats_type}_${lang}_${token_type}_${tag}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    # For training set
    local/prepare_data.sh ${tr_dir} ${feats_dir}/data/local/train ${feats_dir}/data/train || exit 1;
    # # For dev and test set
    for x in Android iOS Mic; do
        local/prepare_data.sh ${dev_tst_dir}/${x}/dev ${feats_dir}/data/local/dev_${x,,} ${feats_dir}/data/dev_${x,,} || exit 1;
        local/prepare_data.sh ${dev_tst_dir}/${x}/test ${feats_dir}/data/local/test_${x,,} ${feats_dir}/data/test_${x,,} || exit 1;
    done 
    # Normalize text to capital letters
    for x in train dev_android dev_ios dev_mic test_android test_ios test_mic; do
        mv ${feats_dir}/data/${x}/text ${feats_dir}/data/${x}/text.org
        paste -d " " <(cut -f 1 ${feats_dir}/data/${x}/text.org) <(cut -f 2- ${feats_dir}/data/${x}/text.org \
             | tr 'A-Z' 'a-z' | tr -d " ") \
            > ${feats_dir}/data/${x}/text
        tools/text2token.py -n 1 -s 1 ${feats_dir}/data/${x}/text > ${feats_dir}/data/${x}/text.org
        mv ${feats_dir}/data/${x}/text.org ${feats_dir}/data/${x}/text
    done
fi

feat_train_dir=${feats_dir}/${dumpdir}/${train_set}; mkdir -p ${feat_train_dir}
feat_dev_dir=${feats_dir}/${dumpdir}/${valid_set}; mkdir -p ${feat_dev_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"
    # compute fbank features
    fbankdir=${feats_dir}/fbank
    steps/compute_fbank.sh --cmd "$train_cmd" --nj $nj --speed_perturb ${speed_perturb} \
        ${feats_dir}/data/train ${exp_dir}/exp/make_fbank/train ${fbankdir}/train
    tools/fix_data_feat.sh ${fbankdir}/train
    for x in android ios mic; do
        steps/compute_fbank.sh --cmd "$train_cmd" --nj $nj \
            ${feats_dir}/data/dev_${x} ${exp_dir}/exp/make_fbank/dev_${x} ${fbankdir}/dev_${x}
        tools/fix_data_feat.sh ${fbankdir}/dev_${x}
        steps/compute_fbank.sh --cmd "$train_cmd" --nj $nj \
            ${feats_dir}/data/test_${x} ${exp_dir}/exp/make_fbank/test_${x} ${fbankdir}/test_${x}
        tools/fix_data_feat.sh ${fbankdir}/test_${x}
    done
    
    # compute global cmvn
    steps/compute_cmvn.sh --cmd "$train_cmd" --nj $nj \
        ${fbankdir}/train ${exp_dir}/exp/make_fbank/train

    # apply cmvn 
    steps/apply_cmvn.sh --cmd "$train_cmd" --nj $nj \
        ${fbankdir}/${train_set} ${fbankdir}/train/cmvn.json ${exp_dir}/exp/make_fbank/${train_set} ${feat_train_dir}
    steps/apply_cmvn.sh --cmd "$train_cmd" --nj $nj \
        ${fbankdir}/${valid_set} ${fbankdir}/train/cmvn.json ${exp_dir}/exp/make_fbank/${valid_set} ${feat_dev_dir}
    for x in android ios mic; do
        steps/apply_cmvn.sh --cmd "$train_cmd" --nj $nj \
            ${fbankdir}/test_${x} ${fbankdir}/train/cmvn.json ${exp_dir}/exp/make_fbank/test_${x} ${feats_dir}/${dumpdir}/test_${x}
    done
    
    cp ${fbankdir}/${train_set}/text ${fbankdir}/${train_set}/speech_shape ${fbankdir}/${train_set}/text_shape ${feat_train_dir}
    tools/fix_data_feat.sh ${feat_train_dir}
    cp ${fbankdir}/${valid_set}/text ${fbankdir}/${valid_set}/speech_shape ${fbankdir}/${valid_set}/text_shape ${feat_dev_dir}
    tools/fix_data_feat.sh ${feat_dev_dir}
    for x in android ios mic; do
        cp ${fbankdir}/test_${x}/text ${fbankdir}/test_${x}/speech_shape ${fbankdir}/test_${x}/text_shape ${feats_dir}/${dumpdir}/test_${x}
        tools/fix_data_feat.sh ${feats_dir}/${dumpdir}/test_${x}
    done
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
    tools/text2token.py -s 1 -n 1 --space "" ${feats_dir}/data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0}' >> ${token_list}
    num_token=$(cat ${token_list} | wc -l)
    echo "<unk>" >> ${token_list}
    vocab_size=$(cat ${token_list} | wc -l)
    awk -v v=,${vocab_size} '{print $0v}' ${feat_train_dir}/text_shape > ${feat_train_dir}/text_shape.char
    awk -v v=,${vocab_size} '{print $0v}' ${feat_dev_dir}/text_shape > ${feat_dev_dir}/text_shape.char
    mkdir -p ${feats_dir}/asr_stats_fbank_zh_char/${train_set}
    mkdir -p ${feats_dir}/asr_stats_fbank_zh_char/${valid_set}
    cp ${feat_train_dir}/speech_shape ${feat_train_dir}/text_shape ${feat_train_dir}/text_shape.char ${feats_dir}/asr_stats_fbank_zh_char/${train_set} 
    cp ${feat_dev_dir}/speech_shape ${feat_dev_dir}/text_shape ${feat_dev_dir}/text_shape.char ${feats_dir}/asr_stats_fbank_zh_char/${valid_set}
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
            data2vec_train.py \
                --gpu_id $gpu_id \
                --use_preprocessor true \
                --dataset_type $dataset_type \
                --train_data_file $feats_dir/$dumpdir/${train_set}/data.list \
                --valid_data_file $feats_dir/$dumpdir/${valid_set}/data.list \
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