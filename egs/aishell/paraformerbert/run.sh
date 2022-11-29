#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
CUDA_VISIBLE_DEVICES="0,1"
gpu_num=2
count=1
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=8
train_cmd=utils/run.pl

# general configuration
feats_dir=".." #feature output dictionary, for large data
lang=zh
dumpdir=dump/fbank
feats_type=fbank
token_type=char
scp=feats.scp
type=kaldi_ark
stage=0
stop_stage=4

skip_extract_embed=false
bert_model_root="../../huggingface_models"
bert_model_name="bert-base-chinese"

# feature configuration
feats_dim=80
sample_frequency=16000
nj=32
speed_perturb="0.9,1.0,1.1"

# data
data_aishell=

# exp tag
tag=""

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="dev test"

asr_config=conf/train_asr_paraformerbert_conformer_12e_6d_2048_256.yaml
run_dir="exp"
model_dir="baseline_$(basename "${asr_config}" .yaml)_${feats_type}_${lang}_${token_type}_${tag}"
exp_dir=$run_dir/$model_dir

inference_config=conf/decode_asr_transformer.yaml
inference_asr_model=valid.acc.ave_10best.pth

# you can set gpu num for decoding here
gpuid_list=$CUDA_VISIBLE_DEVICES  # set gpus for decoding, the same as training stage by default
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')

if ${gpu_inference}; then
    inference_nj=$[${ngpu}*${njob}]
else
    inference_nj=$njob
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    # Data preparation
    local/aishell_data_prep.sh ${data_aishell}/data_aishell/wav ${data_aishell}/data_aishell/transcript
    for x in train dev test; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
            > data/${x}/text
        utils/text2token.py -n 1 -s 1 data/${x}/text > data/${x}/text.org
        mv data/${x}/text.org data/${x}/text
    done
fi

feat_train_dir=${feats_dir}/${dumpdir}/train; mkdir -p ${feat_train_dir}
feat_dev_dir=${feats_dir}/${dumpdir}/dev; mkdir -p ${feat_dev_dir}
feat_test_dir=${feats_dir}/${dumpdir}/test; mkdir -p ${feat_test_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"
    # compute fbank features
    fbankdir=${feats_dir}/fbank
    utils/compute_fbank.sh --cmd "$train_cmd" --nj $nj --speed_perturb ${speed_perturb} \
        data/train exp/make_fbank/train ${fbankdir}/train
    utils/fix_data_feat.sh ${fbankdir}/train
    utils/compute_fbank.sh --cmd "$train_cmd" --nj $nj \
        data/dev exp/make_fbank/dev ${fbankdir}/dev
    utils/fix_data_feat.sh ${fbankdir}/dev
    utils/compute_fbank.sh --cmd "$train_cmd" --nj $nj \
        data/test exp/make_fbank/test ${fbankdir}/test
    utils/fix_data_feat.sh ${fbankdir}/test
     
    # compute global cmvn
    utils/compute_cmvn.sh --cmd "$train_cmd" --nj $nj \
        ${fbankdir}/train exp/make_fbank/train

    # apply cmvn 
    utils/apply_cmvn.sh --cmd "$train_cmd" --nj $nj \
        ${fbankdir}/train ${fbankdir}/train/cmvn.json exp/make_fbank/train ${feat_train_dir}
    utils/apply_cmvn.sh --cmd "$train_cmd" --nj $nj \
        ${fbankdir}/dev ${fbankdir}/train/cmvn.json exp/make_fbank/dev ${feat_dev_dir}
    utils/apply_cmvn.sh --cmd "$train_cmd" --nj $nj \
        ${fbankdir}/test ${fbankdir}/train/cmvn.json exp/make_fbank/test ${feat_test_dir}
    
    cp ${fbankdir}/train/text ${fbankdir}/train/speech_shape ${fbankdir}/train/text_shape ${feat_train_dir}
    cp ${fbankdir}/dev/text ${fbankdir}/dev/speech_shape ${fbankdir}/dev/text_shape ${feat_dev_dir}
    cp ${fbankdir}/test/text ${fbankdir}/test/speech_shape ${fbankdir}/test/text_shape ${feat_test_dir}

    utils/fix_data_feat.sh ${feat_train_dir}
    utils/fix_data_feat.sh ${feat_dev_dir}
    utils/fix_data_feat.sh ${feat_test_dir}
fi

token_list=${feats_dir}/data/${lang}_token_list/char/tokens.txt
echo "dictionary: ${token_list}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary Preparation"
    mkdir -p data/${lang}_token_list/char/
   
    echo "make a dictionary"
    echo "<blank>" > ${token_list}
    echo "<s>" >> ${token_list}
    echo "</s>" >> ${token_list}
    utils/text2token.py -s 1 -n 1 --space "" data/train/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0}' >> ${token_list}
    num_token=$(cat ${token_list} | wc -l)
    echo "<unk>" >> ${token_list}
    vocab_size=$(cat ${token_list} | wc -l)
    awk -v v=,${vocab_size} '{print $0v}' ${feat_train_dir}/text_shape > ${feat_train_dir}/text_shape.char
    awk -v v=,${vocab_size} '{print $0v}' ${feat_dev_dir}/text_shape > ${feat_dev_dir}/text_shape.char
    mkdir -p asr_stats_fbank_zh_char/train 
    mkdir -p asr_stats_fbank_zh_char/dev
    cp ${feat_train_dir}/speech_shape ${feat_train_dir}/text_shape ${feat_train_dir}/text_shape.char asr_stats_fbank_zh_char/train
    cp ${feat_dev_dir}/speech_shape ${feat_dev_dir}/text_shape ${feat_dev_dir}/text_shape.char asr_stats_fbank_zh_char/dev
fi

if ! "${skip_extract_embed}"; then
    local/extract_embeds.sh \
        --bert_model_root ${bert_model_root} \
        --bert_model_name ${bert_model_name} \
        --raw_dataset_path ${feats_dir}
fi

# Training Stage
world_size=$gpu_num  # run on one machine
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    mkdir -p $exp_dir
    mkdir -p $exp_dir/log
    INIT_FILE=$exp_dir/ddp_init
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
            asr_train_paraformer.py \
                --gpu_id $gpu_id \
                --use_preprocessor true \
                --token_type char \
                --token_list $token_list \
                --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${train_set}/${scp},speech,${type} \
                --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${train_set}/text,text,text \
                --train_data_path_and_name_and_type ${feats_dir}/embeds/${bert_model_name}/${train_set}/embeds.scp,embed,${type} \
                --train_shape_file ${feats_dir}/asr_stats_fbank_zh_char/${train_set}/speech_shape \
                --train_shape_file ${feats_dir}/asr_stats_fbank_zh_char/${train_set}/text_shape.char \
                --train_shape_file ${feats_dir}/embeds/${bert_model_name}/${train_set}/embeds.shape \
                --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${valid_set}/${scp},speech,${type} \
                --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${valid_set}/text,text,text \
                --valid_data_path_and_name_and_type ${feats_dir}/embeds/${bert_model_name}/${valid_set}/embeds.scp,embed,${type} \
                --valid_shape_file ${feats_dir}/asr_stats_fbank_zh_char/${valid_set}/speech_shape \
                --valid_shape_file ${feats_dir}/asr_stats_fbank_zh_char/${valid_set}/text_shape.char  \
                --valid_shape_file ${feats_dir}/embeds/${bert_model_name}/${valid_set}/embeds.shape \
                --resume true \
                --output_dir $exp_dir \
                --config $asr_config \
                --input_size $feats_dim \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --multiprocessing_distributed true \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --allow_variable_data_keys true \
                --local_rank $local_rank 1> $exp_dir/log/train.log.$i 2>&1
        } &
        done
        wait
fi

# Testing Stage
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    utils/easy_asr_infer.sh \
        --lang zh \
        --datadir ${feats_dir} \
        --feats_type ${feats_type} \
        --feats_dim ${feats_dim} \
        --token_type ${token_type} \
        --gpu_inference ${gpu_inference} \
        --inference_config "${inference_config}" \
        --test_sets "${test_sets}" \
        --token_list $token_list \
        --asr_exp $exp_dir \
        --stage 12 \
        --stop_stage 12 \
        --scp $scp \
        --text text \
        --inference_nj $inference_nj \
        --njob $njob \
        --inference_asr_model $inference_asr_model \
        --gpuid_list $gpuid_list \
        --gpu_inference ${gpu_inference} \
        --mode paraformer
fi

