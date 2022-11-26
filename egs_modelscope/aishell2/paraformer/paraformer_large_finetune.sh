#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
CUDA_VISIBLE_DEVICES="0,1" # set gpus, e.g., CUDA_VISIBLE_DEVICES="0,1"
gpu_num=2
count=1
gpu_inference=true # Whether to perform gpu decoding, set false for cpu decoding
njob=4 # the number of jobs for each gpu
train_cmd=utils/run.pl

# general configuration
feats_dir="." #feature output dictionary, for large data
exp_dir="."
lang=zh
dumpdir=dump/fbank
feats_type=fbank
token_type=char
scp=feats.scp
type=kaldi_ark
stage=0
stop_stage=4

# feature configuration
feats_dim=560
sample_frequency=16000
nj=100
speed_perturb="1.0"
lfr=True
lfr_m=7
lfr_n=6

init_model_name=speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch  # pre-trained model, download from modelscope during fine-tuning
cmvn_file=init_model/${init_model_name}/am.mvn
seg_file=init_model/${init_model_name}/seg_dict
vocab=init_model/${init_model_name}/tokens.txt

# data
tr_dir=
dev_tst_dir=

# exp tag
tag=""

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev_ios
test_sets="dev_ios test_android test_ios test_mic"

asr_config=conf/train_asr_paraformer_sanm_50e_16d_2048_512_lfr6.yaml
init_param="init_model/${init_model_name}/${init_model_name}"

inference_config=conf/decode_asr_transformer_noctc_1best.yaml
inference_asr_model=valid.acc.ave_10best.pth

. utils/parse_options.sh || exit 1;

# download model from modelscope
python modelscope_utils/download_model.py --model_name ${init_model_name}

if [ ! -d ${HOME}/.cache/modelscope/hub/damo/${init_model_name} ]; then
    echo "${HOME}/.cache/modelscope/hub/damo/${init_model_name} must exist"
    exit 1
else
    if [ -d init_model/${init_model_name} ]; then
        echo "init_model/${init_model_name} is already exists. if you want to decode again, please delete init_model/${init_model_name} first."
    else
        mkdir -p init_model/${init_model_name}
        cp -r ${HOME}/.cache/modelscope/hub/damo/${init_model_name}/* init_model/${init_model_name}
    fi
fi

model_dir="baseline_$(basename "${asr_config}" .yaml)_${feats_type}_${lang}_${token_type}_${tag}"

# you can set gpu num for decoding here
gpuid_list=$CUDA_VISIBLE_DEVICES  # set gpus for decoding, the same as training stage by default
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')
inference_nj=$[${ngpu}*${njob}]

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    # For training set
    local/aishell2_data_prep.sh ${tr_dir} ${feats_dir}/data/local/train ${feats_dir}/data/train || exit 1;
    # # For dev and test set
    for x in Android iOS Mic; do
        local/aishell2_data_prep.sh ${dev_tst_dir}/${x}/dev ${feats_dir}/data/local/dev_${x,,} ${feats_dir}/data/dev_${x,,} || exit 1;
        local/aishell2_data_prep.sh ${dev_tst_dir}/${x}/test ${feats_dir}/data/local/test_${x,,} ${feats_dir}/data/test_${x,,} || exit 1;
    done
    # Normalize text to capital letters
    for x in train dev_android dev_ios dev_mic test_android test_ios test_mic; do
        mv ${feats_dir}/data/${x}/text ${feats_dir}/data/${x}/text.org
        paste -d " " <(cut -f 1 ${feats_dir}/data/${x}/text.org) <(cut -f 2- ${feats_dir}/data/${x}/text.org \
             | tr 'A-Z' 'a-z' | tr -d " ") \
            > ${feats_dir}/data/${x}/text
        rm ${feats_dir}/data/${x}/text.org
    done
fi

feat_train_dir=${feats_dir}/${dumpdir}/${train_set}; mkdir -p ${feat_train_dir}
feat_dev_dir=${feats_dir}/${dumpdir}/${valid_set}; mkdir -p ${feat_dev_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Feature Generation"
    # compute fbank features
    fbankdir=${feats_dir}/fbank
    utils/compute_fbank.sh --cmd "$train_cmd" --nj $nj --speed_perturb ${speed_perturb} \
        ${feats_dir}/data/train ${exp_dir}/exp/make_fbank/train ${fbankdir}/train
    utils/fix_data_feat.sh ${fbankdir}/train
    for x in android ios mic; do
        utils/compute_fbank.sh --cmd "$train_cmd" --nj $nj \
            ${feats_dir}/data/dev_${x} ${exp_dir}/exp/make_fbank/dev_${x} ${fbankdir}/dev_${x}
        utils/fix_data_feat.sh ${fbankdir}/dev_${x}
        utils/compute_fbank.sh --cmd "$train_cmd" --nj $nj \
            ${feats_dir}/data/test_${x} ${exp_dir}/exp/make_fbank/test_${x} ${fbankdir}/test_${x}
        utils/fix_data_feat.sh ${fbankdir}/test_${x}
    done

    echo "apply low_frame_rate and cmvn"
    [ ! -f ${cmvn_file} ] && echo "$0: cmvn file is required" && exit 1;
    utils/apply_lfr_and_cmvn.sh --cmd "$train_cmd" --nj $nj \
        --lfr $lfr --lfr-m $lfr_m --lfr-n $lfr_n \
        ${fbankdir}/${train_set} ${cmvn_file} ${exp_dir}/exp/make_fbank/train ${feat_train_dir}
    utils/apply_lfr_and_cmvn.sh --cmd "$train_cmd" --nj $nj \
        --lfr $lfr --lfr-m $lfr_m --lfr-n $lfr_n \
        ${fbankdir}/${valid_set} ${cmvn_file} ${exp_dir}/exp/make_fbank/dev ${feat_dev_dir}
    for x in android ios mic; do
        feat_test_dir=${feats_dir}/${dumpdir}/test_${x}; mkdir ${feat_test_dir}
        utils/apply_lfr_and_cmvn.sh --cmd "$train_cmd" --nj $nj \
            --lfr $lfr --lfr-m $lfr_m --lfr-n $lfr_n \
            ${fbankdir}/test_${x} ${cmvn_file} ${exp_dir}/exp/make_fbank/test_${x} ${feat_test_dir}
    done

    echo "Text Tokenize"
    # 我爱reading->我 爱 read@@ ing
    utils/text_tokenize.sh --cmd "$train_cmd" --nj $nj ${fbankdir}/${train_set} ${seg_file} ${feat_train_dir}/log ${feat_train_dir}
    utils/fix_data_feat.sh ${feat_train_dir}
    utils/text_tokenize.sh --cmd "$train_cmd" --nj $nj ${fbankdir}/${valid_set} ${seg_file} ${feat_dev_dir}/log ${feat_dev_dir}
    utils/fix_data_feat.sh ${feat_dev_dir}
    for x in android ios mic; do
      feat_test_dir=${feats_dir}/${dumpdir}/test_${x} 
      cp ${fbankdir}/test_${x}/text  ${feat_test_dir}
    done
fi

token_list=${feats_dir}/data/${lang}_token_list/char/tokens.txt
echo "dictionary: ${token_list}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary Preparation"
    mkdir -p ${feats_dir}/data/${lang}_token_list/char/
    cp $vocab ${token_list}

    vocab_size=$(wc -l <${token_list})
    awk -v v=,${vocab_size} '{print $0v}' ${feat_train_dir}/text_shape > ${feat_train_dir}/text_shape.char
    awk -v v=,${vocab_size} '{print $0v}' ${feat_dev_dir}/text_shape > ${feat_dev_dir}/text_shape.char
    mkdir -p ${feats_dir}/asr_stats_fbank_zh_char/train
    mkdir -p ${feats_dir}/asr_stats_fbank_zh_char/dev_ios
    cp ${feat_train_dir}/speech_shape ${feat_train_dir}/text_shape ${feat_train_dir}/text_shape.char ${feats_dir}/asr_stats_fbank_zh_char/${train_set}
    cp ${feat_dev_dir}/speech_shape ${feat_dev_dir}/text_shape ${feat_dev_dir}/text_shape.char ${feats_dir}/asr_stats_fbank_zh_char/${valid_set}
fi

# Training Stage
world_size=$gpu_num  # run on one machine
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # update asr train config.yaml
    python modelscope_utils/update_config.py --modelscope_config init_model/${init_model_name}/asr_train_config.yaml --finetune_config ${asr_config} --output_config init_model/${init_model_name}/asr_finetune_config.yaml
    finetune_config=init_model/${init_model_name}/asr_finetune_config.yaml

    mkdir -p ${exp_dir}/exp/${model_dir}
    mkdir -p ${exp_dir}/exp/${model_dir}/log
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
                --token_type $token_type \
                --token_list $token_list \
                --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${train_set}/${scp},speech,${type} \
                --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${train_set}/text,text,text \
                --train_shape_file ${feats_dir}/asr_stats_fbank_zh_char/${train_set}/speech_shape \
                --train_shape_file ${feats_dir}/asr_stats_fbank_zh_char/${train_set}/text_shape.char \
                --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${valid_set}/${scp},speech,${type} \
                --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${valid_set}/text,text,text \
                --valid_shape_file ${feats_dir}/asr_stats_fbank_zh_char/${valid_set}/speech_shape \
                --valid_shape_file ${feats_dir}/asr_stats_fbank_zh_char/${valid_set}/text_shape.char  \
                --resume true \
                --output_dir ${exp_dir}/exp/${model_dir} \
                --init_param $init_param \
                --config $finetune_config \
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

# Testing Stage
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    ./utils/easy_asr_infer.sh \
        --lang zh \
        --datadir ${feats_dir} \
        --feats_type ${feats_type} \
        --feats_dim ${feats_dim} \
        --token_type ${token_type} \
        --gpu_inference ${gpu_inference} \
        --inference_config "${inference_config}" \
        --test_sets "${test_sets}" \
        --token_list $token_list \
        --asr_exp ${exp_dir}/exp/${model_dir} \
        --stage 12 \
        --stop_stage 12 \
        --scp $scp \
        --text text \
        --inference_nj $inference_nj \
        --njob $njob \
        --inference_asr_model $inference_asr_model \
        --gpuid_list $gpuid_list \
        --mode paraformer
fi

