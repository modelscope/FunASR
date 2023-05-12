#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
gpu_num=8
count=1
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=5
train_cmd=utils/run.pl
infer_cmd=utils/run.pl

# general configuration
feats_dir="../DATA" #feature output dictionary
exp_dir="."
lang=zh
token_type=char
type=sound
scp=wav.scp
speed_perturb="0.9 1.0 1.1"
stage=1
stop_stage=1

# feature configuration
feats_dim=80
nj=64

# data
tr_dir=/nfs/wangjiaming.wjm/asr_data/aishell2/AISHELL-2/iOS/data
dev_tst_dir=/nfs/wangjiaming.wjm/asr_data/aishell2/AISHELL-DEV-TEST-SET

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
test_sets="dev_ios test_ios"

asr_config=conf/train_asr_conformer.yaml
model_dir="baseline_$(basename "${asr_config}" .yaml)_${lang}_${token_type}_${tag}"

inference_config=conf/decode_asr_transformer.yaml
inference_asr_model=valid.acc.ave_10best.pb

# you can set gpu num for decoding here
gpuid_list=$CUDA_VISIBLE_DEVICES  # set gpus for decoding, the same as training stage by default
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')

if ${gpu_inference}; then
    inference_nj=$[${ngpu}*${njob}]
    _ngpu=1
else
    inference_nj=$njob
    _ngpu=0
fi

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
    utils/compute_cmvn.sh --cmd "$train_cmd" --nj $nj --feats_dim ${feats_dim} ${feats_dir}/data/${train_set}
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
            asr_train.py \
                --gpu_id $gpu_id \
                --use_preprocessor true \
                --dataset_type $dataset_type \
                --token_type char \
                --token_list $token_list \
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

# Testing Stage
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Inference"
    for dset in ${test_sets}; do
        asr_exp=${exp_dir}/exp/${model_dir}
        inference_tag="$(basename "${inference_config}" .yaml)"
        _dir="${asr_exp}/${inference_tag}/${inference_asr_model}/${dset}"
        _logdir="${_dir}/logdir"
        if [ -d ${_dir} ]; then
            echo "${_dir} is already exists. if you want to decode again, please delete this dir first."
            exit 0
        fi
        mkdir -p "${_logdir}"
        _data="${feats_dir}/${dumpdir}/${dset}"
        key_file=${_data}/${scp}
        num_scp_file="$(<${key_file} wc -l)"
        _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
        split_scps=
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}
        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi
        ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
            python -m funasr.bin.asr_inference_launch \
                --batch_size 1 \
                --ngpu "${_ngpu}" \
                --njob ${njob} \
                --gpuid_list ${gpuid_list} \
                --data_path_and_name_and_type "${_data}/${scp},speech,${type}" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --asr_train_config "${asr_exp}"/config.yaml \
                --asr_model_file "${asr_exp}"/"${inference_asr_model}" \
                --output_dir "${_logdir}"/output.JOB \
                --mode asr \
                ${_opts}

        for f in token token_int score text; do
            if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                done | sort -k1 >"${_dir}/${f}"
            fi
        done
        python utils/proce_text.py ${_dir}/text ${_dir}/text.proc
        python utils/proce_text.py ${_data}/text ${_data}/text.proc
        python utils/compute_wer.py ${_data}/text.proc ${_dir}/text.proc ${_dir}/text.cer
        tail -n 3 ${_dir}/text.cer > ${_dir}/text.cer.txt
        cat ${_dir}/text.cer.txt
    done
fi



