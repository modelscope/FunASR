#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
CUDA_VISIBLE_DEVICES="0,1"
gpu_num=2
count=1
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=5
train_cmd=utils/run.pl
infer_cmd=utils/run.pl

# general configuration
feats_dir="../DATA" #feature output dictionary
exp_dir="."
lang=en
token_type=bpe
type=sound
scp=wav.scp
stage=0
stop_stage=0

# feature configuration
feats_dim=80
nj=64

# data
raw_data=/nfs/wangjiaming.wjm/librispeech
data_url=www.openslr.org/resources/12

# bpe model
nbpe=5000
bpemode=unigram

# exp tag
tag="exp1"

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
valid_set=dev
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/train_asr_conformer.yaml
model_dir="baseline_$(basename "${asr_config}" .yaml)_${lang}_${token_type}_${tag}"

inference_config=conf/decode_asr_transformer.yaml
#inference_config=conf/decode_asr_transformer_beam60_ctc0.3.yaml
inference_asr_model=valid.acc.ave_10best.pth

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


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    for part in dev-clean test-clean dev-other test-other train-clean-100; do
        local/download_and_untar.sh ${raw_data} ${data_url} ${part}
    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    # Data preparation
    for x in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
        local/data_prep.sh ${raw_data}/LibriSpeech/${x} ${feats_dir}/data/${x//-/_}
    done
fi

feat_train_dir=${feats_dir}/${dumpdir}/$train_set; mkdir -p ${feat_train_dir}
feat_dev_clean_dir=${feats_dir}/${dumpdir}/dev_clean; mkdir -p ${feat_dev_clean_dir}
feat_dev_other_dir=${feats_dir}/${dumpdir}/dev_other; mkdir -p ${feat_dev_other_dir}
feat_test_clean_dir=${feats_dir}/${dumpdir}/test_clean; mkdir -p ${feat_test_clean_dir}
feat_test_other_dir=${feats_dir}/${dumpdir}/test_other; mkdir -p ${feat_test_other_dir}
feat_dev_dir=${feats_dir}/${dumpdir}/$valid_set; mkdir -p ${feat_dev_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"
    # compute fbank features
    fbankdir=${feats_dir}/fbank
    for x in dev_clean dev_other test_clean test_other; do
        utils/compute_fbank.sh --cmd "$train_cmd" --nj 1 --max_lengths 3000 --feats_dim ${feats_dim} --sample_frequency ${sample_frequency} \
            ${feats_dir}/data/${x} ${exp_dir}/exp/make_fbank/${x} ${fbankdir}/${x}
        utils/fix_data_feat.sh ${fbankdir}/${x}
    done

    mkdir ${feats_dir}/data/$train_set
    train_sets="train_clean_100 train_clean_360 train_other_500"
    for file in wav.scp text; do
        ( for f in $train_sets; do cat $feats_dir/data/$f/$file; done ) | sort -k1 > $feats_dir/data/$train_set/$file || exit 1;
    done
    utils/compute_fbank.sh --cmd "$train_cmd" --nj $nj --max_lengths 3000 --feats_dim ${feats_dim} --sample_frequency ${sample_frequency} --speed_perturb ${speed_perturb} \
    ${feats_dir}/data/$train_set ${exp_dir}/exp/make_fbank/$train_set ${fbankdir}/$train_set
    utils/fix_data_feat.sh ${fbankdir}/$train_set

    # compute global cmvn
    utils/compute_cmvn.sh --cmd "$train_cmd" --nj $nj --feats_dim ${feats_dim} \
        ${fbankdir}/$train_set ${exp_dir}/exp/make_fbank/$train_set

    # apply cmvn
    utils/apply_cmvn.sh --cmd "$train_cmd" --nj $nj \
        ${fbankdir}/$train_set ${fbankdir}/$train_set/cmvn.json ${exp_dir}/exp/make_fbank/$train_set ${feat_train_dir}
    utils/apply_cmvn.sh --cmd "$train_cmd" --nj 1 \
        ${fbankdir}/dev_clean ${fbankdir}/$train_set/cmvn.json ${exp_dir}/exp/make_fbank/dev_clean ${feat_dev_clean_dir}
    utils/apply_cmvn.sh --cmd "$train_cmd" --nj 1\
        ${fbankdir}/dev_other ${fbankdir}/$train_set/cmvn.json ${exp_dir}/exp/make_fbank/dev_other ${feat_dev_other_dir}
    utils/apply_cmvn.sh --cmd "$train_cmd" --nj 1 \
        ${fbankdir}/test_clean ${fbankdir}/$train_set/cmvn.json ${exp_dir}/exp/make_fbank/test_clean ${feat_test_clean_dir}
    utils/apply_cmvn.sh --cmd "$train_cmd" --nj 1 \
        ${fbankdir}/test_other ${fbankdir}/$train_set/cmvn.json ${exp_dir}/exp/make_fbank/test_other ${feat_test_other_dir}

    cp ${fbankdir}/$train_set/text ${fbankdir}/$train_set/speech_shape ${fbankdir}/$train_set/text_shape ${feat_train_dir}
    cp ${fbankdir}/dev_clean/text ${fbankdir}/dev_clean/speech_shape ${fbankdir}/dev_clean/text_shape ${feat_dev_clean_dir}
    cp ${fbankdir}/dev_other/text ${fbankdir}/dev_other/speech_shape ${fbankdir}/dev_other/text_shape ${feat_dev_other_dir}
    cp ${fbankdir}/test_clean/text ${fbankdir}/test_clean/speech_shape ${fbankdir}/test_clean/text_shape ${feat_test_clean_dir}
    cp ${fbankdir}/test_other/text ${fbankdir}/test_other/speech_shape ${fbankdir}/test_other/text_shape ${feat_test_other_dir}

    dev_sets="dev_clean dev_other"
    for file in feats.scp text speech_shape text_shape; do
        ( for f in $dev_sets; do cat $feats_dir/${dumpdir}/$f/$file; done ) | sort -k1 > $feat_dev_dir/$file || exit 1;
    done

    #generate ark list
    utils/gen_ark_list.sh --cmd "$train_cmd" --nj $nj ${feat_train_dir} ${fbankdir}/${train_set} ${feat_train_dir}
    utils/gen_ark_list.sh --cmd "$train_cmd" --nj $nj ${feat_dev_dir} ${fbankdir}/${valid_set} ${feat_dev_dir}
fi

dict=${feats_dir}/data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=${feats_dir}/data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p ${feats_dir}/data/lang_char/
    echo "<blank>" > ${dict}
    echo "<s>" >> ${dict}
    echo "</s>" >> ${dict}
    cut -f 2- -d" " ${feats_dir}/data/${train_set}/text > ${feats_dir}/data/lang_char/input.txt
    spm_train --input=${feats_dir}/data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece < ${feats_dir}/data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0}' >> ${dict}
    echo "<unk>" >> ${dict}
    wc -l ${dict}

    vocab_size=$(cat ${dict} | wc -l)
    awk -v v=,${vocab_size} '{print $0v}' ${feat_train_dir}/text_shape > ${feat_train_dir}/text_shape.char
    awk -v v=,${vocab_size} '{print $0v}' ${feat_dev_dir}/text_shape > ${feat_dev_dir}/text_shape.char
    mkdir -p ${feats_dir}/asr_stats_fbank_zh_char/$train_set
    mkdir -p ${feats_dir}/asr_stats_fbank_zh_char/$valid_set
    cp ${feat_train_dir}/speech_shape ${feat_train_dir}/text_shape ${feat_train_dir}/text_shape.char ${feats_dir}/asr_stats_fbank_zh_char/$train_set
    cp ${feat_dev_dir}/speech_shape ${feat_dev_dir}/text_shape ${feat_dev_dir}/text_shape.char ${feats_dir}/asr_stats_fbank_zh_char/$valid_set
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
                --split_with_space false \
                --bpemodel ${bpemodel}.model \
                --token_type $token_type \
                --dataset_type $dataset_type \
                --token_list $dict \
                --train_data_file $feats_dir/$dumpdir/${train_set}/ark_txt.scp \
                --valid_data_file $feats_dir/$dumpdir/${valid_set}/ark_txt.scp \
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
        python utils/compute_wer.py ${_data}/text ${_dir}/text ${_dir}/text.cer
        tail -n 3 ${_dir}/text.cer > ${_dir}/text.cer.txt
        cat ${_dir}/text.cer.txt
    done
fi