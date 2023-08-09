#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
CUDA_VISIBLE_DEVICES="0,1"
gpu_num=2
count=1
train_cmd=utils/run.pl
infer_cmd=utils/run.pl

# general configuration
lang=zh
nlsyms_txt=none            # Non-linguistic symbol list if existing.
cleaner=none               # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
lm_fold_length=150         # fold_length for LM training.
word_vocab_size=10000 # Size of word vocabulary.
token_type=char
lm_token_list=

nj=10
## path to AISHELL2 trans
lm_train_text=
lm_dev_text=
lm_test_text=

train_data_path_and_name_and_type=${lm_train_text},text,text
train_shape_file=
valid_data_path_and_name_and_type=${lm_dev_text},text,text
valid_shape_file=
lm_config=conf/train_lm_transformer.yaml
exp_dir=./data
tag=exp1
model_dir="baseline_$(basename "${lm_config}" .yaml)_${lang}_${token_type}_${tag}"
lm_exp=${exp_dir}/exp/${model_dir}
inference_lm=valid.loss.ave.pb       # Language model path for decoding.

stage=0
stop_stage=3

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}

# you can set gpu num for decoding here
gpuid_list=$CUDA_VISIBLE_DEVICES  # set gpus for decoding, e.g., gpuid_list=2,3, the same as training stage by default
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')

mkdir -p ${exp_dir}/exp/${model_dir}
token_list=${exp_dir}/exp/${model_dir}/vocab.txt
blank="<blank>" # CTC blank symbole
sos="<s>"       # sos symbole
eos="</s>"      # eos symbole
oov="<unk>"     # Out of vocabulary symbol.
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    if [ "${token_type}" = char ] || [ "${token_type}" = word ]; then
        echo "Stage 0: Generate character level token_list from ${lm_train_text}"

        # The first symbol in token_list must be "<blank>":
        # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task
        python -m funasr.bin.tokenize_text  \
            --token_type "${token_type}" \
            --input "${lm_train_text}" \
            --output "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --field 2- \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --write_vocabulary true \
            --add_symbol "${blank}:0" \
            --add_symbol "${sos}:1" \
            --add_symbol "${eos}:2" \
            --add_symbol "${oov}:-1"

    else
        echo "Error: not supported --token_type '${token_type}'"
        exit 2
    fi

    ## use_word_lm=false
    ## # Create word-list for word-LM training
    ## if ${use_word_lm} && [ "${token_type}" != word ]; then
    ##     echo "Generate word level token_list from ${lm_train_text}"
    ##     python -m funasr.bin.tokenize_text \
    ##         --token_type word \
    ##         --input "${lm_train_text}" \
    ##         --output "${token_list}" \
    ##         --field 2- \
    ##         --cleaner "${cleaner}" \
    ##         --g2p "${g2p}" \
    ##         --write_vocabulary true \
    ##         --vocabulary_size "${word_vocab_size}" \
    ##         --add_symbol "${blank}:0" \
    ##         --add_symbol "${sos}:1" \
    ##         --add_symbol "${eos}:2" \
    ##         --add_symbol "${oov}:-1" 
    ## fi

    lm_token_list="${token_list}"

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Data preparation"
    
    # 1. Split the key file
    _logdir="${exp_dir}/exp/${model_dir}/log"
    mkdir -p "${_logdir}"
    # Get the minimum number among ${nj} and the number lines of input files
    _nj=$(min "${nj}" "$(<${lm_train_text} wc -l)" "$(<${lm_dev_text} wc -l)")

    key_file="${lm_train_text}"
    split_scps=""
    for n in $(seq ${_nj}); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${lm_dev_text}"
    split_scps=""
    for n in $(seq ${_nj}); do
        split_scps+=" ${_logdir}/dev.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Submit jobs
    ## python ../../funasr/bin/lm_train.py \
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        python -m funasr.bin.lm_train \
            --collect_stats true \
            --use_preprocessor true \
            --token_type "${token_type}" \
            --token_list "${lm_token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --train_data_path_and_name_and_type "${lm_train_text},text,text" \
            --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/dev.JOB.scp" \
            --output_dir "${_logdir}/stats.JOB" \
            --config ${lm_config} || { cat "${_logdir}"/stats.*.log; exit 1; }

    # 3. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    lm_stats_dir=${exp_dir}/exp/${model_dir}
    # shellcheck disable=SC2086
    python -m funasr.bin.aggregate_stats_dirs ${_opts} --output_dir "${lm_stats_dir}"

    # Append the num-tokens at the last dimensions. This is used for batch-bins count
    <"${lm_stats_dir}/train/text_shape" \
        awk -v N="$(<${lm_token_list} wc -l)" '{ print $0 "," N }' \
        >"${lm_stats_dir}/train/text_shape.${token_type}"

    <"${lm_stats_dir}/valid/text_shape" \
        awk -v N="$(<${lm_token_list} wc -l)" '{ print $0 "," N }' \
        >"${lm_stats_dir}/valid/text_shape.${token_type}"
    
    train_shape_file=${lm_stats_dir}/train/text_shape.${token_type}
    valid_shape_file=${lm_stats_dir}/valid/text_shape.${token_type}
    
fi

# Training Stage
world_size=$gpu_num  # run on one machine
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Training"
    mkdir -p ${lm_exp}
    mkdir -p ${lm_exp}/log
    INIT_FILE=${lm_exp}/ddp_init
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
            python  ../../../funasr/bin/lm_train.py \
                --gpu_id ${gpu_id} \
                --use_preprocessor true \
                --token_type "${token_type}" \
                --token_list "${lm_token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --train_data_path_and_name_and_type "${train_data_path_and_name_and_type}" \
                --train_shape_file "${train_shape_file}" \
                --valid_data_path_and_name_and_type "${valid_data_path_and_name_and_type}" \
                --valid_shape_file "${valid_shape_file}" \
                --fold_length "${lm_fold_length}" \
                --resume true \
                --output_dir "${lm_exp}" \
                --config ${lm_config} \
                --ngpu ${gpu_num} \
                --num_worker_count ${count} \
                --multiprocessing_distributed true \
                --dist_init_method ${init_method} \
                --dist_world_size ${world_size} \
                --dist_rank ${rank} \
                --local_rank ${local_rank} 1> ${exp_dir}/exp/${model_dir}/log/train.log.$i 2>&1
        } & 
      done
      wait
fi

# Testing Stage
gpu_num=1
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Calc perplexity: ${lm_test_text}"
    
    python ../../../funasr/bin/lm_inference_launch.py \
        --output_dir "${lm_exp}/perplexity_test" \
        --ngpu "${gpu_num}" \
        --batch_size 1 \
        --train_config "${lm_exp}"/config.yaml \
        --model_file "${lm_exp}/${inference_lm}" \
        --data_path_and_name_and_type "${lm_test_text},text,text" \
        --num_workers 1 \
        --split_with_space false 
fi

