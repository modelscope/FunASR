# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh
workspace=`pwd`

CUDA_VISIBLE_DEVICES="0,1"

stage=2
stop_stage=3

inference_device="cuda" #"cpu"
inference_checkpoint="model.pt.avg10"
inference_scp="wav.scp"
inference_batch_size=32
nj=32
test_sets="test"

# model_name from model_hub, or model_dir in local path

## option 1, download model automatically, unsupported currently
model_name_or_model_dir="iic/speech_charctc_kws_phone-xiaoyun_mt"

## option 2, download model by git
local_path_root=${workspace}/modelscope_models
model_name_or_model_dir=${local_path_root}/${model_name_or_model_dir}
if [ ! -d $model_name_or_model_dir ]; then
  mkdir -p ${model_name_or_model_dir}
  git clone https://www.modelscope.cn/iic/speech_charctc_kws_phone-xiaoyun_mt.git ${model_name_or_model_dir}
fi

config=fsmn_4e_l10r2_250_128_fdim80_t2599_t4.yaml
token_list=${model_name_or_model_dir}/funasr/tokens_2599.txt
token_list2=${model_name_or_model_dir}/funasr/tokens_xiaoyun.txt
lexicon_list=${model_name_or_model_dir}/funasr/lexicon.txt
cmvn_file=${model_name_or_model_dir}/funasr/am.mvn.dim80_l2r2
init_param="${model_name_or_model_dir}/funasr/basetrain_fsmn_4e_l10r2_250_128_fdim80_t2599.pt"


# data prepare
# data dir, which contains: train.json, val.json
data_dir=../../data

train_data="${data_dir}/train.jsonl"
val_data="${data_dir}/val.jsonl"
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "stage 1: Generate audio json list"
  # generate train.jsonl and val.jsonl from wav.scp and text.txt
  python $FUNASR_DIR/funasr/datasets/audio_datasets/scp2jsonl.py \
  ++scp_file_list='['''${data_dir}/train_wav.scp''', '''${data_dir}/train_text.txt''']' \
  ++data_type_list='["source", "target"]' \
  ++jsonl_file_out="${train_data}"

  python $FUNASR_DIR/funasr/datasets/audio_datasets/scp2jsonl.py \
  ++scp_file_list='['''${data_dir}/val_wav.scp''', '''${data_dir}/val_text.txt''']' \
  ++data_type_list='["source", "target"]' \
  ++jsonl_file_out="${val_data}"
fi

# exp output dir
output_dir="${workspace}/exp/finetune_outputs"


# Training Stage
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: KWS Training"

  mkdir -p ${output_dir}
  current_time=$(date "+%Y-%m-%d_%H-%M")
  log_file="${output_dir}/train.log.txt.${current_time}"
  echo "log_file: ${log_file}"
  echo "finetune use basetrain model: ${init_param}"

  export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
  gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  torchrun --nnodes 1 --nproc_per_node ${gpu_num} \
  ../../../funasr/bin/train.py \
  --config-path "${workspace}/conf" \
  --config-name "${config}" \
  ++init_param="${init_param}" \
  ++token_lists='['''${token_list}''', '''${token_list2}''']' \
  ++seg_dicts='['''${lexicon_list}''', '''${lexicon_list}''']' \
  ++disable_update=true \
  ++train_data_set_list="${train_data}" \
  ++valid_data_set_list="${val_data}" \
  ++frontend_conf.cmvn_file="${cmvn_file}" \
  ++output_dir="${output_dir}" &> ${log_file}
fi


# Testing Stage
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: Inference"
  keywords=(小云小云)
  keywords_string=$(IFS=,; echo "${keywords[*]}")
  echo "keywords: $keywords_string"

  if [ ${inference_device} == "cuda" ]; then
      nj=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  else
      inference_batch_size=1
      CUDA_VISIBLE_DEVICES=""
      for JOB in $(seq ${nj}); do
          CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"-1,"
      done
  fi

  for dset in ${test_sets}; do
    inference_dir="${output_dir}/inference-${inference_checkpoint}/${dset}"
    _logdir="${inference_dir}/logdir"
    echo "inference_dir: ${inference_dir}"

    mkdir -p "${_logdir}"
    test_data_dir="${data_dir}/${dset}"
    key_file=${test_data_dir}/${inference_scp}

    split_scps=
    for JOB in $(seq "${nj}"); do
        split_scps+=" ${_logdir}/keys.${JOB}.scp"
    done
    $FUNASR_DIR/examples/aishell/paraformer/utils/split_scp.pl "${key_file}" ${split_scps}

    gpuid_list_array=(${CUDA_VISIBLE_DEVICES//,/ })
    for JOB in $(seq ${nj}); do
        {
          id=$((JOB-1))
          gpuid=${gpuid_list_array[$id]}

          echo "${output_dir}"

          export CUDA_VISIBLE_DEVICES=${gpuid}
          python ../../../funasr/bin/inference.py \
          --config-path="${output_dir}" \
          --config-name="config.yaml" \
          ++init_param="${output_dir}/${inference_checkpoint}" \
          ++token_lists='['''${token_list}''', '''${token_list2}''']' \
          ++seg_dicts='['''${lexicon_list}''', '''${lexicon_list}''']' \
          ++frontend_conf.cmvn_file="${cmvn_file}" \
          ++keywords="\"$keywords_string"\" \
          ++input="${_logdir}/keys.${JOB}.scp" \
          ++output_dir="${inference_dir}/${JOB}" \
          ++device="${inference_device}" \
          ++ncpu=1 \
          ++disable_log=true \
          ++batch_size="${inference_batch_size}" &> ${_logdir}/log.${JOB}.txt
        }&

    done
    wait

    for f in detect detect2; do
        if [ -f "${inference_dir}/${JOB}/${f}" ]; then
          for JOB in $(seq "${nj}"); do
              cat "${inference_dir}/${JOB}/${f}"
          done | sort -k1 >"${inference_dir}/${f}"
        fi
    done

    mkdir -p ${inference_dir}/task1
    python funasr/utils/compute_det_ctc.py \
        --keywords ${keywords_string} \
        --test_data ${test_data_dir}/wav.scp \
        --trans_data ${test_data_dir}/text \
        --score_file ${inference_dir}/detect \
        --stats_dir ${inference_dir}/task1

    mkdir -p ${inference_dir}/task2
    python funasr/utils/compute_det_ctc.py \
        --keywords ${keywords_string} \
        --test_data ${test_data_dir}/wav.scp \
        --trans_data ${test_data_dir}/text \
        --score_file ${inference_dir}/detect2 \
        --stats_dir ${inference_dir}/task2
  done

fi
