#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES="0,1,2,3"

# general configuration
feats_dir="/ssd/zhuang/code/FunASR2024/examples/kespeech/DATA" #feature output dictionary
exp_dir=`pwd`
lang=zh
token_type=char
stage=2
stop_stage=2

# feature configuration
nj=32

inference_device="cuda" #"cpu"
inference_checkpoint="model.pt.avg10"
inference_scp="wav.scp"
inference_batch_size=1

# data
#raw_data=/data/nas/zhuang/dataset/data_aishell
raw_data=/data/nas/ASR_Datasets/data_aishell/
#data_url=www.openslr.org/resources/33s

# exp tag
tag="wenetctc_version"
workspace=`pwd`

master_port=12345

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=WD/train
valid_set=dev
test_sets="dev test"

config=transformer_12e_6d_2048_256.yaml
model_dir="baseline_$(basename "${config}" .yaml)_${lang}_${token_type}_${tag}"



if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    # Data preparation
    # please run DATA.ipynb to generate the data
    # local/aishell2_data_prep.py --raw_data ${raw_data} --outpath ${feats_dir}
    # local/aishell2_data_prep.sh ${raw_data}/data_aishell2/wav ${raw_data}/data_aishell2/transcript ${raw_data}/devNtest ${feats_dir}

    for x in ES/Beijing/train ES/Beijing/dev ES/Beijing/test ES/Ji-Lu/train ES/Ji-Lu/dev ES/Ji-Lu/test ES/Jiang-Huai/train ES/Jiang-Huai/dev ES/Jiang-Huai/test ES/Jiao-Liao/train ES/Jiao-Liao/dev ES/Jiao-Liao/test; do
        echo "processing ${feats_dir}/data/${x}"
        utils/text2token.py -n 1 -s 1 ${feats_dir}/data/${x}/text > ${feats_dir}/data/${x}/text.org
        mv ${feats_dir}/data/${x}/text.org ${feats_dir}/data/${x}/text

        # convert wav.scp text to jsonl
        scp_file_list_arg="++scp_file_list='[\"${feats_dir}/data/${x}/wav.scp\",\"${feats_dir}/data/${x}/text\"]'"
        python ../../../funasr/datasets/audio_datasets/scp2jsonl.py \
        ++data_type_list='["source", "target"]' \
        ++jsonl_file_out=${feats_dir}/data/${x}/audio_datasets.jsonl \
        ${scp_file_list_arg}
    done

    for x in ES/Lan-Yin/train ES/Lan-Yin/dev ES/Lan-Yin/test ES/Mandarin/train ES/Mandarin/dev ES/Mandarin/test ES/Northeastern/train ES/Northeastern/dev ES/Northeastern/test ES/Southwestern/train ES/Southwestern/dev ES/Southwestern/test ES/Zhongyuan/train ES/Zhongyuan/dev ES/Zhongyuan/test; do
        echo "processing ${feats_dir}/data/${x}"
        utils/text2token.py -n 1 -s 1 ${feats_dir}/data/${x}/text > ${feats_dir}/data/${x}/text.org
        mv ${feats_dir}/data/${x}/text.org ${feats_dir}/data/${x}/text

        # convert wav.scp text to jsonl
        scp_file_list_arg="++scp_file_list='[\"${feats_dir}/data/${x}/wav.scp\",\"${feats_dir}/data/${x}/text\"]'"
        python ../../../funasr/datasets/audio_datasets/scp2jsonl.py \
        ++data_type_list='["source", "target"]' \
        ++jsonl_file_out=${feats_dir}/data/${x}/audio_datasets.jsonl \
        ${scp_file_list_arg}
    done
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature and CMVN Generation"
    echo -"-config-path ${workspace}/conf   --config-name ${config}  ++train_data_set_list=${feats_dir}/data/${train_set}/audio_datasets.jsonl   ++cmvn_file=${feats_dir}/data/${train_set}/cmvn.json "
    python ../../../funasr/bin/compute_audio_cmvn.py \
    --config-path "${workspace}/conf" \
    --config-name "${config}" \
    ++train_data_set_list="${feats_dir}/data/${train_set}/audio_datasets.jsonl" \
    ++cmvn_file="${feats_dir}/data/${train_set}/cmvn.json" \


fi

token_list=${feats_dir}/data/${lang}_token_list/$token_type/tokens.txt
echo "dictionary: ${token_list}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary Preparation"
    mkdir -p ${feats_dir}/data/${lang}_token_list/$token_type/

    echo "make a dictionary"
    echo "<blank>" > ${token_list}
    echo "<s>" >> ${token_list}
    echo "</s>" >> ${token_list}
    utils/text2token.py -s 1 -n 1 --space "" ${feats_dir}/data/$train_set/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0}' >> ${token_list}
    echo "<unk>" >> ${token_list}
fi

# LM Training Stage
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Training"
fi

# ASR Training Stage
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "stage 4: ASR Training"

  mkdir -p ${exp_dir}/exp/${model_dir}
  current_time=$(date "+%Y-%m-%d_%H-%M")
  log_file="${exp_dir}/exp/${model_dir}/train.log.txt.${current_time}"
  echo "log_file: ${log_file}"

  export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
  gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  torchrun \
  --nnodes 1 \
  --nproc_per_node ${gpu_num} \
  --master_port ${master_port} \
  ../../../funasr/bin/train.py \
  --config-path "${workspace}/conf" \
  --config-name "${config}" \
  ++train_data_set_list="${feats_dir}/data/${train_set}/audio_datasets.jsonl" \
  ++valid_data_set_list="${feats_dir}/data/${valid_set}/audio_datasets.jsonl" \
  ++tokenizer_conf.token_list="${token_list}" \
  ++frontend_conf.cmvn_file="${feats_dir}/data/${train_set}/am.mvn" \
  ++output_dir="${exp_dir}/exp/${model_dir}" &> ${log_file}
fi



# Testing Stage
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "stage 5: Inference"

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

    inference_dir="${exp_dir}/exp/${model_dir}/inference-${inference_checkpoint}/${dset}_funasr_attn_rescore"
    _logdir="${inference_dir}/logdir"
    echo "inference_dir: ${inference_dir}"

    mkdir -p "${_logdir}"
    data_dir="${feats_dir}/data/${dset}"
    key_file=${data_dir}/${inference_scp}

    split_scps=
    for JOB in $(seq "${nj}"); do
        split_scps+=" ${_logdir}/keys.${JOB}.scp"
    done
    utils/split_scp.pl "${key_file}" ${split_scps}

    gpuid_list_array=(${CUDA_VISIBLE_DEVICES//,/ })
    for JOB in $(seq ${nj}); do
        {
          id=$((JOB-1))
          gpuid=${gpuid_list_array[$id]}

          export CUDA_VISIBLE_DEVICES=${gpuid}
          python ../../../funasr/bin/inference.py \
          --config-path="${exp_dir}/exp/${model_dir}" \
          --config-name="config.yaml" \
          ++init_param="${exp_dir}/exp/${model_dir}/${inference_checkpoint}" \
          ++tokenizer_conf.token_list="${token_list}" \
          ++frontend_conf.cmvn_file="${feats_dir}/data/${train_set}/am.mvn" \
          ++input="${_logdir}/keys.${JOB}.scp" \
          ++output_dir="${inference_dir}/${JOB}" \
          ++device="${inference_device}" \
          ++ncpu=1 \
          ++disable_log=true \
          ++batch_size="${inference_batch_size}" &> ${_logdir}/log.${JOB}.txt
        }&

    done
    wait

    mkdir -p ${inference_dir}/1best_recog
    for f in token score text; do
        if [ -f "${inference_dir}/${JOB}/1best_recog/${f}" ]; then
          for JOB in $(seq "${nj}"); do
              cat "${inference_dir}/${JOB}/1best_recog/${f}"
          done | sort -k1 >"${inference_dir}/1best_recog/${f}"
        fi
    done

    echo "Computing WER ..."
    python utils/postprocess_text_zh.py ${inference_dir}/1best_recog/text ${inference_dir}/1best_recog/text.proc
    python utils/postprocess_text_zh.py  ${data_dir}/text ${inference_dir}/1best_recog/text.ref
    python utils/compute_wer.py ${inference_dir}/1best_recog/text.ref ${inference_dir}/1best_recog/text.proc ${inference_dir}/1best_recog/text.cer
    tail -n 3 ${inference_dir}/1best_recog/text.cer
  done

fi