#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES="0,1,2,3"

# general configuration
# feats_dir="../DATA" #feature output dictionary
# kespeech
feats_dir=/ssd/zhuang/code/FunASR/examples/kespeech/DATA/
exp_dir=`pwd`
# feature configuration
nj=32
inference_device="cuda" #"cpu"
inference_checkpoint="model.pt.avg10"
inference_scp="wav.scp"
inference_batch_size=1
workspace=`pwd`
master_port=12345

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


# test_sets="test"
# test_sets="ES/Beijing/test ES/Ji-Lu/test ES/Jiang-Huai/test ES/Jiao-Liao/test ES/Lan-Yin/test ES/Northeastern/test ES/Southwestern/test ES/Zhongyuan/test MD/test"
test_sets=ES/Southwestern/test

config_path=/ssd/zhuang/code/FunASR/examples/aishell/paraformer/exp/baseline_paraformer_conformer_12e_6d_2048_256_zh_char_exp1
model=/ssd/zhuang/code/FunASR/examples/kespeech/paraformer/exp/baseline_paraformer_conformer_12e_6d_2048_256_zh_char_SW-LoRA-FT/model.pt.avg10
token_list=/ssd/zhuang/code/FunASR/examples/aishell/paraformer/exp/speech_paraformer_asr_nat-aishell1-pytorch/tokens.txt
cmvn_file=/ssd/zhuang/code/FunASR/examples/aishell/paraformer/exp/speech_paraformer_asr_nat-aishell1-pytorch/am.mvn
use_lora=true
lora_details=/ssd/zhuang/code/FunASR/examples/kespeech/paraformer/exp/baseline_paraformer_conformer_12e_6d_2048_256_zh_char_SW-LoRA-FT/lora_config.json





# Testing Stage
echo "Inference"
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
    data_dir="${feats_dir}/data/${dset}"
    key_file=${data_dir}/${inference_scp}

    inference_dir=/ssd/zhuang/code/FunASR/examples/kespeech/paraformer/exp/baseline_paraformer_conformer_12e_6d_2048_256_zh_char_SW-LoRA-FT/inference/${dset}
    _logdir="${inference_dir}/logdir"

    echo "inference_dir: ${inference_dir}"
    mkdir -p "${_logdir}"


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
            --config-path="${config_path}" \
            --config-name="config.yaml" \
            ++init_param="${model}" \
            ++tokenizer_conf.token_list="${token_list}" \
            ++frontend_conf.cmvn_file="${cmvn_file}" \
            ++input="${_logdir}/keys.${JOB}.scp" \
            ++output_dir="${inference_dir}/${JOB}" \
            ++device="${inference_device}" \
            ++use_lora=${use_lora} \
            ++lora_details=${lora_details} \
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














