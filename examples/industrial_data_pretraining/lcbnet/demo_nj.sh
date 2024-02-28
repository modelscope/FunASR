file_dir="/nfs/yufan.yf/workspace/github/FunASR/examples/industrial_data_pretraining/lcbnet/exp/speech_lcbnet_contextual_asr-en-16k-bpe-vocab5002-pytorch"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
inference_device="cuda"

if [ ${inference_device} == "cuda" ]; then
    nj=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
else
    inference_batch_size=1
    CUDA_VISIBLE_DEVICES=""
    for JOB in $(seq ${nj}); do
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"-1,"
    done
fi

inference_dir="outputs/test"
_logdir="${inference_dir}/logdir"
echo "inference_dir: ${inference_dir}"

# mkdir -p "${_logdir}"
# key_file1=${file_dir}/wav.scp
# key_file2=${file_dir}/ocr.txt
# split_scps1=
# split_scps2=
# for JOB in $(seq "${nj}"); do
#     split_scps1+=" ${_logdir}/wav.${JOB}.scp"
#     split_scps2+=" ${_logdir}/ocr.${JOB}.txt"
# done
# utils/split_scp.pl "${key_file1}" ${split_scps1}
# utils/split_scp.pl "${key_file2}" ${split_scps2}

# gpuid_list_array=(${CUDA_VISIBLE_DEVICES//,/ })
# for JOB in $(seq ${nj}); do
#     {
#         id=$((JOB-1))
#         gpuid=${gpuid_list_array[$id]}

#         export CUDA_VISIBLE_DEVICES=${gpuid}

#         python -m funasr.bin.inference \
#         --config-path=${file_dir} \
#         --config-name="config.yaml" \
#         ++init_param=${file_dir}/model.pb \
#         ++tokenizer_conf.token_list=${file_dir}/tokens.txt \
#         ++input=[${_logdir}/wav.${JOB}.scp,${_logdir}/ocr.${JOB}.txt] \
#         +data_type='["kaldi_ark", "text"]' \
#         ++tokenizer_conf.bpemodel=${file_dir}/bpe.model \
#         ++output_dir="${inference_dir}/${JOB}" \
#         ++device="${inference_device}" \
#         ++ncpu=1 \
#         ++disable_log=true  &> ${_logdir}/log.${JOB}.txt

#     }&
# done
# wait


#mkdir -p ${inference_dir}/1best_recog

if [ -f "${inference_dir}/${JOB}/1best_recog/token" ]; then
    for JOB in $(seq "${nj}"); do
        cat "${inference_dir}/${JOB}/1best_recog/token" >> "${inference_dir}/1best_recog/token"
    done  
fi

echo "Computing WER ..."
echo "Computing WER ..."
#python utils/postprocess_text_zh.py ${inference_dir}/1best_recog/text ${inference_dir}/1best_recog/text.proc

#cp  ${data_dir}/text ${inference_dir}/1best_recog/text.ref
#python utils/compute_wer.py ${inference_dir}/1best_recog/text.ref ${inference_dir}/1best_recog/text.proc ${inference_dir}/1best_recog/text.cer
#tail -n 3 ${inference_dir}/1best_recog/text.cer