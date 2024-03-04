file_dir="/nfs/yufan.yf/workspace/github/FunASR/examples/industrial_data_pretraining/lcbnet/exp/speech_lcbnet_contextual_asr-en-16k-bpe-vocab5002-pytorch"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
inference_device="cuda"
test_set="dev_wav"
if [ ${inference_device} == "cuda" ]; then
    nj=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
else
    inference_batch_size=1
    CUDA_VISIBLE_DEVICES=""
    for JOB in $(seq ${nj}); do
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"-1,"
    done
fi

inference_dir="outputs/slidespeech_dev_beamsearch_wav"
_logdir="${inference_dir}/logdir"
echo "inference_dir: ${inference_dir}"

mkdir -p "${_logdir}"
key_file1=${file_dir}/${test_set}/wav.scp
key_file2=${file_dir}/${test_set}/ocr.txt
split_scps1=
split_scps2=
for JOB in $(seq "${nj}"); do
    split_scps1+=" ${_logdir}/wav.${JOB}.scp"
    split_scps2+=" ${_logdir}/ocr.${JOB}.txt"
done
utils/split_scp.pl "${key_file1}" ${split_scps1}
utils/split_scp.pl "${key_file2}" ${split_scps2}

gpuid_list_array=(${CUDA_VISIBLE_DEVICES//,/ })
for JOB in $(seq ${nj}); do
    {
        id=$((JOB-1))
        gpuid=${gpuid_list_array[$id]}

        export CUDA_VISIBLE_DEVICES=${gpuid}

        python -m funasr.bin.inference \
        --config-path=${file_dir} \
        --config-name="config.yaml" \
        ++init_param=${file_dir}/model.pb \
        ++tokenizer_conf.token_list=${file_dir}/tokens.txt \
        ++input=[${_logdir}/wav.${JOB}.scp,${_logdir}/ocr.${JOB}.txt] \
        +data_type='["sound", "text"]' \
        ++tokenizer_conf.bpemodel=${file_dir}/bpe.model \
        ++output_dir="${inference_dir}/${JOB}" \
        ++device="${inference_device}" \
        ++ncpu=1 \
        ++disable_log=true  &> ${_logdir}/log.${JOB}.txt

    }&
done
wait


mkdir -p ${inference_dir}/1best_recog

for JOB in $(seq "${nj}"); do
   cat "${inference_dir}/${JOB}/1best_recog/token" >> "${inference_dir}/1best_recog/token"
done  

echo "Computing WER ..."
sed -e 's/ /\t/' -e 's/ //g' -e 's/▁/ /g' -e 's/\t /\t/'  ${inference_dir}/1best_recog/token > ${inference_dir}/1best_recog/token.proc
cp  ${file_dir}/${test_set}/text ${inference_dir}/1best_recog/token.ref
cp  ${file_dir}/${test_set}/ocr.list ${inference_dir}/1best_recog/ocr.list
python utils/compute_wer.py ${inference_dir}/1best_recog/token.ref ${inference_dir}/1best_recog/token.proc ${inference_dir}/1best_recog/token.cer
tail -n 3 ${inference_dir}/1best_recog/token.cer

./run_bwer_recall.sh  ${inference_dir}/1best_recog/
tail -n 6 ${inference_dir}/1best_recog/BWER-UWER.results |head -n 5