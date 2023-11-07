
split_scps_tool=split_scp.pl
inference_tool=test_cer.py
proce_text_tool=proce_text.py
compute_wer_tool=compute_wer.py

nj=32
stage=0
stop_stage=2

scp="/nfs/haoneng.lhn/funasr_data/aishell-1/data/test/wav.scp"
label_text="/nfs/haoneng.lhn/funasr_data/aishell-1/data/test/text"
export_root="/nfs/zhifu.gzf/export"


#:<<!
model_name="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
backend="onnx" # "torch"
quantize='true' # 'False'
fallback_op_num_torch=20
tag=${model_name}/${backend}_quantize_${quantize}_${fallback_op_num_torch}
!

output_dir=${export_root}/logs/${tag}/split$nj
mkdir -p ${output_dir}
echo ${output_dir}


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ];then

    python -m funasr.export.export_model --model-name ${model_name} --export-dir ${export_root} --type ${backend} --quantize ${quantize} --audio_in ${scp} --fallback-num ${fallback_op_num_torch}

fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ];then

  model_dir=${export_root}/${model_name}
  split_scps=""
  for JOB in $(seq ${nj}); do
      split_scps="$split_scps $output_dir/wav.$JOB.scp"
  done

  perl ${split_scps_tool} $scp ${split_scps}


  for JOB in $(seq ${nj}); do
    {
      core_id=`expr $JOB - 1`
      taskset -c ${core_id} python ${inference_tool} --backend ${backend} --model_dir ${model_dir} --wav_file ${output_dir}/wav.$JOB.scp --quantize ${quantize} --output_dir ${output_dir}/${JOB} &> ${output_dir}/log.$JOB.txt
    }&

  done
  wait

  mkdir -p ${output_dir}/1best_recog
  for f in token text; do
      if [ -f "${output_dir}/1/${f}" ]; then
        for JOB in $(seq "${nj}"); do
            cat "${output_dir}/${JOB}/${f}"
        done | sort -k1 >"${output_dir}/1best_recog/${f}"
      fi
  done

fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ];then
    echo "Computing WER ..."
    python ${proce_text_tool} ${output_dir}/1best_recog/text ${output_dir}/1best_recog/text.proc
    python ${proce_text_tool} ${label_text} ${output_dir}/1best_recog/text.ref
    python ${compute_wer_tool} ${output_dir}/1best_recog/text.ref ${output_dir}/1best_recog/text.proc ${output_dir}/1best_recog/text.cer
    tail -n 3 ${output_dir}/1best_recog/text.cer
fi

