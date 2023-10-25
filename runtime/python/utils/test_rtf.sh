
nj=32
stage=0

scp="/nfs/haoneng.lhn/funasr_data/aishell-1/data/test/wav.scp"
export_root="/nfs/zhifu.gzf/export"
split_scps_tool=split_scp.pl
rtf_tool=test_rtf.py

#:<<!
model_name="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
backend="onnx" # "torch"
quantize='true' # 'False'
tag=${model_name}/${backend}_quantize_${quantize}
!

logs_outputs_dir=${export_root}/logs/${tag}/split$nj
mkdir -p ${logs_outputs_dir}
echo ${logs_outputs_dir}


if [ ${stage} -le 0 ];then

    python -m funasr.export.export_model --model-name ${model_name} --export-dir ${export_root} --type ${backend} --quantize ${quantize} --audio_in ${scp}

fi


if [ ${stage} -le 1 ];then

model_dir=${export_root}/${model_name}
split_scps=""
for JOB in $(seq ${nj}); do
    split_scps="$split_scps $logs_outputs_dir/wav.$JOB.scp"
done

perl ${split_scps_tool} $scp ${split_scps}


for JOB in $(seq ${nj}); do
  {
    core_id=`expr $JOB - 1`
    taskset -c ${core_id} python ${rtf_tool} --backend ${backend} --model_dir ${model_dir} --wav_file ${logs_outputs_dir}/wav.$JOB.scp --quantize ${quantize} &> ${logs_outputs_dir}/log.$JOB.txt
  }&

done
wait


rm -rf ${logs_outputs_dir}/total_time_comput.txt
rm -rf ${logs_outputs_dir}/total_time_wav.txt
rm -rf ${logs_outputs_dir}/total_rtf.txt
for JOB in $(seq ${nj}); do
  {
    cat ${logs_outputs_dir}/log.$JOB.txt | grep "total_time_comput" | awk -F ' '  '{print $2}' >> ${logs_outputs_dir}/total_time_comput.txt
    cat ${logs_outputs_dir}/log.$JOB.txt | grep "total_time_wav" | awk -F ' '  '{print $2}' >> ${logs_outputs_dir}/total_time_wav.txt
    cat ${logs_outputs_dir}/log.$JOB.txt | grep "total_rtf" | awk -F ' '  '{print $2}' >> ${logs_outputs_dir}/total_rtf.txt
  }

done

total_time_comput=`cat ${logs_outputs_dir}/total_time_comput.txt | awk 'BEGIN {max = 0} {if ($1+0>max+0) max=$1 fi} END {print max}'`
total_time_wav=`cat ${logs_outputs_dir}/total_time_wav.txt | awk '{sum +=$1};END {print sum}'`
rtf=`awk 'BEGIN{printf "%.5f\n",'$total_time_comput'/'$total_time_wav'}'`
speed=`awk 'BEGIN{printf "%.2f\n",1/'$rtf'}'`

echo "total_time_comput_ms: $total_time_comput"
echo "total_time_wav: $total_time_wav"
echo "total_rtf: $rtf, speech: $speed"

fi