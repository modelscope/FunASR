model_type=$1
nj=$2
batch_size=$4

if [ "$model_type" = "libtorch" ]; then
    backend=libtorch
    model_dir="amp_int8/libtorch"
    tag=${backend}_fp32
elif [ "$model_type" = "bladedisc" ]; then
    backend=libtorch
    model_dir="amp_int8/bladedisc"
    tag=${backend}_bladedisc
elif [ "$model_type" = "bladedisc_fp16" ]; then
    backend=libtorch
    model_dir="amp_int8/bladedisc_fp16"
    tag=${backend}_bladedisc
elif [ "$model_type" = "libtorch_fb20" ]; then
    backend=libtorch
    model_dir="amp_int8/libtorch_fb20"
    tag=${backend}_amp_fb20
elif [ "$model_type" = "onnx" ]; then
    backend=onnxruntime
    model_dir="amp_int8/onnx"
    tag=${backend}_fp32
elif [ "$model_type" = "onnx_dynamic" ]; then
    backend=onnxruntime
    model_dir="amp_int8/onnx_dynamic"
    tag=${backend}_dynamic
else
    echo 'Only support: libtorch, libtorch_fb20, onnx, onnx_dynamic'
    exit
fi
echo "=======" $model_type $nj "start======="

#scp=/nfs/haoneng.lhn/funasr_data/aishell-1/data/test/wav.scp
# scp="rtf_test_data/test/wav.scp"
scp=$3
# scp="rtf_test_data/test/wav_1500.scp"
local_scp_dir=data_debug/test/${tag}/split$nj

rtf_tool=my_test_rtf.py
# rtf_tool=my_test_rtf_nobatch.py

mkdir -p ${local_scp_dir}
# echo ${local_scp_dir}

split_scps=""
for JOB in $(seq ${nj}); do
    split_scps="$split_scps $local_scp_dir/wav.$JOB.scp"
done

# echo $split_scps

perl split_scp.pl $scp ${split_scps}

for JOB in $(seq ${nj}); do
  {
    # core_id=`expr $JOB - 1`
    # core_id=$(( $core_id % 32 ))
    # # echo $core_id
    # taskset -c ${core_id} python ${rtf_tool} ${backend} ${model_dir} ${local_scp_dir}/wav.$JOB.scp $batch_size &> ${local_scp_dir}/log.$JOB.txt
    # gpu
    wav_file=${local_scp_dir}/wav.$JOB.scp
    res_file=${local_scp_dir}/wav.$JOB.text
    log_file=${local_scp_dir}/log.$JOB.txt
    python $rtf_tool $backend $model_dir $wav_file $batch_size $res_file &> $log_file
  }&

done
wait

echo ${local_scp_dir}
rm -rf ${local_scp_dir}/total_time_comput.txt
rm -rf ${local_scp_dir}/total_time_wav.txt
rm -rf ${local_scp_dir}/total_rtf.txt
for JOB in $(seq ${nj}); do
  {
    cat ${local_scp_dir}/log.$JOB.txt | grep "total_time_comput" | awk -F ' '  '{print $2}' >> ${local_scp_dir}/total_time_comput.txt
    cat ${local_scp_dir}/log.$JOB.txt | grep "total_time_wav" | awk -F ' '  '{print $2}' >> ${local_scp_dir}/total_time_wav.txt
    cat ${local_scp_dir}/log.$JOB.txt | grep "total_rtf" | awk -F ' '  '{print $2}' >> ${local_scp_dir}/total_rtf.txt
  }

done

total_time_comput=`cat ${local_scp_dir}/total_time_comput.txt | awk 'BEGIN {max = 0} {if ($1+0>max+0) max=$1 fi} END {print max}'`
total_time_wav=`cat ${local_scp_dir}/total_time_wav.txt | awk '{sum +=$1};END {print sum}'`
rtf=`awk 'BEGIN{printf "%.5f\n",'$total_time_comput'/'$total_time_wav'}'`
speed=`awk 'BEGIN{printf "%.2f\n",1/'$rtf'}'`

echo "total_time_comput_ms: $total_time_comput"
echo "total_time_wav: $total_time_wav"
echo "total_rtf: $rtf, speech: $speed"
echo "=======" $model_type $nj "finish======="

_data=${scp%/*}
_dir=$local_scp_dir
cat ${_dir}/wav.*.text > ${_dir}/text
if [ -f ${_data}/trans.txt ];then
    echo "compute WER in shell"
    cp ${_data}/trans.txt ${_data}/text
    sed -i "s/\t/ /g" ${_data}/text
    python utils/proce_text.py ${_dir}/text ${_dir}/text.proc
    python utils/proce_text.py ${_data}/text ${_data}/text.proc
    python utils/compute_wer.py ${_data}/text.proc ${_dir}/text.proc ${_dir}/text.cer
    tail -n 3 ${_dir}/text.cer > ${_dir}/text.cer.txt
    cer=`grep "WER" ${_dir}/text.cer |cut -d ' ' -f 2`
    ser=`grep "SER" ${_dir}/text.cer |cut -d ' ' -f 2`
    echo "${model_type}:${cer}" |tee -a ${_dir}/RESULTS.txt
    echo $model_type
    cat ${_dir}/text.cer.txt
fi
