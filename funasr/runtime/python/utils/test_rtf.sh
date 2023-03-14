
nj=64

#:<<!
backend=libtorch
model_dir="/nfs/zhifu.gzf/export/damo/amp_int8/libtorch"
tag=${backend}_fp32
!

:<<!
backend=libtorch
model_dir="/nfs/zhifu.gzf/export/damo/amp_int8/libtorch_fb20"
tag=${backend}_amp_fb20
!

:<<!
backend=onnxruntime
model_dir="/nfs/zhifu.gzf/export/damo/amp_int8/onnx"
tag=${backend}_fp32
!

:<<!
backend=onnxruntime
model_dir="/nfs/zhifu.gzf/export/damo/amp_int8/onnx_dynamic"
tag=${backend}_fp32
!

scp=/nfs/haoneng.lhn/funasr_data/aishell-1/data/test/wav.scp
scp="/nfs/zhifu.gzf/data_debug/test/wav_1500.scp"
local_scp_dir=/nfs/zhifu.gzf/data_debug/test/${tag}/split$nj

rtf_tool=test_rtf.py

mkdir -p ${local_scp_dir}
echo ${local_scp_dir}

split_scps=""
for JOB in $(seq ${nj}); do
    split_scps="$split_scps $local_scp_dir/wav.$JOB.scp"
done

perl egs/aishell/transformer/utils/split_scp.pl $scp ${split_scps}


for JOB in $(seq ${nj}); do
  {
    core_id=`expr $JOB - 1`
    taskset -c ${core_id} python ${rtf_tool} ${backend} ${model_dir} ${local_scp_dir}/wav.$JOB.scp &> ${local_scp_dir}/log.$JOB.txt
  }&

done
wait


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