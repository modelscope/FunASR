

ckpt_id="model.pt.ep0.90000"
device="cuda:0"

ckpt_id=$1
device=$2

ckpt_dir="/nfs/beinian.lzr/workspace/GPT-4o/Exp/exp6/5m-8gpu/exp6_speech2text_linear_ddp_0609"
jsonl_dir="/nfs/beinian.lzr/workspace/GPT-4o/Data/Speech2Text/TestData"

out_dir="${ckpt_dir}/inference-${ckpt_id}"
mkdir -p ${out_dir}
for data_set in "librispeech_test_clean_speech2text.jsonl" "librispeech_test_other_speech2text.jsonl"; do
{
    jsonl=${jsonl_dir}/${data_set}
    output_dir=${out_dir}/${data_set}
    mkdir -p ${output_dir}
    pred_file=${output_dir}/1best_recog/text_tn
    ref_file=${output_dir}/1best_recog/label

    python ./demo_speech2text.py ${ckpt_dir} ${ckpt_id} ${jsonl} ${output_dir} ${device}

    python /mnt/workspace/zhifu.gzf/codebase/FunASR/funasr/metrics/wer.py ++ref_file=${ref_file} ++hyp_file=${pred_file} ++cer_file=${pred_file}.cer ++cn_postprocess=false

}&
done
wait

for data_set in "aishell1_test_speech2text.jsonl" "aishell2_ios_test_speech2text.jsonl"; do
{
    jsonl=${jsonl_dir}/${data_set}
    output_dir=${out_dir}/${data_set}
    mkdir -p ${output_dir}
    pred_file=${output_dir}/1best_recog/text_tn
    ref_file=${output_dir}/1best_recog/label

    python ./demo_speech2text.py ${ckpt_dir} ${ckpt_id} ${jsonl} ${output_dir} ${device}

    python /mnt/workspace/zhifu.gzf/codebase/FunASR/funasr/metrics/wer.py ++ref_file=${ref_file} ++hyp_file=${pred_file} ++cer_file=${pred_file}.cer ++cn_postprocess=true

}&
done
wait

for data_set in "common_voice_zh-CN_speech2text.jsonl" "common_voice_en_speech2text.jsonl"; do
{
    jsonl=${jsonl_dir}/${data_set}
    output_dir=${out_dir}/${data_set}
    mkdir -p ${output_dir}
    pred_file=${output_dir}/1best_recog/text_tn
    ref_file=${output_dir}/1best_recog/label

    python ./demo_speech2text.py ${ckpt_dir} ${ckpt_id} ${jsonl} ${output_dir} ${device}

    cn_postprocess=false
    if [ $data_set = "common_voice_zh-CN_speech2text.jsonl" ];then
      cn_postprocess=true
    fi

    python /mnt/workspace/zhifu.gzf/codebase/FunASR/funasr/metrics/wer.py ++ref_file=${ref_file} ++hyp_file=${pred_file} ++cer_file=${pred_file}.cer ++cn_postprocess=${cn_postprocess}

}&
done
