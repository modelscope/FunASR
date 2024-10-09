
##############################################
#  aishell_librispeech_wenetspeech_cv_fluers #
##############################################

######
ckpt_dir="/nfs/beinian.lzr/workspace/GPT-4o/Exp/Speech2Text_Align_8m-8gpu/Speech2Text_Align_V2p5_7b_1004"
ckpt_id="ds-model.pt.ep0.640000"
device="cuda:0"

stage=1
stop_stage=8
decode="true"

#data dir
jsonl_dir="/nfs/beinian.lzr/workspace/GPT-4o/Data/Speech2Text_V2/TestData/ASR"

metrics_tool=../../../funasr/metrics/wer.py

out_dir="${ckpt_dir}/inference-${ckpt_id}"

######
. utils/parse_options.sh || exit 1;

mkdir -p ${out_dir}


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

  for data_set in "aishell1_test_speech2text.jsonl" "aishell2_ios_test_speech2text.jsonl"; do
  {
      jsonl=${jsonl_dir}/${data_set}
      output_dir=${out_dir}/${data_set}
      mkdir -p ${output_dir}
      pred_file=${output_dir}/1best_recog/text_tn
      ref_file=${output_dir}/1best_recog/label
      log_file=${output_dir}/log.txt

      echo "${output_dir}"
      if [ $decode == "true" ];then

        python ./demo_speech2text_multi_lora.py ${ckpt_dir} ${ckpt_id} ${jsonl} ${output_dir} ${device} &> ${log_file}

      fi

      python ${metrics_tool} ++ref_file=${ref_file} ++hyp_file=${pred_file} ++cer_file=${pred_file}.cer ++cn_postprocess=false

      pred_file=${output_dir}/1best_recog/text
      cut ${pred_file} -d " " -f 1 > ${pred_file}.key
      cut ${pred_file} -d " " -f 2- > ${pred_file}.text

      python utils/cn_tn.py ${pred_file}.text ${pred_file}.text.tn
      paste -d " " ${pred_file}.key ${pred_file}.text.tn > ${pred_file}.tn.proc

      python utils/format5resV2.py ${ref_file} 1 > ${ref_file}.itn
      python utils/format5resV2.py ${pred_file}.tn.proc 1 > ${pred_file}.tn.proc.itn
      python ${metrics_tool} ++ref_file=${ref_file}.itn ++hyp_file=${pred_file}.tn.proc.itn ++cer_file=${pred_file}.tn.proc.itn.cer ++cn_postprocess=false

  } &
  done
  wait

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

  for data_set in "librispeech_test_clean_speech2text.jsonl" "librispeech_test_other_speech2text.jsonl"; do
  {
      jsonl=${jsonl_dir}/${data_set}
      output_dir=${out_dir}/${data_set}
      mkdir -p ${output_dir}
      pred_file=${output_dir}/1best_recog/text_tn
      ref_file=${output_dir}/1best_recog/label

      log_file=${output_dir}/log.txt

      echo "${output_dir}"
      if [ $decode == "true" ];then

        python ./demo_speech2text_multi_lora.py ${ckpt_dir} ${ckpt_id} ${jsonl} ${output_dir} ${device} &> ${log_file}

      fi

      python ${metrics_tool} ++ref_file=${ref_file} ++hyp_file=${pred_file} ++cer_file=${pred_file}.cer ++cn_postprocess=false

      pred_file=${output_dir}/1best_recog/text
      python utils/text_normalize/whisper_english_normalize.py ${pred_file} ${pred_file}.tn.proc
      python utils/text_normalize/whisper_english_normalize.py ${ref_file} ${ref_file}.tn.proc
      python ${metrics_tool} ++ref_file=${ref_file}.tn.proc ++hyp_file=${pred_file}.tn.proc ++cer_file=${pred_file}.tn.proc.cer ++cn_postprocess=false

  }
  done
  # wait

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then

  for data_set in "wenetspeech_test_meeting_speech2text.jsonl" "wenetspeech_test_net_speech2text.jsonl"; do
  {
      jsonl=${jsonl_dir}/${data_set}
      output_dir=${out_dir}/${data_set}
      mkdir -p ${output_dir}
      pred_file=${output_dir}/1best_recog/text_tn
      ref_file=${output_dir}/1best_recog/label
      log_file=${output_dir}/log.txt

      echo "${output_dir}"
      if [ $decode == "true" ];then

        python ./demo_speech2text_multi_lora.py ${ckpt_dir} ${ckpt_id} ${jsonl} ${output_dir} ${device} &> ${log_file}

      fi

      python ${metrics_tool} ++ref_file=${ref_file} ++hyp_file=${pred_file} ++cer_file=${pred_file}.cer ++cn_postprocess=false

      pred_file=${output_dir}/1best_recog/text
      cut ${pred_file} -d " " -f 1 > ${pred_file}.key
      cut ${pred_file} -d " " -f 2- > ${pred_file}.text

      python utils/cn_tn.py ${pred_file}.text ${pred_file}.text.tn
      paste -d " " ${pred_file}.key ${pred_file}.text.tn > ${pred_file}.tn.proc

      python utils/clean_res.py ${ref_file} ${ref_file}.tn.proc
      python utils/format5resV2.py ${ref_file}.tn.proc 1 > ${ref_file}.itn

      python utils/format5resV2.py ${pred_file}.tn.proc 1 > ${pred_file}.tn.proc.itn
      python ${metrics_tool} ++ref_file=${ref_file}.itn ++hyp_file=${pred_file}.tn.proc.itn ++cer_file=${pred_file}.tn.proc.itn.cer ++cn_postprocess=false

  }
  done
  # wait

fi


jsonl_dir="/nfs/beinian.lzr/workspace/GPT-4o/Data/Speech2Text/TestData"

new_prompt="语音转写，不进行文本规整："

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then

  for data_set in "common_voice_zh-CN_with_punc_itn_speech2text_singleprompt.jsonl" "fleurs_cmn_hans_cn_with_punc_itn_speech2text_singleprompt.jsonl"; do
  {
      jsonl=${jsonl_dir}/${data_set}
      output_dir=${out_dir}/${data_set}
      mkdir -p ${output_dir}
      pred_file=${output_dir}/1best_recog/text
      ref_file=${output_dir}/1best_recog/label

      log_file=${output_dir}/log.txt

      echo "${output_dir}"
      if [ $decode == "true" ];then

          python ./demo_speech2text_multi_lora.py ${ckpt_dir} ${ckpt_id} ${jsonl} ${output_dir} ${device} ${new_prompt} &> ${log_file}

          cp ${ref_file} ${ref_file}.ori

      fi

      python ${metrics_tool} ++ref_file=${ref_file} ++hyp_file=${pred_file} ++cer_file=${pred_file}.cer ++cn_postprocess=false


      pred_file=${output_dir}/1best_recog/text
      cut ${pred_file} -d " " -f 1 > ${pred_file}.key
      cut ${pred_file} -d " " -f 2- > ${pred_file}.text

      python utils/cn_tn.py ${pred_file}.text ${pred_file}.text.tn
      paste -d " " ${pred_file}.key ${pred_file}.text.tn > ${pred_file}.tn.proc


      python utils/clean_res.py ${ref_file}.ori ${ref_file}
      cut ${ref_file} -f 1 > ${ref_file}.key
      cut ${ref_file} -f 2- > ${ref_file}.text

      python utils/cn_tn.py ${ref_file}.text ${ref_file}.text.tn
      paste -d " " ${ref_file}.key ${ref_file}.text.tn > ${ref_file}.tn.proc


      python utils/format5resV2.py ${ref_file}.tn.proc 1 > ${ref_file}.tn.proc.itn
      python utils/format5resV2.py ${pred_file}.tn.proc 1 > ${pred_file}.tn.proc.itn

      python ${metrics_tool} ++ref_file=${ref_file}.tn.proc.itn ++hyp_file=${pred_file}.tn.proc.itn ++cer_file=${pred_file}.tn.proc.itn.cer ++cn_postprocess=false

  }
  done
#   wait

fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then

  for data_set in "common_voice_en_with_punc_itn_speech2text_singleprompt.jsonl" "fleurs_en_us_with_punc_itn_speech2text_singleprompt.jsonl"; do
  {
      jsonl=${jsonl_dir}/${data_set}
      output_dir=${out_dir}/${data_set}
      mkdir -p ${output_dir}
      pred_file=${output_dir}/1best_recog/text
      ref_file=${output_dir}/1best_recog/label

      log_file=${output_dir}/log.txt

      echo "${output_dir}"
      if [ $decode == "true" ];then

          python ./demo_speech2text_multi_lora.py ${ckpt_dir} ${ckpt_id} ${jsonl} ${output_dir} ${device} ${new_prompt} &> ${log_file}

      fi


      python ${metrics_tool} ++ref_file=${ref_file} ++hyp_file=${pred_file} ++cer_file=${pred_file}.cer ++cn_postprocess=false

      pred_file=${output_dir}/1best_recog/text
      python utils/text_normalize/whisper_english_normalize.py ${pred_file} ${pred_file}.tn.proc
      python utils/text_normalize/whisper_english_normalize.py ${ref_file} ${ref_file}.tn.proc
      python ${metrics_tool} ++ref_file=${ref_file}.tn.proc ++hyp_file=${pred_file}.tn.proc ++cer_file=${pred_file}.tn.proc.cer ++cn_postprocess=false

  }
  done
#   wait

fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then

  for data_set in "common_voice_ja_with_punc_itn_speech2text_singleprompt.jsonl" "common_voice_ko_with_punc_itn_speech2text_singleprompt.jsonl" "fleurs_ja_jp_with_punc_itn_speech2text_singleprompt.jsonl" "fleurs_ko_kr_with_punc_itn_speech2text_singleprompt.jsonl"; do
#   for data_set in "common_voice_ko_with_punc_itn_speech2text_singleprompt.jsonl" "fleurs_ko_kr_with_punc_itn_speech2text_singleprompt.jsonl"; do

  {
      jsonl=${jsonl_dir}/${data_set}
      output_dir=${out_dir}/${data_set}
      mkdir -p ${output_dir}
      pred_file=${output_dir}/1best_recog/text
      ref_file=${output_dir}/1best_recog/label

      log_file=${output_dir}/log.txt

      echo "${output_dir}"
      if [ $decode == "true" ];then

          python ./demo_speech2text_multi_lora.py ${ckpt_dir} ${ckpt_id} ${jsonl} ${output_dir} ${device} ${new_prompt} &> ${log_file}


      fi

      python utils/text_normalize/whisper_basic_normalize.py ${pred_file} ${pred_file}.tn
      python utils/text_normalize/add_space_for_zh.py ${pred_file}.tn ${pred_file}.tn.proc
      python utils/text_normalize/whisper_basic_normalize.py ${ref_file} ${ref_file}.tn
      python utils/text_normalize/add_space_for_zh.py ${ref_file}.tn ${ref_file}.tn.proc

      python ${metrics_tool} ++ref_file=${ref_file}.tn.proc ++hyp_file=${pred_file}.tn.proc ++cer_file=${pred_file}.tn.proc.cer ++cn_postprocess=false

   }
   done
   # wait

fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then

  for data_set in "common_voice_yue_with_punc_itn_speech2text_singleprompt.jsonl" "fleurs_yue_hant_hk_with_punc_itn_speech2text_singleprompt.jsonl"; do
  {
      jsonl=${jsonl_dir}/${data_set}
      output_dir=${out_dir}/${data_set}
      mkdir -p ${output_dir}
      pred_file=${output_dir}/1best_recog/text
      ref_file=${output_dir}/1best_recog/label

      log_file=${output_dir}/log.txt


      echo "${output_dir}"
      if [ $decode == "true" ];then

          python ./demo_speech2text_multi_lora.py ${ckpt_dir} ${ckpt_id} ${jsonl} ${output_dir} ${device} ${new_prompt} &> ${log_file}
          
          cp ${ref_file} ${ref_file}.ori

      fi

      python ${metrics_tool} ++ref_file=${ref_file} ++hyp_file=${pred_file} ++cer_file=${pred_file}.cer ++cn_postprocess=false

      cp ${ref_file} ${ref_file}.ori

      pred_file=${output_dir}/1best_recog/text
      cut ${pred_file} -d " " -f 1 > ${pred_file}.key
      cut ${pred_file} -d " " -f 2- > ${pred_file}.text

      python utils/cn_tn.py ${pred_file}.text ${pred_file}.text.tn
      paste -d " " ${pred_file}.key ${pred_file}.text.tn > ${pred_file}.tn.proc


      python utils/clean_res.py ${ref_file}.ori ${ref_file}
      cut ${ref_file} -f 1 > ${ref_file}.key
      cut ${ref_file} -f 2- > ${ref_file}.text

      python utils/cn_tn.py ${ref_file}.text ${ref_file}.text.tn
      paste -d " " ${ref_file}.key ${ref_file}.text.tn > ${ref_file}.tn.proc


      python utils/format5resV2.py ${ref_file}.tn.proc 1 > ${ref_file}.tn.proc.itn
      python utils/format5resV2.py ${pred_file}.tn.proc 1 > ${pred_file}.tn.proc.itn

      python utils/text_normalize/zh_hant2zh_cn_process.py --input_file ${pred_file}.tn.proc.itn --output_file ${pred_file}.tn.proc.itn.cn
      python utils/text_normalize/zh_hant2zh_cn_process.py --input_file ${ref_file}.tn.proc.itn --output_file ${ref_file}.tn.proc.itn.cn

      python ${metrics_tool} ++ref_file=${ref_file}.tn.proc.itn.cn ++hyp_file=${pred_file}.tn.proc.itn.cn ++cer_file=${pred_file}.tn.proc.itn.cn.cer ++cn_postprocess=false




  }
  done
#   wait

fi


if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then

  for data_set in "common_voice_de_with_punc_itn_speech2text.jsonl" "common_voice_ko_with_punc_itn_speech2text_singleprompt.jsonl" "fleurs_ja_jp_with_punc_itn_speech2text_singleprompt.jsonl" "fleurs_ko_kr_with_punc_itn_speech2text_singleprompt.jsonl"; do
  {
      jsonl=${jsonl_dir}/${data_set}
      output_dir=${out_dir}/${data_set}
      mkdir -p ${output_dir}
      pred_file=${output_dir}/1best_recog/text
      ref_file=${output_dir}/1best_recog/label

      log_file=${output_dir}/log.txt
      if [ $decode == "true" ];then
          python ./demo_speech2text_multi_lora.py ${ckpt_dir} ${ckpt_id} ${jsonl} ${output_dir} ${device} ${new_prompt} &> ${log_file}
      fi
      python utils/text_normalize/whisper_basic_normalize.py ${pred_file} ${pred_file}.tn
      python utils/text_normalize/add_space_for_zh.py ${pred_file}.tn ${pred_file}.tn.proc
      python utils/text_normalize/whisper_basic_normalize.py ${ref_file} ${ref_file}.tn
      python utils/text_normalize/add_space_for_zh.py ${ref_file}.tn ${ref_file}.tn.proc

      python ${metrics_tool} ++ref_file=${ref_file}.tn.proc ++hyp_file=${pred_file}.tn.proc ++cer_file=${pred_file}.tn.proc.cer ++cn_postprocess=false

   }
   done
   # wait

fi
