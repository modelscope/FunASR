#now_result_name=asr_conformer_acc1_lr002_warm20000/decode_asr_asr_model_valid.acc.ave
#hotword_type=ocr_1ngram_top10_hotwords_list
hot_exp_suf=$1


python compute_wer_details.py --v 1 \
   --ref ${hot_exp_suf}/token.ref \
   --ref_ocr ${hot_exp_suf}/ocr.list  \
   --rec_name base \
   --rec_file ${hot_exp_suf}/token.proc \
   > ${hot_exp_suf}/BWER-UWER.results
