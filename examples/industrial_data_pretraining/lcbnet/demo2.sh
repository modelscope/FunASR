file_dir="/nfs/yufan.yf/workspace/github/FunASR/examples/industrial_data_pretraining/lcbnet/exp/speech_lcbnet_contextual_asr-en-16k-bpe-vocab5002-pytorch"


python -m funasr.bin.inference \
--config-path=${file_dir} \
--config-name="config.yaml" \
++init_param=${file_dir}/model.pb \
++tokenizer_conf.token_list=${file_dir}/tokens.txt \
++frontend_conf.cmvn_file=${file_dir}/am.mvn \
++input=${file_dir}/wav.scp \
++input=${file_dir}/ocr_text \
+data_type='["sound", "text"]' \
++tokenizer_conf.bpemodel=${file_dir}/bpe.model \
++output_dir="./outputs/debug" \
++device="" \
