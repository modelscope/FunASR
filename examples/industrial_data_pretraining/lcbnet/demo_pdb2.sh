file_dir="/nfs/yufan.yf/workspace/github/FunASR/examples/industrial_data_pretraining/lcbnet/exp/speech_lcbnet_contextual_asr-en-16k-bpe-vocab5002-pytorch"

#CUDA_VISIBLE_DEVICES="" \
python -m funasr.bin.inference \
--config-path=${file_dir} \
--config-name="config.yaml" \
++init_param=${file_dir}/model.pb \
++tokenizer_conf.token_list=${file_dir}/tokens.txt \
++input=[${file_dir}/dev_wav/wav.scp,${file_dir}/dev_wav/ocr.txt] \
+data_type='["sound", "text"]' \
++tokenizer_conf.bpemodel=${file_dir}/bpe.model \
++output_dir="./outputs/debug" \
++device="cpu" \

#++input=[${file_dir}/dev_wav/wav.scp,${file_dir}/dev_wav/ocr.txt] \