

python funasr/bin/inference.py \
--config-path="/nfs/zhifu.gzf/ckpt/llm_asr_nar_exp1" \
--config-name="config.yaml" \
++init_param="/nfs/zhifu.gzf/ckpt/llm_asr_nar_exp1/model.pt.ep5" \
++input="/Users/zhifu/funasr1.0/test_local/data_tmp/tmp_wav_10.jsonl" \
++output_dir="/nfs/zhifu.gzf/ckpt/llm_asr_nar_exp1/inference/aishell2-dev_ios-funasr" \
++device="cpu"