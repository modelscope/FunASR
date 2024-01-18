

python funasr/bin/inference.py \
--config-path="/Users/zhifu/funasr_github/test_local/funasr_cli_egs" \
--config-name="config.yaml" \
++init_param="/Users/zhifu/funasr_github/test_local/funasr_cli_egs/model.pt" \
+tokenizer_conf.token_list="/Users/zhifu/funasr_github/test_local/funasr_cli_egs/tokens.txt" \
+frontend_conf.cmvn_file="/Users/zhifu/funasr_github/test_local/funasr_cli_egs/am.mvn" \
+input="data/wav.scp" \
+output_dir="./outputs/debug" \
+device="cuda" \

