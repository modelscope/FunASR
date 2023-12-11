
cmd="funasr_cli/cli/train_cli.py"

python $cmd \
--config-path "/Users/zhifu/funasr_github/test_local/funasr_cli_egs" \
--config-name "config.yaml" \
+token_list="/Users/zhifu/.cache/modelscope/hub/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/tokens.txt" \
+train_data_set_list="/Users/zhifu/funasr_github/test_local/aishell2_dev_ios/asr_task_debug_len.jsonl" \
+output_dir="/nfs/zhifu.gzf/ckpt/funasr2/exp1"