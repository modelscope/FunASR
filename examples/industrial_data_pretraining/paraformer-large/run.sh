
cmd="funasr/cli/train_cli.py"

python $cmd \
+model_pretrain="/Users/zhifu/modelscope_models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
+token_list="/Users/zhifu/.cache/modelscope/hub/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/tokens.txt" \
+train_data_set_list="/Users/zhifu/funasr_github/test_local/aishell2_dev_ios/asr_task_debug_len.jsonl" \
+output_dir="/Users/zhifu/Downloads/ckpt/funasr2/exp2" \
+device="cpu"

#--config-path "/Users/zhifu/funasr_github/examples/industrial_data_pretraining/paraformer-large/conf" \
#--config-name "finetune.yaml" \