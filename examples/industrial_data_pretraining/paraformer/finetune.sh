
## download model
#local_path_root=../modelscope_models
#mkdir -p ${local_path_root}
#local_path=${local_path_root}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
#git clone https://www.modelscope.cn/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git ${local_path}


python funasr/bin/train.py \
+model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
+model_revision="v2.0.2" \
+train_data_set_list="/Users/zhifu/funasr_github/test_local/aishell2_dev_ios/asr_task_debug_len_10.jsonl" \
+valid_data_set_list="/Users/zhifu/funasr_github/test_local/aishell2_dev_ios/asr_task_debug_len_10.jsonl" \
++dataset_conf.batch_size=64 \
++dataset_conf.batch_type="example" \
++train_conf.max_epoch=2 \
++dataset_conf.num_workers=4 \
+output_dir="outputs/debug/ckpt/funasr2/exp2" \
+debug="true"