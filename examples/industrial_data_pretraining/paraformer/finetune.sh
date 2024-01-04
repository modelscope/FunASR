
# download model
local_path_root=../modelscope_models
mkdir -p ${local_path_root}
local_path=${local_path_root}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
git clone https://www.modelscope.cn/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git ${local_path}


python funasr/bin/train.py \
+model="../modelscope_models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
+token_list="../modelscope_models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/tokens.txt" \
+train_data_set_list="data/list/audio_datasets.jsonl" \
+output_dir="outputs/debug/ckpt/funasr2/exp2" \
+device="cpu"