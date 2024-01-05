
# download model
local_path_root=../modelscope_models
mkdir -p ${local_path_root}
local_path=${local_path_root}/speech_timestamp_prediction-v1-16k-offline
git clone https://www.modelscope.cn/damo/speech_timestamp_prediction-v1-16k-offline.git ${local_path}


python funasr/bin/inference.py \
+model="${local_path}" \
+input='["/Users/zhifu/funasr_github/test_local/wav.scp", "/Users/zhifu/funasr_github/test_local/text.txt"]' \
+data_type='["sound", "text"]' \
+output_dir="./outputs/debug" \
+device="cpu" \
+batch_size=2 \
+debug="true"

