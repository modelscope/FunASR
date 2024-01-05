
# download model
local_path_root=../modelscope_models
mkdir -p ${local_path_root}
local_path=${local_path_root}/speech_timestamp_prediction-v1-16k-offline
# git clone https://www.modelscope.cn/damo/speech_timestamp_prediction-v1-16k-offline.git ${local_path}


python funasr/bin/inference.py \
+model="${local_path}" \
+input='["/Users/shixian/code/modelscope_models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav", "欢迎大家来到魔搭社区进行体验"]' \
+data_type='["sound", "text"]' \
+output_dir="../outputs/debug" \
+device="cpu" \
+batch_size=2 
