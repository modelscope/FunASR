
# download model
local_path_root=./modelscope_models
mkdir -p ${local_path_root}
local_path=${local_path_root}/speech_fsmn_vad_zh-cn-16k-common-pytorch
git clone https://www.modelscope.cn/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch.git ${local_path}


python funasr/bin/inference.py \
+model="${local_path}" \
+input="${local_path}/example/vad_example.wav" \
+output_dir="./outputs/debug" \
+device="cpu" \
