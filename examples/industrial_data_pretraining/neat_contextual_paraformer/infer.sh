
# download model
local_path_root=./modelscope_models
mkdir -p ${local_path_root}
local_path=${local_path_root}/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404
git clone https://www.modelscope.cn/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404.git ${local_path}


python funasr/bin/inference.py \
+model="${local_path}" \
+input="${local_path}/example/asr_example.wav" \
+output_dir="./outputs/debug" \
+device="cpu" \
+"hotword='达魔院 魔搭'"

