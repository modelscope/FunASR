
# download model
local_path_root=./modelscope_models
mkdir -p ${local_path_root}

local_path=${local_path_root}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
git clone https://www.modelscope.cn/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404.git ${local_path}

local_path_vad=${local_path_root}/speech_fsmn_vad_zh-cn-16k-common-pytorch
git clone https://www.modelscope.cn/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404.git ${local_path_vad}

local_path_punc=${local_path_root}/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
git clone https://www.modelscope.cn/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404.git ${local_path_punc}


python funasr/bin/inference.py \
+model="${local_path}" \
+vad_model="${local_path_vad}"
+punc_model="${local_path_punc}"
+input="${local_path}/example/asr_example.wav" \
+output_dir="./outputs/debug" \
+device="cpu" \
+batch_size_s=300 \
+batch_size_threshold_s=60 \
+debug="true"

