
# download model
local_path_root=../modelscope_models
mkdir -p ${local_path_root}
local_path=${local_path_root}/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
git clone https://www.modelscope.cn/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch.git ${local_path}


python funasr/bin/inference.py \
+model="${local_path}" \
+input="${local_path}/example/punc_example.txt" \
+output_dir="./outputs/debug" \
+device="cpu"
