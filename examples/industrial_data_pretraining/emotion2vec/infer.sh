
# download model
local_path_root=../modelscope_models
mkdir -p ${local_path_root}
local_path=${local_path_root}/emotion2vec_base
git clone https://www.modelscope.cn/damo/emotion2vec_base.git ${local_path}
#local_path=/Users/zhifu/Downloads/modelscope_models/emotion2vec_base

python funasr/bin/inference.py \
+model="${local_path}" \
+input="${local_path}/example/test.wav" \
+output_dir="./outputs/debug" \
+device="cpu" \
+debug=true
