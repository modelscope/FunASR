
#model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
#model_revision="v2.0.4"

model="damo/punc_ct-transformer_cn-en-common-vocab471067-large"
model_revision="v2.0.4"

python funasr/bin/inference.py \
+model=${model} \
+model_revision=${model_revision} \
+input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_text/punc_example.txt" \
+output_dir="./outputs/debug" \
+device="cpu"
