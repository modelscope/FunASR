
model="damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
model_revision="v2.0.4"

python funasr/bin/inference.py \
+model=${model} \
+model_revision=${model_revision} \
+input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_text/punc_example.txt" \
+output_dir="./outputs/debug" \
+device="cpu"
