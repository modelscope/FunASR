
model="damo/emotion2vec_base"
model_revision="v2.0.4"

python funasr/bin/inference.py \
+model=${model} \
+model_revision=${model_revision} \
+input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav" \
+output_dir="./outputs/debug" \
+device="cpu" \
