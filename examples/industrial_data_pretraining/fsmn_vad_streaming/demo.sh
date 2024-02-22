

model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
model_revision="v2.0.4"

python funasr/bin/inference.py \
+model=${model} \
+model_revision=${model_revision} \
+input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav" \
+output_dir="./outputs/debug" \
+device="cpu" \
