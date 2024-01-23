
model="damo/speech_timestamp_prediction-v1-16k-offline"
model_revision="v2.0.4"

python funasr/bin/inference.py \
+model=${model} \
+model_revision=${model_revision} \
+input='["https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav", "欢迎大家来到魔搭社区进行体验"]' \
+data_type='["sound", "text"]' \
+output_dir="../outputs/debug" \
+device="cpu" \
+batch_size=2 
