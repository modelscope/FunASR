
model="iic/speech_timestamp_prediction-v1-16k-offline"


python funasr/bin/inference.py \
+model=${model} \
+input='["https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav", "欢迎大家来到魔搭社区进行体验"]' \
+data_type='["sound", "text"]' \
+output_dir="../outputs/debug" \
+device="cpu" \
+batch_size=2 
