

model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"


python funasr/bin/inference.py \
+model=${model} \
+input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav" \
+output_dir="./outputs/debug" \
+device="cpu" \
