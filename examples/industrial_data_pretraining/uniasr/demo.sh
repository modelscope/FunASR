
model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"


python funasr/bin/inference.py \
+model=${model} \
+input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav" \
+output_dir="./outputs/debug" \
+device="cpu" \

