
model="iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"


python ../../../funasr/bin/inference.py \
+model=${model} \
+input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav" \
+output_dir="./outputs/debug" \
+device="cpu" \
+"hotword='达摩院 魔搭'"

