
model="iic/speech_conformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch"

python funasr/bin/inference.py \
+model=${model} \
+input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav" \
+output_dir="./outputs/debug" \
+device="cpu" \

