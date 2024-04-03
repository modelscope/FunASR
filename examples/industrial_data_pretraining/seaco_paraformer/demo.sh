
model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
vad_model_revision="master"
punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
punc_model_revision="master"

python funasr/bin/inference.py \
+model=${model} \
+vad_model=${vad_model} \
+vad_model_revision=${vad_model_revision} \
+punc_model=${punc_model} \
+punc_model_revision=${punc_model_revision} \
+input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav" \
+output_dir="./outputs/debug" \
+device="cpu" \
+"hotword='达摩院 魔搭'"
