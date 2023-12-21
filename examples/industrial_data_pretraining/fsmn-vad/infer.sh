
cmd="funasr/bin/inference.py"

python $cmd \
+model="/Users/zhifu/Downloads/modelscope_models/speech_fsmn_vad_zh-cn-16k-common-pytorch" \
+input="/Users/zhifu/Downloads/asr_example.wav" \
+output_dir="/Users/zhifu/Downloads/ckpt/funasr2/exp2_vad" \
+device="cpu" \
