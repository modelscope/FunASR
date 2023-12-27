
cmd="funasr/bin/inference.py"

python $cmd \
+model="/Users/zhifu/Downloads/modelscope_models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
+vad_model="/Users/zhifu/Downloads/modelscope_models/speech_fsmn_vad_zh-cn-16k-common-pytorch" \
+punc_model="/Users/zhifu/Downloads/modelscope_models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch" \
+input="/Users/zhifu/funasr_github/test_local/vad_example.wav" \
+output_dir="/Users/zhifu/Downloads/ckpt/funasr2/exp2" \
+device="cpu" \
+batch_size_s=300 \
+batch_size_threshold_s=60 \
+debug="true"

#python $cmd \
#+model="/Users/zhifu/Downloads/modelscope_models/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404" \
#+input="/Users/zhifu/Downloads/asr_example.wav" \
#+output_dir="/Users/zhifu/Downloads/ckpt/funasr2/exp2" \
#+device="cpu" \
#+"hotword='达魔院 魔搭'"

#+input="/Users/zhifu/funasr_github/test_local/wav.scp"
#+input="/Users/zhifu/funasr_github/test_local/asr_example.wav" \
#+input="/Users/zhifu/funasr_github/test_local/aishell2_dev_ios/asr_task_debug_len.jsonl" \
#+input="/Users/zhifu/funasr_github/test_local/aishell2_dev_ios/asr_task_debug_len_10.jsonl" \
#+model="/Users/zhifu/modelscope_models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \

#+model="/Users/zhifu/modelscope_models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
#+model="/Users/zhifu/modelscope_models/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404" \
#+"hotword='达魔院 魔搭'"

#+vad_model="/Users/zhifu/Downloads/modelscope_models/speech_fsmn_vad_zh-cn-16k-common-pytorch" \
