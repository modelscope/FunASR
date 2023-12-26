
cmd="funasr/bin/inference.py"

python $cmd \
+model="/Users/zhifu/Downloads/modelscope_models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch" \
+input="/Users/zhifu/FunASR/egs_modelscope/punctuation/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/data/punc_example.txt" \
+output_dir="/Users/zhifu/Downloads/ckpt/funasr2/exp2_punc" \
+device="cpu" \
+debug="true"
