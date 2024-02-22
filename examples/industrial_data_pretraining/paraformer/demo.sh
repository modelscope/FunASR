
# method1, inference from model hub

model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model_revision="v2.0.4"

python -m funasr.bin.inference \
+model=${model} \
+model_revision=${model_revision} \
+input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav" \
+output_dir="./outputs/debug" \
+device="cpu" \


# method2, inference from local model

#python -m funasr.bin.inference \
#--config-path="/Users/zhifu/funasr_github/test_local/funasr_cli_egs" \
#--config-name="config.yaml" \
#++init_param="/Users/zhifu/funasr_github/test_local/funasr_cli_egs/model.pt" \
#++tokenizer_conf.token_list="/Users/zhifu/funasr_github/test_local/funasr_cli_egs/tokens.txt" \
#++frontend_conf.cmvn_file="/Users/zhifu/funasr_github/test_local/funasr_cli_egs/am.mvn" \
#++input="data/wav.scp" \
#++output_dir="./outputs/debug" \
#++device="cuda" \




