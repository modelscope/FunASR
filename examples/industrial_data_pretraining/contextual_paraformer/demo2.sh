python -m funasr.bin.inference \
--config-path="/nfs/yufan.yf/workspace/model_download/modelscope/hub/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404" \
--config-name="config.yaml" \
++init_param="/nfs/yufan.yf/workspace/model_download/modelscope/hub/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/model.pb" \
++tokenizer_conf.token_list="/nfs/yufan.yf/workspace/model_download/modelscope/hub/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/tokens.txt" \
++frontend_conf.cmvn_file="/nfs/yufan.yf/workspace/model_download/modelscope/hub/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/am.mvn" \
++input="/nfs/yufan.yf/workspace/model_download/modelscope/hub/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/asr_example_zh.wav" \
++output_dir="./outputs/debug2" \
++device="" \
