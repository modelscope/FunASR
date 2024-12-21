# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)



python -m funasr.bin.inference \
--config-path="/mnt/workspace/FunASR/examples/aishell/paraformer/exp/baseline_paraformer_conformer_12e_6d_2048_256_zh_char_exp3" \
--config-name="config.yaml" \
++init_param="/mnt/workspace/FunASR/examples/aishell/paraformer/exp/baseline_paraformer_conformer_12e_6d_2048_256_zh_char_exp3/model.pt.ep38" \
++tokenizer_conf.token_list="/mnt/nfs/zhifu.gzf/data/AISHELL-1-feats/DATA/data/zh_token_list/char/tokens.txt" \
++frontend_conf.cmvn_file="/mnt/nfs/zhifu.gzf/data/AISHELL-1-feats/DATA/data/train/am.mvn" \
++input="/mnt/nfs/zhifu.gzf/data/AISHELL-1/data_aishell/wav/train/S0002/BAC009S0002W0122.wav" \
++output_dir="./outputs/debug" \
++device="cuda:0" \

