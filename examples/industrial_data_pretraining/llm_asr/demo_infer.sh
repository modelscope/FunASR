# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)



python -m funasr.bin.inference \
--config-path="/root/FunASR/examples/aishell/llm_asr_nar/conf" \
--config-name="template.yaml" \
++init_param="/mnt/workspace/FunASR/examples/aishell/paraformer/exp/baseline_paraformer_conformer_12e_6d_2048_256_zh_char_exp3/model.pt.ep38" \
++input="/nfs/beinian.lzr/workspace/datasets/data/16k/opendata/aishell1/dev/wav/S0724/BAC009S0724W0121.wav" \
++scope_map="encoder.model,audio_encoder,encoder_projector,adaptor" \
++output_dir="./outputs/debug" \
++device="cpu" \

