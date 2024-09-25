# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

config_path="/home/pengteng.spt/source/FunASR_KWS/examples/industrial_data_pretraining/sanm_kws/conf"
config_path="/home/pengteng.spt/source/FunASR_KWS/examples/industrial_data_pretraining/sanm_kws/exp/20240914_xiaoyun_finetune_sanm_6e_320_256_feats_dim40_char_t2602_offline"

config_file="sanm_6e_320_256_fdim40_t2602.yaml"
config_file="config.yaml"

model_path="./modelscope_models_kws/speech_charctc_kws_phone-xiaoyun/funasr/finetune_sanm_6e_320_256_fdim40_t2602_online_xiaoyun_commands.pt"

python -m funasr.bin.export \
    --config-path="${config_path}" \
    --config-name="${config_file}" \
    ++init_param=${model_path} \
    ++type="onnx" \
    ++quantize=true
