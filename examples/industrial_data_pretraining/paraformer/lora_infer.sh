#!/usr/bin/env bash
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

set -euo pipefail

workspace=$(pwd)

# model path and config (from training output)
model_dir="${workspace}/examples/industrial_data_pretraining/paraformer/outputs_lora"
init_param="${model_dir}/model.pt"
config_path="${model_dir}"
config_name="config.yaml"

# input jsonl (must contain source/target)
input_jsonl="${workspace}/data/list/val.jsonl"

# output directory
output_dir="${model_dir}/infer"

# device
device="cuda:0"

python ${workspace}/examples/industrial_data_pretraining/paraformer/lora_infer.py \
  --model "${model_dir}" \
  --config-path "${config_path}" \
  --config-name "${config_name}" \
  --init-param "${init_param}" \
  --input-jsonl "${input_jsonl}" \
  --output-dir "${output_dir}" \
  --device "${device}" \
  --batch-size 1
