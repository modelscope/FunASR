#!/usr/bin/env bash
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

set -euo pipefail

workspace=$(pwd)

infer_dir="${workspace}/examples/industrial_data_pretraining/paraformer/outputs_lora/infer"
ref_file="${infer_dir}/text.ref"
hyp_file="${infer_dir}/text.hyp"
cer_file="${infer_dir}/text.cer"

python -m funasr.metrics.wer \
  ++ref_file="${ref_file}" \
  ++hyp_file="${hyp_file}" \
  ++cer_file="${cer_file}" \
  ++cn_postprocess=false

# Show final CER summary
if [ -f "${cer_file}" ]; then
  tail -n 3 "${cer_file}"
fi
