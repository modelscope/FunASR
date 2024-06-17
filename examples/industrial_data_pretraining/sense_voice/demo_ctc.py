#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import sys
from funasr import AutoModel

model_dir = "/nfs/beinian.lzr/workspace/models/funasr_results/asr/sense_voice/sensevoice_sanm_ctc"
input_file = (
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
)

model = AutoModel(
    model=model_dir,
)

res = model.generate(
    input=input_file,
    cache={},
    language="zh",
    text_norm="wotextnorm",
)

print(res)
