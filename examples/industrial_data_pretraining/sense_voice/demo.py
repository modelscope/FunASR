#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import sys
from funasr import AutoModel

model_dir = "iic/SenseVoiceSmall"
input_file = (
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
)

model = AutoModel(
    model=model_dir,
)

res = model.generate(
    input=input_file,
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
)

print(res)
