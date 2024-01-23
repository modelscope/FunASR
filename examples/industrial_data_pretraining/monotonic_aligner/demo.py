#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(model="damo/speech_timestamp_prediction-v1-16k-offline", model_revision="v2.0.4")

res = model.generate(input=("https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
                   "欢迎大家来到魔搭社区进行体验"),
            data_type=("sound", "text"),
            batch_size=2,
            )
print(res)
