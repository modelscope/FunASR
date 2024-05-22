#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

# Transducer, BAT and RWKV_BAT models are just same to use, use the correct model_revision
# https://modelscope.cn/models?name=transducer&page=1&tasks=auto-speech-recognition&type=audio
model = AutoModel(
    model="iic/speech_bat_asr-zh-cn-16k-aishell1-vocab4234-pytorch",
)

res = model.generate(
    input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
)
print(res)
