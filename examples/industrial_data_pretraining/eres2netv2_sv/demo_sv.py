#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

# Standalone speaker verification / embedding extraction
model = AutoModel(
    model="iic/speech_eres2netv2_sv_zh-cn_16k-common",
    device="cuda:0",
)

# Extract speaker embedding
res = model.generate(
    input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
)

embedding = res[0]["spk_embedding"]
print(f"Speaker embedding shape: {embedding.shape}")
