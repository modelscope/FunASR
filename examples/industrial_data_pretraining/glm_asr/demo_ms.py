#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# Requirements: pip install transformers>=5.0.0 torchaudio

from funasr import AutoModel

# Download from ModelScope (recommended for Chinese users)
# ModelScope model id: ZhipuAI/GLM-ASR-Nano-2512
model = AutoModel(
    model="ZhipuAI/GLM-ASR-Nano-2512",
    hub="ms",
    device="cuda:0",
    dtype="bf16",
)

res = model.generate(
    input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
)
print(res[0]["text"])
