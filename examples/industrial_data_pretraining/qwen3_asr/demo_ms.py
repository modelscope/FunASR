#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# To install requirements: pip install -U "qwen-asr==0.0.6" "transformers==4.57.6" accelerate

from funasr import AutoModel

# Download from ModelScope (recommended for Chinese users)
model = AutoModel(
    model="Qwen/Qwen3-ASR-1.7B",
    hub="ms",
    device="cuda:0",
    dtype="bf16",
)

res = model.generate(
    input="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
    language="Chinese",
)
print(res[0]["text"])
