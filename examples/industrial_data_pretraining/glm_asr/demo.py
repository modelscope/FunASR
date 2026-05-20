#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# Requirements: pip install transformers>=5.0.0 torchaudio

from funasr import AutoModel

# Initialize GLM-ASR model
# hub: "ms" for ModelScope (default), "hf" for HuggingFace
model = AutoModel(
    model="zai-org/GLM-ASR-Nano-2512",
    hub="hf",
    device="cuda:0",
    dtype="bf16",
)

# Chinese speech recognition
res = model.generate(
    input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
)
print("Chinese:", res[0]["text"])

# English speech recognition
res = model.generate(
    input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_en.wav",
)
print("English:", res[0]["text"])
