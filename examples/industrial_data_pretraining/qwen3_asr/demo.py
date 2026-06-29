#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# To install requirements: pip install -U "qwen-asr==0.0.6" "transformers==4.57.6" accelerate

from funasr import AutoModel

# Initialize Qwen3-ASR model
# hub: "ms" for ModelScope (default), "hf" for HuggingFace
model = AutoModel(
    model="Qwen/Qwen3-ASR-1.7B",
    hub="hf",
    device="cuda:0",
    dtype="bf16",
)

# Chinese speech recognition
res = model.generate(
    input="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
    language="Chinese",
)
print("Chinese:", res[0]["text"])

# English speech recognition
res = model.generate(
    input="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    language="English",
)
print("English:", res[0]["text"])

# Auto language detection (supports 52 languages)
res = model.generate(
    input="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
)
print("Auto:", res[0]["text"], "| Language:", res[0].get("language", ""))
