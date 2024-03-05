#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

# model = AutoModel(model="Whisper-small", hub="openai")
# model = AutoModel(model="Whisper-medium", hub="openai")
model = AutoModel(model="Whisper-large-v3", hub="openai")
# model = AutoModel(model="Whisper-large-v3", hub="openai")

res = model.generate(
	language=None,
	task="transcribe",
	input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")
print(res)
