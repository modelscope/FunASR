#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# To install requirements: pip3 install -U "funasr[llm]"

from funasr import AutoModel

model = AutoModel(
    model="Qwen-Audio-Chat",
    model_path="/nfs/zhifu.gzf/init_model/qwen/Qwen-Audio-Chat",
)

audio_in = (
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
)

# 1st dialogue turn
prompt = "what does the person say?"
cache = {"history": None}
res = model.generate(input=audio_in, prompt=prompt, cache=cache)
print(res)


# 2nd dialogue turn
prompt = 'Find the start time and end time of the word "middle classes"'
res = model.generate(input=None, prompt=prompt, cache=cache)
print(res)
