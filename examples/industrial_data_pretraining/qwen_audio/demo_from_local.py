#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# To install requirements: pip3 install -U "funasr[llm]"

from funasr import AutoModel

model = AutoModel(
    model="Qwen-Audio",
    model_path="/nfs/zhifu.gzf/init_model/qwen/Qwen-Audio",
)

audio_in = "https://github.com/QwenLM/Qwen-Audio/raw/main/assets/audio/1272-128104-0000.flac"
prompt = "<|startoftranscription|><|en|><|transcribe|><|en|><|notimestamps|><|wo_itn|>"

res = model.generate(input=audio_in, prompt=prompt)
print(res)
