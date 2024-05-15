#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(
    model="/Users/zhifu/Downloads/modelscope_models/SenseVoiceModelscope",
    # vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    # vad_kwargs={"max_single_segment_time": 30000},
)


input_wav = (
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
)

DecodingOptions = {
    "task": ("ASR", "AED", "SER"),
    "language": "auto",
    "fp16": True,
    "gain_event": True,
    "beam_size": 5,
}

res = model.generate(input=input_wav, batch_size_s=0, DecodingOptions=DecodingOptions)
print(res)
