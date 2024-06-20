#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(
    model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 60000},
    punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    # spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
)

res = model.generate(
    input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
    cache={},
)

print(res)


""" call english model like below for detailed timestamps
# choose english paraformer model first
# iic/speech_paraformer_asr-en-16k-vocab4199-pytorch
res = model.generate(
    input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_en.wav",
    cache={},
    pred_timestamp=True,
    return_raw_text=True,
    sentence_timestamp=True,
    en_post_proc=True,
)
"""

""" can not use currently
from funasr import AutoFrontend

frontend = AutoFrontend(model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")

fbanks = frontend(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav", batch_size=2)

for batch_idx, fbank_dict in enumerate(fbanks):
    res = model.generate(**fbank_dict)
    print(res)
"""
