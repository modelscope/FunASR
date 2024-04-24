#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(
    model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    # spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
)

res = model.generate(
    input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_vad_punc_example.wav",
    batch_size_s=300,
    batch_size_threshold_s=60,
)
print(res)
