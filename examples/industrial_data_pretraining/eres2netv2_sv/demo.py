#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

# ASR with speaker diarization using ERes2NetV2
model = AutoModel(
    model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 60000},
    punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    spk_model="iic/speech_eres2netv2_sv_zh-cn_16k-common",
    device="cuda:0",
)

res = model.generate(
    input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
    batch_size_s=300,
)

for sentence in res:
    print(f"[Speaker {sentence.get('spk', '?')}] {sentence['text']}")
