#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(model="damo/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                  model_revision="v2.0.4",
                  vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                  vad_model_revision="v2.0.4",
                  punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                  punc_model_revision="v2.0.4",
                  spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                  spk_model_revision="v2.0.2"
                  )

res = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
                     hotword='达摩院 磨搭')
print(res)