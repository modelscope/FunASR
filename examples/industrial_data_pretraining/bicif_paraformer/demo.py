#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(model="damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                    model_revision="v2.0.0",
                    vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                    vad_model_revision="v2.0.1",
                    punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                    punc_model_revision="v2.0.1",
                    spk_model="/Users/shixian/code/modelscope_models/speech_campplus_sv_zh-cn_16k-common",
                  )

res = model(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_vad_punc_example.wav", batch_size_s=300, batch_size_threshold_s=60)
print(res)

'''try asr with speaker label with
model = AutoModel(model="damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                    model_revision="v2.0.0",
                    vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                    vad_model_revision="v2.0.1",
                    punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                    punc_model_revision="v2.0.1",
                    spk_model="/Users/shixian/code/modelscope_models/speech_campplus_sv_zh-cn_16k-common",
                    spk_mode='punc_segment',
                  )

res = model(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_speaker_demo.wav", batch_size_s=300, batch_size_threshold_s=60)
print(res)
'''