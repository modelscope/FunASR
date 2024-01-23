#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", model_revision="v2.0.4")

res = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_text/punc_example.txt")
print(res)


from funasr import AutoModel

model = AutoModel(model="damo/punc_ct-transformer_cn-en-common-vocab471067-large", model_revision="v2.0.4")

res = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_text/punc_example.txt")
print(res)