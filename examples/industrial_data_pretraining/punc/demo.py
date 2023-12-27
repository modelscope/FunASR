#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(model="/Users/zhifu/Downloads/modelscope_models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch")

res = model(input="/Users/zhifu/Downloads/modelscope_models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/example/punc_example.txt")
print(res)