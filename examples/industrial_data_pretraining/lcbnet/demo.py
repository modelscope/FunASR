#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(model="iic/LCB-NET", model_revision="v1.0.0")


res = model.generate(
    input=(
        "https://www.modelscope.cn/api/v1/models/iic/LCB-NET/repo?Revision=master&FilePath=example/asr_example.wav",
        "https://www.modelscope.cn/api/v1/models/iic/LCB-NET/repo?Revision=master&FilePath=example/ocr.txt",
    ),
    data_type=("sound", "text"),
)

print(res)
