#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import sys
from funasr import AutoModel

model_dir=sys.argv[1]
input_file=sys.argv[2]

model = AutoModel(
    model=model_dir,
)

res = model.generate(
    input=input_file,
    cache={},
)

print(res)
