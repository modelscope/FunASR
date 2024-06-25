#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import json
import os
import sys

from funasr import AutoModel

if len(sys.argv) > 1:
    ckpt_dir = sys.argv[1]
    ckpt_id = sys.argv[2]
    jsonl = sys.argv[3]
    output_dir = sys.argv[4]
    device = sys.argv[5]
else:
    ckpt_dir = "/nfs/beinian.lzr/workspace/GPT-4o/Exp/exp6/5m-8gpu/exp6_speech2text_linear_ddp_0609"
    ckpt_id = "model.pt.ep0.90000"
    jsonl = "/nfs/beinian.lzr/workspace/GPT-4o/Data/Speech2Text/TestData/aishell1_test_speech2text.jsonl"
    dataset = jsonl.split("/")[-1]
    output_dir = os.path.join(ckpt_dir, f"inference-{ckpt_id}", dataset)
    device = "cuda:0"


model = AutoModel(
    model=ckpt_dir,
    init_param=f"{os.path.join(ckpt_dir, ckpt_id)}",
    output_dir=output_dir,
    device=device,
    fp16=False,
    bf16=False,
    llm_dtype="bf16",
)


with open(jsonl, "r") as f:
    lines = f.readlines()

tearchforing = False
for i, line in enumerate(lines):
    data_dict = json.loads(line.strip())
    data = data_dict["messages"]

    res = model.generate(
        input=[data],
        tearchforing=tearchforing,
        cache={},
    )

    print(res)
