#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import json
import os
import sys

from funasr import AutoModel

ckpt_dir = "/nfs/beinian.lzr/workspace/GPT-4o/Exp/exp6/5m-8gpu/exp6_speech2text_linear_ddp_0609"
ckpt_id = "model.pt.ep0.90000"
jsonl = (
    "/nfs/beinian.lzr/workspace/GPT-4o/Data/Speech2Text/TestData/aishell1_test_speech2text.jsonl"
)
output_dir = f"{os.path.join(ckpt_dir, ckpt_id)}"
device = "cuda:0"

ckpt_dir = sys.argv[1]
ckpt_id = sys.argv[2]
jsonl = sys.argv[3]
output_dir = sys.argv[4]
device = sys.argv[5]

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

    key = f"dialog_{i}"

    data_dict = json.loads(line.strip())
    data = data_dict["messages"]

    contents = model.model.data_template(data)

    system = contents["system"]
    user = contents["user"]
    assistant = contents["assistant"]

    system_i, user_i, assistant_i = [], [], []

    for j, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
        key = f"{key}_turn_{j}"

        system_i += [system_prompt]
        user_i += [user_prompt]
        assistant_i += [target_out]

        contents_i = {"system": system_i, "user": user_i, "assistant": assistant}

        res = model.generate(
            input=[contents_i],
            tearchforing=tearchforing,
            cache={},
            key=key,
        )

        print(res)
