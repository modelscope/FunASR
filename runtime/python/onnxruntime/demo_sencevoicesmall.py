#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os
import torch
from pathlib import Path
from funasr import AutoModel
from funasr_onnx import SenseVoiceSmallONNX as SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess


model_dir = "iic/SenseVoiceSmall"
model = AutoModel(
    model=model_dir,
    device="cuda:0",
)

res = model.export(type="onnx", quantize=False)

# export model init
model_path = "{}/.cache/modelscope/hub/{}".format(Path.home(), model_dir)
model_bin = SenseVoiceSmall(model_path)

# build tokenizer
try:
    from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
    tokenizer = SentencepiecesTokenizer(bpemodel=os.path.join(model_path, "chn_jpn_yue_eng_ko_spectok.bpe.model"))
except:
    tokenizer = None

# inference
wav_or_scp = "/Users/shixian/Downloads/asr_example_hotword.wav"
language_list = [0]
textnorm_list = [15]
res = model_bin(wav_or_scp, language_list, textnorm_list, tokenizer=tokenizer)
print([rich_transcription_postprocess(i) for i in res])
