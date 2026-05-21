#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
Fun-ASR-Nano with Speaker Diarization

This demo shows how to use Fun-ASR-Nano with VAD + Speaker Model + Punctuation Model
to get per-sentence speaker labels.
"""

import torch
from funasr import AutoModel


def main():
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        spk_model="cam++",
        device=device,
        hub="ms",
    )

    wav_path = f"{model.model_path}/example/zh.mp3"
    res = model.generate(
        input=[wav_path],
        cache={},
        batch_size=1,
        language="中文",
    )

    # Print full text
    print("Text:", res[0]["text"])
    print()

    # Print per-sentence results with speaker labels
    print("Speaker Diarization Results:")
    for sent in res[0]["sentence_info"]:
        print(
            f"  Speaker {sent['spk']}: [{sent['start']}ms - {sent['end']}ms] {sent['text']}"
        )


if __name__ == "__main__":
    main()
