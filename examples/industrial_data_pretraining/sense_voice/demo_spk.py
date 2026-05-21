#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
SenseVoice with Speaker Diarization

This demo shows how to use SenseVoice with VAD + Speaker Model + Punctuation Model
to get per-sentence speaker labels.
"""

import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


def main():
    model_dir = "iic/SenseVoiceSmall"
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = AutoModel(
        model=model_dir,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        spk_model="cam++",
        device=device,
    )

    wav_path = f"{model.model_path}/example/zh.mp3"
    res = model.generate(
        input=wav_path,
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )

    # Print full text
    text = rich_transcription_postprocess(res[0]["text"])
    print("Text:", text)
    print()

    # Print per-sentence results with speaker labels
    print("Speaker Diarization Results:")
    for sent in res[0]["sentence_info"]:
        sent_text = rich_transcription_postprocess(sent["text"])
        print(
            f"  Speaker {sent['spk']}: [{sent['start']}ms - {sent['end']}ms] {sent_text}"
        )


if __name__ == "__main__":
    main()
