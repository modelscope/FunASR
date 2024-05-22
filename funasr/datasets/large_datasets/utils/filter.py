#!/usr/bin/env python


def filter(
    data, speech_length_min=100, speech_length_max=15000, token_length_min=0, token_length_max=200
):
    assert "speech" in data or "text" in data

    if "speech" in data and "text" in data:
        if "sampling_rate" in data:
            speech_length = (data["speech"].shape[0] / data["sampling_rate"]) * 1000.0
        else:
            speech_length = data["speech"].shape[0]
        num_tokens = len(data["text"])
        return (
            speech_length_min < speech_length < speech_length_max
            and token_length_min < num_tokens < token_length_max
        )
    elif "speech" in data:
        if "sampling_rate" in data:
            speech_length = (data["speech"].shape[0] / data["sampling_rate"]) * 1000.0
        else:
            speech_length = data["speech"].shape[0]
        return speech_length_min < speech_length < speech_length_max
    else:
        num_tokens = len(data["text"])
        return token_length_min < num_tokens < token_length_max
