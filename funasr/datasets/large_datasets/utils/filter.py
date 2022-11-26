#!/usr/bin/env python


def filter(data,
           min_length=10,
           max_length=10000,
           min_token_length=0,
           max_token_length=200):
    assert "speech" in data
    assert "text" in data

    num_frames = data["speech"].shape[0]
    num_tokens = len(data['text'])

    return min_length < num_frames < max_length and min_token_length < num_tokens < max_token_length