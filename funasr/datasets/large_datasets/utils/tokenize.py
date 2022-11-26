#!/usr/bin/env python
import numpy as np

def tokenize(data,
             vocab=None):
    assert "text" in data
    assert isinstance(vocab, dict)
    text = data["text"]
    token = []
    for x in text:
        if x in vocab:
            token.append(vocab[x])
        else:
            token.append(vocab['<unk>'])

    data["text"] = np.array(token)
    return data
