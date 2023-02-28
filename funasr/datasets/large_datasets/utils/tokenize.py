#!/usr/bin/env python
import re
import numpy as np

def forward_segment(text, seg_dict):
    word_list = []
    i = 0
    while i < len(text):
        longest_word = text[i]
        for j in range(i + 1, len(text) + 1):
            word = text[i:j]
            if word in seg_dict:
                if len(word) > len(longest_word):
                    longest_word = word
        word_list.append(longest_word)
        i += len(longest_word)
    return word_list

def seg_tokenize(txt, seg_dict):
    out_txt = ""
    pattern = re.compile(r"([\u4E00-\u9FA5A-Za-z0-9])")
    for word in txt:
        if pattern.match(word):
            if word in seg_dict:
                out_txt += seg_dict[word] + " "
            else:
                out_txt += "<unk>" + " "
        else:
            continue
    return out_txt.strip().split()

def tokenize(data,
             vocab=None,
             seg_dict=None,
             punc_dict=None):
    assert "text" in data
    assert isinstance(vocab, dict)
    text = data["text"]
    token = []
    vad = -2

    if seg_dict is not None:
        assert isinstance(seg_dict, dict)
        txt = forward_segment("".join(text).lower(), seg_dict)
        text = seg_tokenize(txt, seg_dict)

    length = len(text)
    for i in range(length):
        x = text[i]
        if i == length-1 and "punc" in data and text[i].startswith("vad:"):
            vad = x[-1][4:]
            if len(vad) == 0:
                vad = -1
            else:
                vad = int(vad)
        elif x in vocab:
            token.append(vocab[x])
        else:
            token.append(vocab['<unk>'])

    if "punc" in data and punc_dict is not None:
        punc_token = []
        for punc in data["punc"]:
            if punc in punc_dict:
                punc_token.append(punc_dict[punc])
            else:
                punc_token.append(punc_dict["_"])
        data["punc"] =  np.array(punc_token)

    data["text"] = np.array(token)
    if vad is not -2:
        data["vad_indexes"]=np.array([vad], dtype=np.int64)
    return data
