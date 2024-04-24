#!/usr/bin/env python
import re
import numpy as np
from funasr.datasets.large_datasets.utils.hotword_utils import sample_hotword


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
    pattern = re.compile(r"^[\u4E00-\u9FA50-9]+$")
    out_txt = ""
    for word in txt:
        word = word.lower()
        if word in seg_dict:
            out_txt += seg_dict[word] + " "
        else:
            if pattern.match(word):
                for char in word:
                    if char in seg_dict:
                        out_txt += seg_dict[char] + " "
                    else:
                        out_txt += "<unk>" + " "
            else:
                out_txt += "<unk>" + " "
    return out_txt.strip().split()


def tokenize(data, vocab=None, seg_dict=None, punc_dict=None, bpe_tokenizer=None, hw_config=None):
    assert "text" in data
    assert isinstance(vocab, dict)
    text = data["text"]
    token = []
    vad = -2
    if bpe_tokenizer is not None:
        text = bpe_tokenizer.text2tokens(" ".join(text))
    if seg_dict is not None:
        assert isinstance(seg_dict, dict)
        text = seg_tokenize(text, seg_dict)

    length = len(text)
    if "hw_tag" in data:
        pre_index = None
        if hw_config["pre_hwlist"] is not None and hw_config["pre_prob"] > 0:
            # enable preset hotword detect in sampling
            for hw in hw_config["pre_hwlist"]:
                hw = " ".join(seg_tokenize(hw, seg_dict))
                _find = " ".join(text).find(hw)
                if _find != -1:
                    # _find = text[:_find].count(" ")  # bpe sometimes
                    pre_index = [_find, _find + max(hw.count(" "), 1)]
                    break
        hotword_indxs = sample_hotword(length, **hw_config, pre_index=pre_index)
        data["hotword_indxs"] = hotword_indxs
        del data["hw_tag"]
    for i in range(length):
        x = text[i]
        if i == length - 1 and "punc" in data and x.startswith("vad:"):
            vad = x[4:]
            if len(vad) == 0:
                vad = -1
            else:
                vad = int(vad)
        elif x in vocab:
            token.append(vocab[x])
        else:
            token.append(vocab["<unk>"])

    if "punc" in data and punc_dict is not None:
        punc_token = []
        for punc in data["punc"]:
            if punc in punc_dict:
                punc_token.append(punc_dict[punc])
            else:
                punc_token.append(punc_dict["_"])
        data["punc"] = np.array(punc_token)

    data["text"] = np.array(token)
    if vad is not -2:
        data["vad_indexes"] = np.array([vad], dtype=np.int64)
    return data
