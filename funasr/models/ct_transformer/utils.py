#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import re


def split_to_mini_sentence(words: list, word_limit: int = 20):
    assert word_limit > 1
    if len(words) <= word_limit:
        return [words]
    sentences = []
    length = len(words)
    sentence_len = length // word_limit
    for i in range(sentence_len):
        sentences.append(words[i * word_limit : (i + 1) * word_limit])
    if length % word_limit > 0:
        sentences.append(words[sentence_len * word_limit :])
    return sentences


def split_words(text: str, jieba_usr_dict=None, **kwargs):
    if jieba_usr_dict:
        input_list = text.split()
        token_list_all = []
        langauge_list = []
        token_list_tmp = []
        language_flag = None
        for token in input_list:
            if isEnglish(token) and language_flag == "Chinese":
                token_list_all.append(token_list_tmp)
                langauge_list.append("Chinese")
                token_list_tmp = []
            elif not isEnglish(token) and language_flag == "English":
                token_list_all.append(token_list_tmp)
                langauge_list.append("English")
                token_list_tmp = []

            token_list_tmp.append(token)

            if isEnglish(token):
                language_flag = "English"
            else:
                language_flag = "Chinese"

        if token_list_tmp:
            token_list_all.append(token_list_tmp)
            langauge_list.append(language_flag)

        result_list = []
        for token_list_tmp, language_flag in zip(token_list_all, langauge_list):
            if language_flag == "English":
                result_list.extend(token_list_tmp)
            else:
                seg_list = jieba_usr_dict.cut(join_chinese_and_english(token_list_tmp), HMM=False)
                result_list.extend(seg_list)

        return result_list

    else:
        words = []
        segs = text.split()
        for seg in segs:
            # There is no space in seg.
            current_word = ""
            for c in seg:
                if len(c.encode()) == 1:
                    # This is an ASCII char.
                    current_word += c
                else:
                    # This is a Chinese char.
                    if len(current_word) > 0:
                        words.append(current_word)
                        current_word = ""
                    words.append(c)
            if len(current_word) > 0:
                words.append(current_word)
        return words


def isEnglish(text: str):
    if re.search("^[a-zA-Z']+$", text):
        return True
    else:
        return False


def join_chinese_and_english(input_list):
    line = ""
    for token in input_list:
        if isEnglish(token):
            line = line + " " + token
        else:
            line = line + token

    line = line.strip()
    return line
