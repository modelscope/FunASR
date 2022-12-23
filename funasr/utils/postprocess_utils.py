# Copyright (c) Alibaba, Inc. and its affiliates.

import string
from typing import Any, List, Union


def isChinese(ch: str):
    if '\u4e00' <= ch <= '\u9fff':
        return True
    return False


def isAllChinese(word: Union[List[Any], str]):
    word_lists = []
    for i in word:
        cur = i.replace(' ', '')
        cur = cur.replace('</s>', '')
        cur = cur.replace('<s>', '')
        word_lists.append(cur)

    if len(word_lists) == 0:
        return False

    for ch in word_lists:
        if isChinese(ch) is False:
            return False
    return True


def isAllAlpha(word: Union[List[Any], str]):
    word_lists = []
    for i in word:
        cur = i.replace(' ', '')
        cur = cur.replace('</s>', '')
        cur = cur.replace('<s>', '')
        word_lists.append(cur)

    if len(word_lists) == 0:
        return False

    for ch in word_lists:
        if ch.isalpha() is False and ch != "'":
            return False
        elif ch.isalpha() is True and isChinese(ch) is True:
            return False

    return True


def abbr_dispose(words: List[Any]) -> List[Any]:
    words_size = len(words)
    word_lists = []
    abbr_begin = []
    abbr_end = []
    last_num = -1
    for num in range(words_size):
        if num <= last_num:
            continue

        if len(words[num]) == 1 and words[num].encode('utf-8').isalpha():
            if num + 1 < words_size and words[
                    num + 1] == ' ' and num + 2 < words_size and len(
                        words[num +
                              2]) == 1 and words[num +
                                                 2].encode('utf-8').isalpha():
                # found the begin of abbr
                abbr_begin.append(num)
                num += 2
                abbr_end.append(num)
                # to find the end of abbr
                while True:
                    num += 1
                    if num < words_size and words[num] == ' ':
                        num += 1
                        if num < words_size and len(
                                words[num]) == 1 and words[num].encode(
                                    'utf-8').isalpha():
                            abbr_end.pop()
                            abbr_end.append(num)
                            last_num = num
                        else:
                            break
                    else:
                        break

    last_num = -1
    for num in range(words_size):
        if num <= last_num:
            continue

        if num in abbr_begin:
            word_lists.append(words[num].upper())
            num += 1
            while num < words_size:
                if num in abbr_end:
                    word_lists.append(words[num].upper())
                    last_num = num
                    break
                else:
                    if words[num].encode('utf-8').isalpha():
                        word_lists.append(words[num].upper())
                num += 1
        else:
            word_lists.append(words[num])

    return word_lists


def sentence_postprocess(words: List[Any]):
    middle_lists = []
    word_lists = []
    word_item = ''

    # wash words lists
    for i in words:
        word = ''
        if isinstance(i, str):
            word = i
        else:
            word = i.decode('utf-8')

        if word in ['<s>', '</s>', '<unk>']:
            continue
        else:
            middle_lists.append(word)

    # all chinese characters
    if isAllChinese(middle_lists):
        for ch in middle_lists:
            word_lists.append(ch.replace(' ', ''))

    # all alpha characters
    elif isAllAlpha(middle_lists):
        for ch in middle_lists:
            word = ''
            if '@@' in ch:
                word = ch.replace('@@', '')
                word_item += word
            else:
                word_item += ch
                word_lists.append(word_item)
                word_lists.append(' ')
                word_item = ''

    # mix characters
    else:
        alpha_blank = False
        for ch in middle_lists:
            word = ''
            if isAllChinese(ch):
                if alpha_blank is True:
                    word_lists.pop()
                word_lists.append(ch)
                alpha_blank = False
            elif '@@' in ch:
                word = ch.replace('@@', '')
                word_item += word
                alpha_blank = False
            elif isAllAlpha(ch):
                word_item += ch
                word_lists.append(word_item)
                word_lists.append(' ')
                word_item = ''
                alpha_blank = True
            else:
                raise ValueError('invalid character: {}'.format(ch))

    word_lists = abbr_dispose(word_lists)
    sentence = ''.join(word_lists).strip()
    return sentence
