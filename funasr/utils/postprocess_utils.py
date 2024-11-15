# Copyright (c) Alibaba, Inc. and its affiliates.

import string
import logging
from typing import Any, List, Union


def isChinese(ch: str):
    if "\u4e00" <= ch <= "\u9fff" or "\u0030" <= ch <= "\u0039" or ch == "@":
        return True
    return False


def isAllChinese(word: Union[List[Any], str]):
    word_lists = []
    for i in word:
        cur = i.replace(" ", "")
        cur = cur.replace("</s>", "")
        cur = cur.replace("<s>", "")
        cur = cur.replace("<unk>", "")
        cur = cur.replace("<OOV>", "")
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
        cur = i.replace(" ", "")
        cur = cur.replace("</s>", "")
        cur = cur.replace("<s>", "")
        cur = cur.replace("<unk>", "")
        cur = cur.replace("<OOV>", "")
        word_lists.append(cur)

    if len(word_lists) == 0:
        return False

    for ch in word_lists:
        if ch.isalpha() is False and ch != "'":
            return False
        elif ch.isalpha() is True and isChinese(ch) is True:
            return False

    return True


# def abbr_dispose(words: List[Any]) -> List[Any]:
def abbr_dispose(words: List[Any], time_stamp: List[List] = None) -> List[Any]:
    words_size = len(words)
    word_lists = []
    abbr_begin = []
    abbr_end = []
    last_num = -1
    ts_lists = []
    ts_nums = []
    ts_index = 0
    for num in range(words_size):
        if num <= last_num:
            continue

        if len(words[num]) == 1 and words[num].encode("utf-8").isalpha():
            if (
                num + 1 < words_size
                and words[num + 1] == " "
                and num + 2 < words_size
                and len(words[num + 2]) == 1
                and words[num + 2].encode("utf-8").isalpha()
            ):
                # found the begin of abbr
                abbr_begin.append(num)
                num += 2
                abbr_end.append(num)
                # to find the end of abbr
                while True:
                    num += 1
                    if num < words_size and words[num] == " ":
                        num += 1
                        if (
                            num < words_size
                            and len(words[num]) == 1
                            and words[num].encode("utf-8").isalpha()
                        ):
                            abbr_end.pop()
                            abbr_end.append(num)
                            last_num = num
                        else:
                            break
                    else:
                        break

    for num in range(words_size):
        if words[num] == " ":
            ts_nums.append(ts_index)
        else:
            ts_nums.append(ts_index)
            ts_index += 1
    last_num = -1
    for num in range(words_size):
        if num <= last_num:
            continue

        if num in abbr_begin:
            if time_stamp is not None:
                begin = time_stamp[ts_nums[num]][0]
            abbr_word = words[num].upper()
            num += 1
            while num < words_size:
                if num in abbr_end:
                    abbr_word += words[num].upper()
                    last_num = num
                    break
                else:
                    if words[num].encode("utf-8").isalpha():
                        abbr_word += words[num].upper()
                num += 1
            word_lists.append(abbr_word)
            if time_stamp is not None and ts_nums[num] < len(time_stamp):
                end = time_stamp[ts_nums[num]][1]
                ts_lists.append([begin, end])
        else:
            word_lists.append(words[num])
            # length of time_stamp may not equal to length of words because of the (somehow improper) threshold set in timestamp_tools.py line 46, e.g., length of time_stamp can be zero but length of words is not.
            # Moreover, move "word_lists.append(words[num])" into if clause, to keep length of word_lists and length of ts_lists equal.
            if time_stamp is not None and ts_nums[num] < len(time_stamp) and words[num] != " ":
                begin = time_stamp[ts_nums[num]][0]
                end = time_stamp[ts_nums[num]][1]
                ts_lists.append([begin, end])
                begin = end

    if time_stamp is not None:
        return word_lists, ts_lists
    else:
        return word_lists


def sentence_postprocess(words: List[Any], time_stamp: List[List] = None):
    middle_lists = []
    word_lists = []
    word_item = ""
    ts_lists = []

    # wash words lists
    for i in words:
        word = ""
        if isinstance(i, str):
            word = i
        else:
            word = i.decode("utf-8")

        if word in ["<s>", "</s>", "<unk>", "<OOV>"]:
            continue
        else:
            middle_lists.append(word)

    # all chinese characters
    if isAllChinese(middle_lists):
        for i, ch in enumerate(middle_lists):
            word_lists.append(ch.replace(" ", ""))
        if time_stamp is not None:
            ts_lists = time_stamp

    # all alpha characters
    elif isAllAlpha(middle_lists):
        ts_flag = True
        for i, ch in enumerate(middle_lists):
            if ts_flag and time_stamp is not None:
                begin = time_stamp[i][0]
                end = time_stamp[i][1]
            word = ""
            if "@@" in ch:
                word = ch.replace("@@", "")
                word_item += word
                if time_stamp is not None:
                    ts_flag = False
                    end = time_stamp[i][1]
            else:
                word_item += ch
                word_lists.append(word_item)
                word_lists.append(" ")
                word_item = ""
                if time_stamp is not None:
                    ts_flag = True
                    end = time_stamp[i][1]
                    ts_lists.append([begin, end])
                    begin = end

    # mix characters
    else:
        alpha_blank = False
        ts_flag = True
        begin = -1
        end = -1
        for i, ch in enumerate(middle_lists):
            if ts_flag and time_stamp is not None:
                begin = time_stamp[i][0]
                end = time_stamp[i][1]
            word = ""
            if isAllChinese(ch):
                if alpha_blank is True:
                    word_lists.pop()
                word_lists.append(ch)
                alpha_blank = False
                if time_stamp is not None:
                    ts_flag = True
                    ts_lists.append([begin, end])
                    begin = end
            elif "@@" in ch:
                word = ch.replace("@@", "")
                word_item += word
                alpha_blank = False
                if time_stamp is not None:
                    ts_flag = False
                    end = time_stamp[i][1]
            elif isAllAlpha(ch):
                word_item += ch
                word_lists.append(word_item)
                word_lists.append(" ")
                word_item = ""
                alpha_blank = True
                if time_stamp is not None:
                    ts_flag = True
                    end = time_stamp[i][1]
                    ts_lists.append([begin, end])
                    begin = end
            else:
                word_lists.append(ch)

    if time_stamp is not None:
        word_lists, ts_lists = abbr_dispose(word_lists, ts_lists)
        real_word_lists = []
        for ch in word_lists:
            if ch != " ":
                real_word_lists.append(ch)
        sentence = " ".join(real_word_lists).strip()
        return sentence, ts_lists, real_word_lists
    else:
        word_lists = abbr_dispose(word_lists)
        real_word_lists = []
        for ch in word_lists:
            if ch != " ":
                real_word_lists.append(ch)
        sentence = "".join(word_lists).strip()
        return sentence, real_word_lists


def sentence_postprocess_sentencepiece(words):
    middle_lists = []
    word_lists = []
    word_item = ""

    # wash words lists
    for i in words:
        word = ""
        if isinstance(i, str):
            word = i
        else:
            word = i.decode("utf-8")

        if word in ["<s>", "</s>", "<unk>", "<OOV>"]:
            continue
        else:
            middle_lists.append(word)

    # all alpha characters
    for i, ch in enumerate(middle_lists):
        word = ""
        if "\u2581" in ch and i == 0:
            word_item = ""
            word = ch.replace("\u2581", "")
            word_item += word
        elif "\u2581" in ch and i != 0:
            word_lists.append(word_item)
            word_lists.append(" ")
            word_item = ""
            word = ch.replace("\u2581", "")
            word_item += word
        else:
            word_item += ch
    if word_item is not None:
        word_lists.append(word_item)
    # word_lists = abbr_dispose(word_lists)
    real_word_lists = []
    for ch in word_lists:
        if ch != " ":
            if ch == "i":
                ch = ch.replace("i", "I")
            elif ch == "i'm":
                ch = ch.replace("i'm", "I'm")
            elif ch == "i've":
                ch = ch.replace("i've", "I've")
            elif ch == "i'll":
                ch = ch.replace("i'll", "I'll")
            real_word_lists.append(ch)
    sentence = "".join(word_lists)
    return sentence, real_word_lists


emo_dict = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}

event_dict = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "🤧",
}

lang_dict = {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {
    "🎼",
    "👏",
    "😀",
    "😭",
    "🤧",
    "😷",
}


def format_str_v2(s):
    sptk_dict = {}
    for sptk in emoji_dict:
        sptk_dict[sptk] = s.count(sptk)
        s = s.replace(sptk, "")
    emo = "<|NEUTRAL|>"
    for e in emo_dict:
        if sptk_dict[e] > sptk_dict[emo]:
            emo = e
    for e in event_dict:
        if sptk_dict[e] > 0:
            s = event_dict[e] + s
    s = s + emo_dict[emo]

    for emoji in emo_set.union(event_set):
        s = s.replace(" " + emoji, emoji)
        s = s.replace(emoji + " ", emoji)
    return s.strip()


def rich_transcription_postprocess(s):
    def get_emo(s):
        return s[-1] if s[-1] in emo_set else None

    def get_event(s):
        return s[0] if s[0] in event_set else None

    s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
    for lang in lang_dict:
        s = s.replace(lang, "<|lang|>")
    s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
    new_s = " " + s_list[0]
    cur_ent_event = get_event(new_s)
    for i in range(1, len(s_list)):
        if len(s_list[i]) == 0:
            continue
        if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
            s_list[i] = s_list[i][1:]
        # else:
        cur_ent_event = get_event(s_list[i])
        if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
            new_s = new_s[:-1]
        new_s += s_list[i].strip().lstrip()
    new_s = new_s.replace("The.", " ")
    return new_s.strip()
