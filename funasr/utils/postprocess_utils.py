# Copyright (c) Alibaba, Inc. and its affiliates.

import string
import logging
from typing import Any, List, Union


def isChinese(ch: str):
    if '\u4e00' <= ch <= '\u9fff' or '\u0030' <= ch <= '\u0039' or ch == '@':
        return True
    return False


def isAllChinese(word: Union[List[Any], str]):
    word_lists = []
    for i in word:
        cur = i.replace(' ', '')
        cur = cur.replace('</s>', '')
        cur = cur.replace('<s>', '')
        cur = cur.replace('<unk>', '')
        cur = cur.replace('<OOV>', '')
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
        cur = cur.replace('<unk>', '')
        cur = cur.replace('<OOV>', '')
        word_lists.append(cur)

    if len(word_lists) == 0:
        return False

    for ch in word_lists:
        if ch.isalpha() is False and ch != "'":
            return False
        elif ch.isalpha() is True and isChinese(ch) is True:
            return False

    return True

def isMy(word: Union[List[Any], str]):
    my_char = ['စေ', 'ကို', 'ဖြစ်', 'ဌ', 'ေါ', 'ရင်း', 'w', 'ပုံ', 'ပတ်', 'လာ', 'စီး', 'ဘက်', 'က်', 'ုံ', 'ဏာ', 'ဖော်', 'အတွင်း', 'r', '၀', 'တို', 'ပြန်', 'ီး', 'h', 'ဖြ', 'က္ခ', 'မ္', 'အထိ', 'ဆွ', 'ပေး', 'တူ', '၈', 'မ်း', 'ဠ', 'ထို', 'စွ', 'ထားသည်', 'အခြေ', 'စာ', 'တို့သည်', 'အက', 'ရဲ့', 'ွ', 'င', 'o', 'ုတ်', 'လွ', 'ပင်', 'နိုင်ငံ', 'ပါတယ်', 'ကား', 'အဖွဲ့', 'အခြား', 'ယ့်', 'ပို', 'ည်း', 'ယာ', 'ဆုံး', '၄', 'ကြောင်း', 'တပ်', 'အနေ', 'ဣ', '၏', 'ိုး', 'ချုပ်', 'နေ', 'စ', 'ဟ', 'လွှ', 'အဆ', '၌', 'ဣ', 'နိုင်', 'တည်', 'တွ', 'အချိန်', 'ပဲ', 'ဝင်', 'ဒီ', 'သူ', 'l', 'ဏ', 'ဲ့', 'အထ', 'ပည', 'စိတ်', 'ကြသည်', 'ဩ', 's', 't', 'သေ', 'လျ', 'ိုက်', 'များသည်', 'ငါ', 'ို', 'ထဲ', 'လာ', 'ဝန်', 'ဓ', 'ခဲ့', 'စွာ', 'မ', 'နှင့်', '၎', 'အစိုး', 'ရာ', '၉', 'တယ်', '၎င်း', '၅', 'ပညာ', 'ကြီး', 'သို့မဟုတ်', '၍', 'ို', 'မူ', 'f', 'ခု', 'ိမ', 'c', 'ုပ်', 'l', 'အမ', 'နောက်', 'သော', 'ုန်း', 'ှ', 'ကြ', 'တ', 'ဌာ', 'p', 'ပေါ်', 'h', 'ပင်', 'ဲ', 'ဒီ', 'ဈ', 'လက္ခ', 'r', 'ပြ', 'ဒေ', 'မှ', 'ရှ', 'လျှ', 'လေ့', 'ရောဂါ', 'ော်', 'လည်', 'ဖွဲ့', 'မ်', 'သိ', 'ထုတ်', 'ရိ', 'အား', 'ာ', 'လ', 'ုပ', 'ကျော်', 'အေ', 'g', 'နာ', 'ရီ', 'ရာ', 'v', 'သော', 'လူ', 'တာ', 'ီး', 'j', 'ကာ', 'ရွာ', 'မျက်နှ', 'ယ', 'q', 'ပ်', 'ဌ', 'ဒ', 'ဝ', 'အခြ', 'd', 'ဍ', 'လှ', 'သည်', 'မြန်မာ', 'ယ်', 'ဖ', 'ဦ', 'ါ', 'ဲ့', 'ပျ', 'ရ', 'မိ', 'ပြီး', 'ကို', 'လည်း', 'ဇ', 'မြ', 'နွေး', 'ဘ', 'အသုံးပြု', 'ော', 'ချ', 'မွ', 'လဲ', 'န့်', 'ဂ', 'ည်', 'ကန်', 'က', 'ဗ', 'ေး', 'လု', 'တီ', 'မြို့', 'ိတ်', 'ဘ', 'အရေး', 'ုပ်', 'p', 'ဖ', 'င်', 'သွား', 'တိုင်း', '၃', 'ဿ', 'စေ', 'ဖြတ်', 'ဖွ', 'k', 'သူ', 'တစ်', 'ြ', 'စက်', 'ကြီး', 'ပြည်နယ်', 'ဝါ', 'ဘူး', 'ထ', 'ငြ', 'တော်', 'ကျ', 'ကိ', 'ဈ', 'i', 'အဲ', 'o', 'ေ', 'b', 'င်္', 'ဒါ', 'ညီ', 'w', 'ငွ', 'သ', 'မှတ်', 'ြ', 'ခြား', 'ကြောင့်', 'နာ', 'မှာ', 'f', 'ပွ', 'ကျွန်ုပ်', '၁၀', 'ခေါ', 'ယ်', '၊', 'ှ', 'အဓ', 'နိုင်', 'သက်', 'ပေး', 'a', 'ကျွန်', 'd', 'ထွ', 't', 'n', 'ဠ', 'အရာ', 'ခွ', 'ထ', 'ိုင်', 'ည့်', 'ိမ်', 'သည်', 'တွေ', 'အချ', 'ကား', 'ဗ', 'သုံး', 'အ', 'သူများ', 'ိုက်', 'အမျိုး', '၇', 'စား', 'ဪ', 'တဲ့', 'များ', 's', 'ေ', 'ယ', 'အဓိ', 'နိုင်သည်', 'ဎ', '္', 'ခ', 'စည်း', '၂', 'န်', 'ရ', 'ခရ', 'နည်း', 'အကြ', 'န်', 'တိ', 'န', 'ပြော', 'မှတ်', 'ောင်း', 'န်း', 'ရေး', 'ဆို', 'ူး', 'ရောက်', 'ထို့', 'ည်', 'ပြန်', 'ဒေ', 'စစ်', 'ဟာ', 'ဏ', 'ပြင်', 'ဆိုင်ရာ', 'z', 'ခုနှစ်', 'နဲ့', 'သ', 'စ္', 'ော', 'c', 'လုပ်', 'မျိုး', 'ကေ', 'ဘာ', 'များ', 'ိတ်', 'စား', 'တို', 'ယား', 'တာ', 'q', 'k', 'ဎ', 'င်း', 'စ်', 'အားလုံး', 'အခ', 'အ', 'အသ', 'ချက်', 'ဆက်', 'ည်း', 'ို့', 'လုပ်', 'ပွဲ', 'ကု', 'စပ်', 'အန', 'ပိုင်း', 'm', 'ဖို့', 'ဃ', 'ု', 'တင်', 'ပ္', 'ပြင်', 'း', 'နယ်', 'm', 'ား', 'အနေ', 'အတွက်', 'င့်', 'ရှိ', 'ခြ', '၄', 'v', 'မဟ', 'က်', 'လေး', 'တိုက်', 'ံ', 'သမ', 'ိုင်', '၏', 'j', 'ကြား', 'ကောင်း', 'ဦး', 'တစ်ခု', 'ထုတ်', 'ကု', 'u', 'မည်', 'ရွ', 'မင်း', 'ပ', 'စ်', 'ဆိုင်', 'ဆက်', 'တွင်', 'မြို့နယ်', 'စု', 'ဟ', 'တစ်ဦး', 'လက်', 'ုတ်', 'သူတို့', '်', 'သာ', 'ဩ', 'မာ', 'ယူ', 'ဤ', '့', 'မန', 'ရောဂ', 'သွ', 'ဝင်', 'အတ', 'ရက်', 'မျက်', 'ထား', '၁', 'တ်', 'တို့', 'ဤ', 'နေ့', 'ရင်', '…', 'ထား', 'ဧ', 'ပါး', 'မာ', 'သား', 'ဆောင်', 'မှု', 'ဂ', 'င', 'အား', 'ဇ', 'ောက်', 'သိ', 'ူ', 'စ', '်', 'အတွ', 'e', 'ဉ', 'ဆို', 'ည', 'သည့်', 'က', 'ဖြစ်', 'တရား', 'ရေ', 'ရပ်', 'ပါ', 'ကူး', 'ကမ္', 'သား', 'ကျ', 'မျိုး', 'ခဲ့', 'ောင်', 'ျ', 'ို့', 'ချ', 'အစိုးရ', 'သတ', 'ပြု', 'ကျွ', 'အရ', 'ိုလ', 'ပြီး', 'လုံး', 'လို', 'z', 'ောက်', 'ဥ', 'တမ်း', 'တရ', 'ကျွန်ုပ်တို့', 'နှစ်', 'ိန်', 'ခံ', 'ကာ', 'ဥပ', 'အသုံး', 'တော်', 'ူး', 'ဘာ', 'ပါ', 'ိပ်', 'ား', 'တ', 'နွ', '္တ', 'ဝ', 'လို့', 'ေ့', 'န္', 'e', 'ေ့', 'စီး', 'y', 'ပြား', 'ပိုး', 'အရ', 'အဖြစ်', 'g', 'ဓာ', 'ပြ', 'တစ်', 'မှ', 'ဖွဲ့', '၍', 'ခြင်း', 'ုံး', 'ဆင်', 'ွန်', 'အလ', 'တော့', 'မို', 'လ', 'စာ', 'ဿ', 'အမြ', 'တင်', 'အကျ', 'ဲ', 'ူ', 'အုပ်', 'y', 'u', 'ဒါ', 'ရော', 'ပို', 'လိုအ', 'a', 'ိ', 'ဆ', '့', 'x', 'လို', '့်', 'ပြည်', 'ယူ', 'ဃ', 'ဆေး', 'ခံ', 'မွ', 'ဘဲ', 'ုံး', 'ော်', 'လိုက်', 'နေ', 'မျ', 'နိုင်င', 'ံ့', 'မှာ', 'နည်း', 'ရန်', 'လက္ခဏာ', 'ဥ', 'င့်', 'ပညာ', 'ပ်', 'အားဖြင့်', 'နှစ်', 'ဆွေး', 'ဖြစ်သည်', 'ဒ', 'ီ', 'နစ်', 'ကျင်', 'ဋ', 'အများ', 'ဉ', 'မ်း', 'န့်', 'ကွ', 'သို့', 'b', '၀', 'ခု', 'ပုံ', 'တော', 'အာ', 'ဖြင့်', 'ဧ', 'သွား', 'အခါ', 'မ', 'င်း', 'ာ', 'ဆ', 'i', 'ဓ', '၆', 'ကြော', 'ရိ', 'သြ', 'တွေ့', '၌', 'ထိ', 'က္', 'အစ', 'ကြ', 'ရွ', 'ု', 'ေး', 'ွ', 'န်း', 'း', 'ပ', 'ဋ', 'ဆာ', 'အောင်', 'မြို့', 'စိတ်', 'ျ', 'ပြင်ဆင်', 'ါ', 'မဟုတ်', 'ပြု', 'ကိုယ်', 'ရှိ', 'ည', 'ဆောင်', 'ဆွေးနွေး', 'င်', 'n', 'တ်', 'ိုင်း', 'စီ', 'လူ', 'ဍ', 'ဟု', 'ည့်', 'သို့', '္', '႓', 'ိုး', 'န', 'ရေ', 'မယ်', 'ခဲ့သည်', 'ုံ', 'ောင်း', 'ောင်', 'ဦး', 'ထိ', 'တို့', 'ိမ့်', 'x', 'နိုင်ငံ', '၊', 'အပြ', 'ံ', 'ထု', 'ရေး', 'စစ်', 'ီ', 'မှု', 'ရှင်', 'ဦ', 'ရှိသည်', 'ပေါ', 'ဂျ', 'အစား', 'မြန်', 'ခ', 'သာ', 'နှ', 'ပထ', 'ိ', 'သင်', '့်']

    word_lists = []
    for i in word:
        cur = i.replace(' ', '')
        cur = cur.replace('</s>', '')
        cur = cur.replace('<s>', '')
        word_lists.append(cur)

    if len(word_lists) == 0:
        return False

    for ch in word_lists:
        if ch.isalpha() is False and ch in my_char:
            return True
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

    for num in range(words_size):
        if words[num] == ' ':
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
                    if words[num].encode('utf-8').isalpha():
                        abbr_word += words[num].upper()
                num += 1
            word_lists.append(abbr_word)
            if time_stamp is not None:
                end = time_stamp[ts_nums[num]][1]
                ts_lists.append([begin, end])
        else:
            word_lists.append(words[num])
            if time_stamp is not None and words[num] != ' ':
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
    word_item = ''
    ts_lists = []

    # wash words lists
    for i in words:
        word = ''
        if isinstance(i, str):
            word = i
        else:
            word = i.decode('utf-8')

        if word in ['<s>', '</s>', '<unk>', '<OOV>']:
            continue
        else:
            middle_lists.append(word)

    # all chinese characters
    if isAllChinese(middle_lists):
        for i, ch in enumerate(middle_lists):
            word_lists.append(ch.replace(' ', ''))
        if time_stamp is not None:
            ts_lists = time_stamp

    # all alpha characters
    elif isAllAlpha(middle_lists):
        ts_flag = True
        for i, ch in enumerate(middle_lists):
            if ts_flag and time_stamp is not None:
                begin = time_stamp[i][0]
                end = time_stamp[i][1]
            word = ''
            if '@@' in ch:
                word = ch.replace('@@', '')
                word_item += word
                if time_stamp is not None:
                    ts_flag = False
                    end = time_stamp[i][1]
            else:
                word_item += ch
                word_lists.append(word_item)
                word_lists.append(' ')
                word_item = ''
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
            word = ''
            if isAllChinese(ch):
                if alpha_blank is True:
                    word_lists.pop()
                word_lists.append(ch)
                alpha_blank = False
                if time_stamp is not None:
                    ts_flag = True
                    ts_lists.append([begin, end])
                    begin = end
            elif '@@' in ch:
                word = ch.replace('@@', '')
                word_item += word
                alpha_blank = False
                if time_stamp is not None:
                    ts_flag = False
                    end = time_stamp[i][1]
            elif isAllAlpha(ch):
                word_item += ch
                word_lists.append(word_item)
                word_lists.append(' ')
                word_item = ''
                alpha_blank = True
                if time_stamp is not None:
                    ts_flag = True
                    end = time_stamp[i][1] 
                    ts_lists.append([begin, end])
                    begin = end
             elif isMy(ch):
                word_item += ch
                word_lists.append(word_item)
                word_lists.append(' ')
                word_item = ''
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
            if ch != ' ':
                real_word_lists.append(ch)
        sentence = ' '.join(real_word_lists).strip()
        return sentence, ts_lists, real_word_lists
    else:
        word_lists = abbr_dispose(word_lists)
        real_word_lists = []
        for ch in word_lists:
            if ch != ' ':
                real_word_lists.append(ch)
        sentence = ''.join(word_lists).strip()
        return sentence, real_word_lists
