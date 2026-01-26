# -*- coding: utf-8 -*-
#!/usr/bin/python
# Author: Mengze Chen

import re
import sys

import cn_tn as cn_tn
import format5res as cn_itn
import pyopenjtalk
import zhconv
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

basic_normalizer = BasicTextNormalizer()
english_normalizer = EnglishTextNormalizer()


def is_only_chinese_and_english(s):
    # 定义正则表达式模式，匹配中文字符范围和英文字母（包括大小写）
    pattern = r"^[\u4e00-\u9fa5A-Za-z0-9,\.!\?:;，。！？：；、%\'\s\-\~]+$"
    # 使用正则表达式进行匹配
    return re.match(pattern, s) is not None


def is_only_english(s):
    # 定义正则表达式模式，匹配中文字符范围和英文字母（包括大小写）
    pattern = r"^[A-Za-z0-9,\.!\?:;，。！？：；、%\'\s\-\~]+$"
    # 使用正则表达式进行匹配
    return re.match(pattern, s) is not None


def is_number(s):
    # 定义正则表达式模式，匹配中文字符范围和英文字母（包括大小写）
    pattern = r"^[0-9,\.!\?:;，。！？：；、%\'\s]+$"
    # 使用正则表达式进行匹配
    return re.match(pattern, s) is not None


def safe_ja_g2p(text, kana=True, max_length=100):
    if len(text) > max_length:
        # 如果文本过长，分段处理
        parts = []
        for i in range(0, len(text), max_length):
            part = text[i : i + max_length]
            try:
                converted = pyopenjtalk.g2p(part, kana=kana)
                parts.append(converted)
            except:
                parts.append(part)  # 如果转换失败，使用原文本
        return " ".join(parts)
    else:
        try:
            return pyopenjtalk.g2p(text, kana=kana)
        except:
            return text  # 如果转换失败，返回原文本


def normalize_text(srcfn, dstfn, kana=False):
    with open(srcfn, "r") as f_read, open(dstfn, "w") as f_write:
        all_lines = f_read.readlines()
        for line in all_lines:
            line = line.strip()
            line_arr = line.split(maxsplit=1)
            if len(line_arr) < 1:
                continue
            if len(line_arr) == 1:
                line_arr.append("")
            key = line_arr[0]
            line_arr[1] = re.sub(r"=", " ", line_arr[1])
            line_arr[1] = re.sub(r"\(", " ", line_arr[1])
            line_arr[1] = re.sub(r"\)", " ", line_arr[1])
            # From Chongjia Ni
            if kana:
                line_arr[1] = safe_ja_g2p(line_arr[1], kana=True, max_length=100)

            line_arr = f"{key}\t{line_arr[1]}".split()
            conts = []
            language_bak = ""
            part = []
            for i in range(1, len(line_arr)):
                out_part = ""
                chn_eng_bool = is_only_chinese_and_english(line_arr[i])
                eng_bool = is_only_english(line_arr[i])
                num_bool = is_number(line_arr[i])
                if eng_bool and not num_bool:
                    language = "en"
                elif chn_eng_bool:
                    language = "chn_en"
                else:
                    language = "not_chn_en"
                if language == language_bak or language_bak == "":
                    part.append(line_arr[i])
                    language_bak = language
                else:
                    if language_bak == "en":
                        out_part1 = english_normalizer(" ".join(part))
                        out_part = cn_itn.scoreformat("", out_part1)
                    elif language_bak == "chn_en":
                        out_part1 = english_normalizer(" ".join(part))
                        out_part2 = cn_tn.normalize_nsw(out_part1)
                        out_part3 = cn_itn.all_convert(out_part2)
                        out_part = zhconv.convert(out_part3, "zh-cn")
                    else:
                        out_part1 = basic_normalizer(" ".join(part))
                        out_part2 = cn_tn.normalize_nsw(out_part1)
                        out_part3 = cn_itn.all_convert(out_part2)
                        out_part = zhconv.convert(out_part3, "zh-cn")
                    conts.append(out_part)
                    language_bak = language
                    part = []
                    part.append(line_arr[i])
                if i == len(line_arr) - 1:
                    if language == "en":
                        out_part1 = english_normalizer(" ".join(part))
                        out_part = cn_itn.scoreformat("", out_part1)
                    elif language == "chn_en":
                        out_part1 = english_normalizer(" ".join(part))
                        out_part2 = cn_tn.normalize_nsw(out_part1)
                        out_part3 = cn_itn.all_convert(out_part2)
                        out_part = zhconv.convert(out_part3, "zh-cn")
                    else:
                        out_part1 = basic_normalizer(" ".join(part))
                        out_part2 = cn_tn.normalize_nsw(out_part1)
                        out_part3 = cn_itn.all_convert(out_part2)
                        out_part = zhconv.convert(out_part3, "zh-cn")
                    conts.append(out_part)

            f_write.write("{0}\t{1}\n".format(key, " ".join(conts).strip()))


if __name__ == "__main__":
    srcfn = sys.argv[1]
    dstfn = sys.argv[2]
    normalize_text(srcfn, dstfn, True if len(sys.argv) > 3 else False)
