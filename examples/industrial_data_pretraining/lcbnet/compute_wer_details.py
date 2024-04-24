#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from enum import Enum
import re, sys, unicodedata
import codecs
import argparse
from tqdm import tqdm
import os
import pdb

remove_tag = False
spacelist = [" ", "\t", "\r", "\n"]
puncts = [
    "!",
    ",",
    "?",
    "、",
    "。",
    "！",
    "，",
    "；",
    "？",
    "：",
    "「",
    "」",
    "︰",
    "『",
    "』",
    "《",
    "》",
]


class Code(Enum):
    match = 1
    substitution = 2
    insertion = 3
    deletion = 4


class WordError(object):
    def __init__(self):
        self.errors = {
            Code.substitution: 0,
            Code.insertion: 0,
            Code.deletion: 0,
        }
        self.ref_words = 0

    def get_wer(self):
        assert self.ref_words != 0
        errors = (
            self.errors[Code.substitution]
            + self.errors[Code.insertion]
            + self.errors[Code.deletion]
        )
        return 100.0 * errors / self.ref_words

    def get_result_string(self):
        return (
            f"error_rate={self.get_wer():.4f}, "
            f"ref_words={self.ref_words}, "
            f"subs={self.errors[Code.substitution]}, "
            f"ins={self.errors[Code.insertion]}, "
            f"dels={self.errors[Code.deletion]}"
        )


def characterize(string):
    res = []
    i = 0
    while i < len(string):
        char = string[i]
        if char in puncts:
            i += 1
            continue
        cat1 = unicodedata.category(char)
        # https://unicodebook.readthedocs.io/unicode.html#unicode-categories
        if cat1 == "Zs" or cat1 == "Cn" or char in spacelist:  # space or not assigned
            i += 1
            continue
        if cat1 == "Lo":  # letter-other
            res.append(char)
            i += 1
        else:
            # some input looks like: <unk><noise>, we want to separate it to two words.
            sep = " "
            if char == "<":
                sep = ">"
            j = i + 1
            while j < len(string):
                c = string[j]
                if ord(c) >= 128 or (c in spacelist) or (c == sep):
                    break
                j += 1
            if j < len(string) and string[j] == ">":
                j += 1
            res.append(string[i:j])
            i = j
    return res


def stripoff_tags(x):
    if not x:
        return ""
    chars = []
    i = 0
    T = len(x)
    while i < T:
        if x[i] == "<":
            while i < T and x[i] != ">":
                i += 1
            i += 1
        else:
            chars.append(x[i])
            i += 1
    return "".join(chars)


def normalize(sentence, ignore_words, cs, split=None):
    """sentence, ignore_words are both in unicode"""
    new_sentence = []
    for token in sentence:
        x = token
        if not cs:
            x = x.upper()
        if x in ignore_words:
            continue
        if remove_tag:
            x = stripoff_tags(x)
        if not x:
            continue
        if split and x in split:
            new_sentence += split[x]
        else:
            new_sentence.append(x)
    return new_sentence


class Calculator:
    def __init__(self):
        self.data = {}
        self.space = []
        self.cost = {}
        self.cost["cor"] = 0
        self.cost["sub"] = 1
        self.cost["del"] = 1
        self.cost["ins"] = 1

    def calculate(self, lab, rec):
        # Initialization
        lab.insert(0, "")
        rec.insert(0, "")
        while len(self.space) < len(lab):
            self.space.append([])
        for row in self.space:
            for element in row:
                element["dist"] = 0
                element["error"] = "non"
            while len(row) < len(rec):
                row.append({"dist": 0, "error": "non"})
        for i in range(len(lab)):
            self.space[i][0]["dist"] = i
            self.space[i][0]["error"] = "del"
        for j in range(len(rec)):
            self.space[0][j]["dist"] = j
            self.space[0][j]["error"] = "ins"
        self.space[0][0]["error"] = "non"
        for token in lab:
            if token not in self.data and len(token) > 0:
                self.data[token] = {"all": 0, "cor": 0, "sub": 0, "ins": 0, "del": 0}
        for token in rec:
            if token not in self.data and len(token) > 0:
                self.data[token] = {"all": 0, "cor": 0, "sub": 0, "ins": 0, "del": 0}
        # Computing edit distance
        for i, lab_token in enumerate(lab):
            for j, rec_token in enumerate(rec):
                if i == 0 or j == 0:
                    continue
                min_dist = sys.maxsize
                min_error = "none"
                dist = self.space[i - 1][j]["dist"] + self.cost["del"]
                error = "del"
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                dist = self.space[i][j - 1]["dist"] + self.cost["ins"]
                error = "ins"
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                if lab_token == rec_token.replace("<BIAS>", ""):
                    dist = self.space[i - 1][j - 1]["dist"] + self.cost["cor"]
                    error = "cor"
                else:
                    dist = self.space[i - 1][j - 1]["dist"] + self.cost["sub"]
                    error = "sub"
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                self.space[i][j]["dist"] = min_dist
                self.space[i][j]["error"] = min_error
        # Tracing back
        result = {
            "lab": [],
            "rec": [],
            "code": [],
            "all": 0,
            "cor": 0,
            "sub": 0,
            "ins": 0,
            "del": 0,
        }
        i = len(lab) - 1
        j = len(rec) - 1
        while True:
            if self.space[i][j]["error"] == "cor":  # correct
                if len(lab[i]) > 0:
                    self.data[lab[i]]["all"] = self.data[lab[i]]["all"] + 1
                    self.data[lab[i]]["cor"] = self.data[lab[i]]["cor"] + 1
                    result["all"] = result["all"] + 1
                    result["cor"] = result["cor"] + 1
                result["lab"].insert(0, lab[i])
                result["rec"].insert(0, rec[j])
                result["code"].insert(0, Code.match)
                i = i - 1
                j = j - 1
            elif self.space[i][j]["error"] == "sub":  # substitution
                if len(lab[i]) > 0:
                    self.data[lab[i]]["all"] = self.data[lab[i]]["all"] + 1
                    self.data[lab[i]]["sub"] = self.data[lab[i]]["sub"] + 1
                    result["all"] = result["all"] + 1
                    result["sub"] = result["sub"] + 1
                result["lab"].insert(0, lab[i])
                result["rec"].insert(0, rec[j])
                result["code"].insert(0, Code.substitution)
                i = i - 1
                j = j - 1
            elif self.space[i][j]["error"] == "del":  # deletion
                if len(lab[i]) > 0:
                    self.data[lab[i]]["all"] = self.data[lab[i]]["all"] + 1
                    self.data[lab[i]]["del"] = self.data[lab[i]]["del"] + 1
                    result["all"] = result["all"] + 1
                    result["del"] = result["del"] + 1
                result["lab"].insert(0, lab[i])
                result["rec"].insert(0, "")
                result["code"].insert(0, Code.deletion)
                i = i - 1
            elif self.space[i][j]["error"] == "ins":  # insertion
                if len(rec[j]) > 0:
                    self.data[rec[j]]["ins"] = self.data[rec[j]]["ins"] + 1
                    result["ins"] = result["ins"] + 1
                result["lab"].insert(0, "")
                result["rec"].insert(0, rec[j])
                result["code"].insert(0, Code.insertion)
                j = j - 1
            elif self.space[i][j]["error"] == "non":  # starting point
                break
            else:  # shouldn't reach here
                print(
                    "this should not happen , i = {i} , j = {j} , error = {error}".format(
                        i=i, j=j, error=self.space[i][j]["error"]
                    )
                )
        return result

    def overall(self):
        result = {"all": 0, "cor": 0, "sub": 0, "ins": 0, "del": 0}
        for token in self.data:
            result["all"] = result["all"] + self.data[token]["all"]
            result["cor"] = result["cor"] + self.data[token]["cor"]
            result["sub"] = result["sub"] + self.data[token]["sub"]
            result["ins"] = result["ins"] + self.data[token]["ins"]
            result["del"] = result["del"] + self.data[token]["del"]
        return result

    def cluster(self, data):
        result = {"all": 0, "cor": 0, "sub": 0, "ins": 0, "del": 0}
        for token in data:
            if token in self.data:
                result["all"] = result["all"] + self.data[token]["all"]
                result["cor"] = result["cor"] + self.data[token]["cor"]
                result["sub"] = result["sub"] + self.data[token]["sub"]
                result["ins"] = result["ins"] + self.data[token]["ins"]
                result["del"] = result["del"] + self.data[token]["del"]
        return result

    def keys(self):
        return list(self.data.keys())


def width(string):
    return sum(1 + (unicodedata.east_asian_width(c) in "AFW") for c in string)


def default_cluster(word):
    unicode_names = [unicodedata.name(char) for char in word]
    for i in reversed(range(len(unicode_names))):
        if unicode_names[i].startswith("DIGIT"):  # 1
            unicode_names[i] = "Number"  # 'DIGIT'
        elif unicode_names[i].startswith("CJK UNIFIED IDEOGRAPH") or unicode_names[i].startswith(
            "CJK COMPATIBILITY IDEOGRAPH"
        ):
            # 明 / 郎
            unicode_names[i] = "Mandarin"  # 'CJK IDEOGRAPH'
        elif unicode_names[i].startswith("LATIN CAPITAL LETTER") or unicode_names[i].startswith(
            "LATIN SMALL LETTER"
        ):
            # A / a
            unicode_names[i] = "English"  # 'LATIN LETTER'
        elif unicode_names[i].startswith("HIRAGANA LETTER"):  # は こ め
            unicode_names[i] = "Japanese"  # 'GANA LETTER'
        elif (
            unicode_names[i].startswith("AMPERSAND")
            or unicode_names[i].startswith("APOSTROPHE")
            or unicode_names[i].startswith("COMMERCIAL AT")
            or unicode_names[i].startswith("DEGREE CELSIUS")
            or unicode_names[i].startswith("EQUALS SIGN")
            or unicode_names[i].startswith("FULL STOP")
            or unicode_names[i].startswith("HYPHEN-MINUS")
            or unicode_names[i].startswith("LOW LINE")
            or unicode_names[i].startswith("NUMBER SIGN")
            or unicode_names[i].startswith("PLUS SIGN")
            or unicode_names[i].startswith("SEMICOLON")
        ):
            # & / ' / @ / ℃ / = / . / - / _ / # / + / ;
            del unicode_names[i]
        else:
            return "Other"
    if len(unicode_names) == 0:
        return "Other"
    if len(unicode_names) == 1:
        return unicode_names[0]
    for i in range(len(unicode_names) - 1):
        if unicode_names[i] != unicode_names[i + 1]:
            return "Other"
    return unicode_names[0]


def get_args():
    parser = argparse.ArgumentParser(description="wer cal")
    parser.add_argument("--ref", type=str, help="Text input path")
    parser.add_argument("--ref_ocr", type=str, help="Text input path")
    parser.add_argument("--rec_name", type=str, action="append", default=[])
    parser.add_argument("--rec_file", type=str, action="append", default=[])
    parser.add_argument("--verbose", type=int, default=1, help="show")
    parser.add_argument("--char", type=bool, default=True, help="show")
    args = parser.parse_args()
    return args


def main(args):
    cluster_file = ""
    ignore_words = set()
    tochar = args.char
    verbose = args.verbose
    padding_symbol = " "
    case_sensitive = False
    max_words_per_line = sys.maxsize
    split = None

    if not case_sensitive:
        ig = set([w.upper() for w in ignore_words])
        ignore_words = ig

    default_clusters = {}
    default_words = {}
    ref_file = args.ref
    ref_ocr = args.ref_ocr
    rec_files = args.rec_file
    rec_names = args.rec_name
    assert len(rec_files) == len(rec_names)

    # load ocr
    ref_ocr_dict = {}
    with codecs.open(ref_ocr, "r", "utf-8") as fh:
        for line in fh:
            if "$" in line:
                line = line.replace("$", " ")
            if tochar:
                array = characterize(line)
            else:
                array = line.strip().split()
            if len(array) == 0:
                continue
            fid = array[0]
            ref_ocr_dict[fid] = normalize(array[1:], ignore_words, case_sensitive, split)

    if split and not case_sensitive:
        newsplit = dict()
        for w in split:
            words = split[w]
            for i in range(len(words)):
                words[i] = words[i].upper()
            newsplit[w.upper()] = words
        split = newsplit

    rec_sets = {}
    calculators_dict = dict()
    ub_wer_dict = dict()
    hotwords_related_dict = dict()  # 记录recall相关的内容
    for i, hyp_file in enumerate(rec_files):
        rec_sets[rec_names[i]] = dict()
        with codecs.open(hyp_file, "r", "utf-8") as fh:
            for line in fh:
                if tochar:
                    array = characterize(line)
                else:
                    array = line.strip().split()
                if len(array) == 0:
                    continue
                fid = array[0]
                rec_sets[rec_names[i]][fid] = normalize(
                    array[1:], ignore_words, case_sensitive, split
                )

        calculators_dict[rec_names[i]] = Calculator()
        ub_wer_dict[rec_names[i]] = {"u_wer": WordError(), "b_wer": WordError(), "wer": WordError()}
        hotwords_related_dict[rec_names[i]] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        # tp: 热词在label里，同时在rec里
        # tn: 热词不在label里，同时不在rec里
        # fp: 热词不在label里，但是在rec里
        # fn: 热词在label里，但是不在rec里

    # record wrong label but in ocr
    wrong_rec_but_in_ocr_dict = {}
    for rec_name in rec_names:
        wrong_rec_but_in_ocr_dict[rec_name] = 0

    _file_total_len = 0
    with os.popen("cat {} | wc -l".format(ref_file)) as pipe:
        _file_total_len = int(pipe.read().strip())

    # compute error rate on the interaction of reference file and hyp file
    for line in tqdm(open(ref_file, "r", encoding="utf-8"), total=_file_total_len):
        if tochar:
            array = characterize(line)
        else:
            array = line.rstrip("\n").split()
        if len(array) == 0:
            continue
        fid = array[0]
        lab = normalize(array[1:], ignore_words, case_sensitive, split)

        if verbose:
            print("\nutt: %s" % fid)

        ocr_text = ref_ocr_dict[fid]
        ocr_set = set(ocr_text)
        print("ocr: {}".format(" ".join(ocr_text)))
        list_match = []  # 指label里面在ocr里面的内容
        list_not_mathch = []
        tmp_error = 0
        tmp_match = 0
        for index in range(len(lab)):
            # text_list.append(uttlist[index+1])
            if lab[index] not in ocr_set:
                tmp_error += 1
                list_not_mathch.append(lab[index])
            else:
                tmp_match += 1
                list_match.append(lab[index])
        print("label in ocr: {}".format(" ".join(list_match)))

        # for each reco file
        base_wrong_ocr_wer = None
        ocr_wrong_ocr_wer = None

        for rec_name in rec_names:
            rec_set = rec_sets[rec_name]
            if fid not in rec_set:
                continue
            rec = rec_set[fid]

            # print(rec)
            for word in rec + lab:
                if word not in default_words:
                    default_cluster_name = default_cluster(word)
                    if default_cluster_name not in default_clusters:
                        default_clusters[default_cluster_name] = {}
                    if word not in default_clusters[default_cluster_name]:
                        default_clusters[default_cluster_name][word] = 1
                    default_words[word] = default_cluster_name

            result = calculators_dict[rec_name].calculate(lab.copy(), rec.copy())
            if verbose:
                if result["all"] != 0:
                    wer = (
                        float(result["ins"] + result["sub"] + result["del"]) * 100.0 / result["all"]
                    )
                else:
                    wer = 0.0
            print("WER(%s): %4.2f %%" % (rec_name, wer), end=" ")
            print(
                "N=%d C=%d S=%d D=%d I=%d"
                % (result["all"], result["cor"], result["sub"], result["del"], result["ins"])
            )

            # print(result['rec'])
            wrong_rec_but_in_ocr = []
            for idx in range(len(result["lab"])):
                if result["lab"][idx] != "":
                    if result["lab"][idx] != result["rec"][idx].replace("<BIAS>", ""):
                        if result["lab"][idx] in list_match:
                            wrong_rec_but_in_ocr.append(result["lab"][idx])
                            wrong_rec_but_in_ocr_dict[rec_name] += 1
            print("wrong_rec_but_in_ocr: {}".format(" ".join(wrong_rec_but_in_ocr)))

            if rec_name == "base":
                base_wrong_ocr_wer = len(wrong_rec_but_in_ocr)
            if "ocr" in rec_name or "hot" in rec_name:
                ocr_wrong_ocr_wer = len(wrong_rec_but_in_ocr)
                if ocr_wrong_ocr_wer < base_wrong_ocr_wer:
                    print(
                        "{} {} helps, {} -> {}".format(
                            fid, rec_name, base_wrong_ocr_wer, ocr_wrong_ocr_wer
                        )
                    )
                elif ocr_wrong_ocr_wer > base_wrong_ocr_wer:
                    print(
                        "{} {} hurts, {} -> {}".format(
                            fid, rec_name, base_wrong_ocr_wer, ocr_wrong_ocr_wer
                        )
                    )

            # recall = 0
            # false_alarm = 0
            # for idx in range(len(result['lab'])):
            #     if "<BIAS>" in result['rec'][idx]:
            #         if result['rec'][idx].replace("<BIAS>", "") in list_match:
            #             recall += 1
            #         else:
            #             false_alarm += 1
            # print("bias hotwords recall: {}, fa: {}, list_match {}, recall: {:.2f}, fa: {:.2f}".format(
            #     recall, false_alarm, len(list_match), recall / len(list_match) if len(list_match) != 0 else 0, false_alarm / len(list_match) if len(list_match) != 0 else 0
            # ))
            # tp: 热词在label里，同时在rec里
            # tn: 热词不在label里，同时不在rec里
            # fp: 热词不在label里，但是在rec里
            # fn: 热词在label里，但是不在rec里
            _rec_list = [word.replace("<BIAS>", "") for word in rec]
            _label_list = [word for word in lab]
            _tp = _tn = _fp = _fn = 0
            hot_true_list = [hotword for hotword in ocr_text if hotword in _label_list]
            hot_bad_list = [hotword for hotword in ocr_text if hotword not in _label_list]
            for badhotword in hot_bad_list:
                count = len([word for word in _rec_list if word == badhotword])
                # print(f"bad {badhotword} count: {count}")
                # for word in _rec_list:
                #     if badhotword == word:
                #         count += 1
                if count == 0:
                    hotwords_related_dict[rec_name]["tn"] += 1
                    _tn += 1
                    # fp: 0
                else:
                    hotwords_related_dict[rec_name]["fp"] += count
                    _fp += count
                    # tn: 0
                # if badhotword in _rec_list:
                #     hotwords_related_dict[rec_name]['fp'] += 1
                # else:
                #     hotwords_related_dict[rec_name]['tn'] += 1
            for hotword in hot_true_list:
                true_count = len([word for word in _label_list if hotword == word])
                rec_count = len([word for word in _rec_list if hotword == word])
                # print(f"good {hotword} true_count: {true_count}, rec_count: {rec_count}")
                if rec_count == true_count:
                    hotwords_related_dict[rec_name]["tp"] += true_count
                    _tp += true_count
                elif rec_count > true_count:
                    hotwords_related_dict[rec_name]["tp"] += true_count
                    # fp: 不在label里，但是在rec里
                    hotwords_related_dict[rec_name]["fp"] += rec_count - true_count
                    _tp += true_count
                    _fp += rec_count - true_count
                else:
                    hotwords_related_dict[rec_name]["tp"] += rec_count
                    # fn: 热词在label里，但是不在rec里
                    hotwords_related_dict[rec_name]["fn"] += true_count - rec_count
                    _tp += rec_count
                    _fn += true_count - rec_count
            print(
                "hotword: tp: {}, tn: {}, fp: {}, fn: {}, all: {}, recall: {:.2f}%".format(
                    _tp,
                    _tn,
                    _fp,
                    _fn,
                    sum([_tp, _tn, _fp, _fn]),
                    _tp / (_tp + _fn) * 100 if (_tp + _fn) != 0 else 0,
                )
            )

            # if hotword in _rec_list:
            #     hotwords_related_dict[rec_name]['tp'] += 1
            # else:
            #     hotwords_related_dict[rec_name]['fn'] += 1
            # 计算uwer, bwer, wer
            for code, rec_word, lab_word in zip(result["code"], result["rec"], result["lab"]):
                if code == Code.match:
                    ub_wer_dict[rec_name]["wer"].ref_words += 1
                    if lab_word in hot_true_list:
                        # tmp_ref.append(ref_tokens[ref_idx])
                        ub_wer_dict[rec_name]["b_wer"].ref_words += 1
                    else:
                        ub_wer_dict[rec_name]["u_wer"].ref_words += 1
                elif code == Code.substitution:
                    ub_wer_dict[rec_name]["wer"].ref_words += 1
                    ub_wer_dict[rec_name]["wer"].errors[Code.substitution] += 1
                    if lab_word in hot_true_list:
                        # tmp_ref.append(ref_tokens[ref_idx])
                        ub_wer_dict[rec_name]["b_wer"].ref_words += 1
                        ub_wer_dict[rec_name]["b_wer"].errors[Code.substitution] += 1
                    else:
                        ub_wer_dict[rec_name]["u_wer"].ref_words += 1
                        ub_wer_dict[rec_name]["u_wer"].errors[Code.substitution] += 1
                elif code == Code.deletion:
                    ub_wer_dict[rec_name]["wer"].ref_words += 1
                    ub_wer_dict[rec_name]["wer"].errors[Code.deletion] += 1
                    if lab_word in hot_true_list:
                        # tmp_ref.append(ref_tokens[ref_idx])
                        ub_wer_dict[rec_name]["b_wer"].ref_words += 1
                        ub_wer_dict[rec_name]["b_wer"].errors[Code.deletion] += 1
                    else:
                        ub_wer_dict[rec_name]["u_wer"].ref_words += 1
                        ub_wer_dict[rec_name]["u_wer"].errors[Code.deletion] += 1
                elif code == Code.insertion:
                    ub_wer_dict[rec_name]["wer"].errors[Code.insertion] += 1
                    if rec_word in hot_true_list:
                        ub_wer_dict[rec_name]["b_wer"].errors[Code.insertion] += 1
                    else:
                        ub_wer_dict[rec_name]["u_wer"].errors[Code.insertion] += 1

            space = {}
            space["lab"] = []
            space["rec"] = []
            for idx in range(len(result["lab"])):
                len_lab = width(result["lab"][idx])
                len_rec = width(result["rec"][idx])
                length = max(len_lab, len_rec)
                space["lab"].append(length - len_lab)
                space["rec"].append(length - len_rec)
            upper_lab = len(result["lab"])
            upper_rec = len(result["rec"])
            lab1, rec1 = 0, 0
            while lab1 < upper_lab or rec1 < upper_rec:
                if verbose > 1:
                    print("lab(%s):" % fid.encode("utf-8"), end=" ")
                else:
                    print("lab:", end=" ")
                lab2 = min(upper_lab, lab1 + max_words_per_line)
                for idx in range(lab1, lab2):
                    token = result["lab"][idx]
                    print("{token}".format(token=token), end="")
                    for n in range(space["lab"][idx]):
                        print(padding_symbol, end="")
                    print(" ", end="")
                print()
                if verbose > 1:
                    print("rec(%s):" % fid.encode("utf-8"), end=" ")
                else:
                    print("rec:", end=" ")

                rec2 = min(upper_rec, rec1 + max_words_per_line)
                for idx in range(rec1, rec2):
                    token = result["rec"][idx]
                    print("{token}".format(token=token), end="")
                    for n in range(space["rec"][idx]):
                        print(padding_symbol, end="")
                    print(" ", end="")
                print()
                # print('\n', end='\n')
                lab1 = lab2
                rec1 = rec2
        print("\n", end="\n")
        # break
    if verbose:
        print("===========================================================================")
        print()

    print(wrong_rec_but_in_ocr_dict)
    for rec_name in rec_names:
        result = calculators_dict[rec_name].overall()

        if result["all"] != 0:
            wer = float(result["ins"] + result["sub"] + result["del"]) * 100.0 / result["all"]
        else:
            wer = 0.0
        print("{} Overall -> {:4.2f} %".format(rec_name, wer), end=" ")
        print(
            "N=%d C=%d S=%d D=%d I=%d"
            % (result["all"], result["cor"], result["sub"], result["del"], result["ins"])
        )
        print(f"WER: {ub_wer_dict[rec_name]['wer'].get_result_string()}")
        print(f"U-WER: {ub_wer_dict[rec_name]['u_wer'].get_result_string()}")
        print(f"B-WER: {ub_wer_dict[rec_name]['b_wer'].get_result_string()}")

        print(
            "hotword: tp: {}, tn: {}, fp: {}, fn: {}, all: {}, recall: {:.2f}%".format(
                hotwords_related_dict[rec_name]["tp"],
                hotwords_related_dict[rec_name]["tn"],
                hotwords_related_dict[rec_name]["fp"],
                hotwords_related_dict[rec_name]["fn"],
                sum([v for k, v in hotwords_related_dict[rec_name].items()]),
                (
                    hotwords_related_dict[rec_name]["tp"]
                    / (
                        hotwords_related_dict[rec_name]["tp"]
                        + hotwords_related_dict[rec_name]["fn"]
                    )
                    * 100
                    if hotwords_related_dict[rec_name]["tp"] + hotwords_related_dict[rec_name]["fn"]
                    != 0
                    else 0
                ),
            )
        )

        # tp: 热词在label里，同时在rec里
        # tn: 热词不在label里，同时不在rec里
        # fp: 热词不在label里，但是在rec里
        # fn: 热词在label里，但是不在rec里
        if not verbose:
            print()
        print()


if __name__ == "__main__":
    args = get_args()

    # print("")
    print(args)
    main(args)
