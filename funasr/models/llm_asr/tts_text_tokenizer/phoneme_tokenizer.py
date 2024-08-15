import logging
from pathlib import Path
import re
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union
import warnings
import os
import json
import jamo


class TtsFrdRich:
    """
    rich text info: phoneme + puncs + boundary + [word2phone]
    """
    def __init__(self, remove_boundary=True, token_type="pronplus"):
        super().__init__()
        self.remove_boundary = remove_boundary
        self.token_type = token_type
        self.g2p = None
        self.lang_type = None
        self.lang_type_map = {"zh-cn": "pinyin", "en-us": "enus"}

    @staticmethod
    def contains_chinese(str):
        return bool(re.search(r'[\u4e00-\u9fff]', str))
    @staticmethod
    def is_full_half_punctuation_string(s):
        # 包含ASCII标点和常见全角标点
        punctuation_pattern = r'[\u0000-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007f\u3000-\u303f\uff00-\uffef]'
        # 使用re.findall找出所有匹配的字符
        results = re.findall(punctuation_pattern, s)
        # 如果字符串长度和匹配到的字符总数一样，说明全部是标点
        return len(s) == len("".join(results))

    def build(self, resource_dir, lang_type="Zh-CN"):
        lang_type = lang_type.lower()
        new_lang_type = self.lang_type_map[lang_type]
        if self.g2p is None:
            import ttsfrd
            assert os.path.isdir(resource_dir)
            fe = ttsfrd.TtsFrontendEngine()
            fe.initialize(resource_dir)
            self.g2p = fe
            # self.lang_type = new_lang_type
            self.set_lang_type(new_lang_type)
        if self.lang_type != new_lang_type:
            # self.lang_type = new_lang_type
            self.set_lang_type(new_lang_type)
    def set_lang_type(self, lang_type):
        if lang_type == "enus":
            self.g2p.set_lang_type(lang_type)
            self.g2p.enable_pinyin_mix(True)
            # self.g2p.set_breakmodel_index(0)
        else:
            self.g2p.set_lang_type(lang_type)
            self.g2p.enable_pinyin_mix(True)
            # self.g2p.set_breakmodel_index(1)
        self.lang_type = lang_type
    def set_token_type(self, token_type):
        assert token_type in ["pronplus", "word2phn", "wordlist"], token_type
        self.token_type = token_type
    def __call__(self, text) -> Union[List[str], str]:
        assert self.g2p is not None
        if not self.contains_chinese(text):
            if self.lang_type != "enus":
                self.set_lang_type("enus")
        else:
            if self.lang_type != "pinyin":
                self.set_lang_type("pinyin")
        if self.token_type == "word2phn":
            return self._get_word2phn(text)
        elif self.token_type == "pronplus":
            return self._get_pronplus(text)
        elif self.token_type == "wordlist":
            return self._get_wordlist(text)
        else:
            raise ValueError(f"only type: [pronplus, word2phn, wordlist] supported, now type: {self.token_type}")
    def _get_pronplus(self, text) -> List[str]:
        pronplus = self.g2p.get_frd_extra_info(text, 'pronplus')
        if self.remove_boundary:
            pronplus = pronplus.replace("/", "") # word boundary
            pronplus = pronplus.replace("#", "") # syllable boundary
            # pronplus = pronplus.replace("\n", "")
            pronplus = pronplus.replace("\n", " ")
        symbols: List[str] = []
        for pron in pronplus.split(" "):
            pron = pron.strip().lower()
            if pron and pron[0].isalpha():
                symbols.append(pron)
            else:
                symbols.extend([mark for mark in pron if mark])
        return symbols

    def text2tokens(self, line: str) -> List[str]:
        json_str = self._get_word2phn(line)
        data = json.loads(json_str)
        retval = []
        for one in data["word2phn"]:
            for key, value in one.items():
                if value is not None:
                    retval.extend([f"@{x}" for x in value])
                else:
                    if key == " ":
                        key = "<|space|>"
                    retval.append(f"@{key}")

        return retval

    def tokens2text(self, tokens: Iterable[str]) -> str:
        pass

    def _get_wordlist(self, text) -> str:
        wordlist = self.g2p.get_frd_extra_info(text, 'wordlist')
        return wordlist
    def _get_word2phn(self, text) -> str:
        wordlist = self.g2p.get_frd_extra_info(text, 'wordlist')
        wordlist_subs = wordlist.split("\n")
        word2phn_info = []
        prev_word_type = None
        prev_word = None
        for json_str in wordlist_subs:
            if len(json_str) == 0:
                continue
            wordlist_info = json.loads(json_str)["wordlist"]
            for word_info in wordlist_info:
                is_english_word = True
                this_phone_list = None
                if word_info["syllables"] is None:
                    # punctuation
                    this_word_type = "punc"
                    pass
                elif self.is_full_half_punctuation_string(word_info["name"]):
                    # punctuation, handle some g2p's mistakes spelling punctuation!!!
                    this_word_type = "punc"
                    pass
                else:
                    this_phone_list = []
                    for syllable_info in word_info["syllables"]:
                        phn_count = syllable_info["phone_count"]
                        syllable_phone_list = syllable_info["pron_text"].split(" ")
                        assert len(syllable_phone_list) == phn_count, len(syllable_phone_list)
                        if "py_text" in syllable_info:
                            # chinese add tone info
                            syllable_phone_list[-1] = syllable_phone_list[-1]+str(syllable_info["tone"])
                            is_english_word = False
                        this_phone_list += syllable_phone_list
                    if is_english_word:
                        this_word_type = "en_word"
                    else:
                        this_word_type = "ch_word"
                if this_word_type == "en_word":
                    if prev_word_type is None:
                        pass
                    elif prev_word_type == "en_word":
                        word2phn_info.append({" ": None})
                    elif prev_word_type == "punc":
                        if (prev_word not in ["\"", "\'", "(", "（", "[", "【"] and
                                prev_word.split(" ")[-1] not in ["\"", "\'", "(", "（", "[", "【"]):
                            word2phn_info.append({" ": None})
                    elif prev_word_type == "ch_word":
                        word2phn_info.append({" ": None})
                elif this_word_type == "ch_word":
                    if prev_word_type is not None and prev_word_type == "en_word":
                        word2phn_info.append({" ": None})
                elif this_word_type == "punc":
                    if word_info["name"] in ["("]:
                        word2phn_info.append({" ": None})
                this_word2phn_dict = {word_info["name"]: this_phone_list}
                word2phn_info.append(this_word2phn_dict)
                prev_word_type = this_word_type
                prev_word = list(word2phn_info[-1].keys())[0]
        return json.dumps({"raw": text, "word2phn": word2phn_info}, ensure_ascii=False)
