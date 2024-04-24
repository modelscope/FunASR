from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union
import warnings
import re

from funasr.tokenizer.abs_tokenizer import BaseTokenizer
from funasr.register import tables


@tables.register("tokenizer_classes", "CharTokenizer")
class CharTokenizer(BaseTokenizer):
    def __init__(
        self,
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
        split_with_space: bool = False,
        seg_dict: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.space_symbol = space_symbol
        if non_linguistic_symbols is None:
            self.non_linguistic_symbols = set()
        elif isinstance(non_linguistic_symbols, (Path, str)):
            non_linguistic_symbols = Path(non_linguistic_symbols)
            try:
                with non_linguistic_symbols.open("r", encoding="utf-8") as f:
                    self.non_linguistic_symbols = set(line.rstrip() for line in f)
            except FileNotFoundError:
                warnings.warn(f"{non_linguistic_symbols} doesn't exist.")
                self.non_linguistic_symbols = set()
        else:
            self.non_linguistic_symbols = set(non_linguistic_symbols)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols
        self.split_with_space = split_with_space
        self.seg_dict = None
        seg_dict = seg_dict if seg_dict is not None else kwargs.get("seg_dict_file", None)
        if seg_dict is not None:
            self.seg_dict = load_seg_dict(seg_dict)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'space_symbol="{self.space_symbol}"'
            f'non_linguistic_symbols="{self.non_linguistic_symbols}"'
            f")"
        )

    def text2tokens(self, line: Union[str, list]) -> List[str]:

        # if self.split_with_space:

        if self.seg_dict is not None:
            tokens = line.strip().split(" ")
            tokens = seg_tokenize(tokens, self.seg_dict)
        else:
            tokens = []
            while len(line) != 0:
                for w in self.non_linguistic_symbols:
                    if line.startswith(w):
                        if not self.remove_non_linguistic_symbols:
                            tokens.append(line[: len(w)])
                        line = line[len(w) :]
                        break
                else:
                    t = line[0]
                    if t == " ":
                        # t = "<space>"
                        line = line[1:]
                        continue
                    tokens.append(t)
                    line = line[1:]
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        tokens = [t if t != self.space_symbol else " " for t in tokens]
        return "".join(tokens)


def load_seg_dict(seg_dict_file):
    seg_dict = {}
    assert isinstance(seg_dict_file, str)
    with open(seg_dict_file, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            s = line.strip().split()
            key = s[0]
            value = s[1:]
            seg_dict[key] = " ".join(value)
    return seg_dict


def seg_tokenize(txt, seg_dict):
    # pattern = re.compile(r'^[\u4E00-\u9FA50-9]+$')
    pattern = re.compile(r"([\u4E00-\u9FA5A-Za-z0-9])")
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
