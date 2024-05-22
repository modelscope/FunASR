import json
import numpy as np
from abc import ABC
from pathlib import Path
from abc import abstractmethod
from typing import Union, Iterable, List, Dict


class AbsTokenizer(ABC):
    @abstractmethod
    def text2tokens(self, line: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def tokens2text(self, tokens: Iterable[str]) -> str:
        raise NotImplementedError


class BaseTokenizer(ABC):
    def __init__(
        self,
        token_list: Union[Path, str, Iterable[str]] = None,
        unk_symbol: str = "<unk>",
        **kwargs,
    ):

        if token_list is not None:
            if isinstance(token_list, (Path, str)) and token_list.endswith(".txt"):
                token_list = Path(token_list)
                self.token_list_repr = str(token_list)
                self.token_list: List[str] = []

                with token_list.open("r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        line = line.rstrip()
                        self.token_list.append(line)
            elif isinstance(token_list, (Path, str)) and token_list.endswith(".json"):
                token_list = Path(token_list)
                self.token_list_repr = str(token_list)
                self.token_list: List[str] = []

                with open(token_list, "r", encoding="utf-8") as f:
                    self.token_list = json.load(f)

            else:
                self.token_list: List[str] = list(token_list)
                self.token_list_repr = ""
                for i, t in enumerate(self.token_list):
                    if i == 3:
                        break
                    self.token_list_repr += f"{t}, "
                self.token_list_repr += f"... (NVocab={(len(self.token_list))})"

            self.token2id: Dict[str, int] = {}
            for i, t in enumerate(self.token_list):
                if t in self.token2id:
                    raise RuntimeError(f'Symbol "{t}" is duplicated')
                self.token2id[t] = i

            self.unk_symbol = unk_symbol
            if self.unk_symbol not in self.token2id:
                raise RuntimeError(f"Unknown symbol '{unk_symbol}' doesn't exist in the token_list")
            self.unk_id = self.token2id[self.unk_symbol]

    def encode(self, text, **kwargs):
        tokens = self.text2tokens(text)
        text_ints = self.tokens2ids(tokens)

        return text_ints

    def decode(self, text_ints):
        token = self.ids2tokens(text_ints)
        text = self.tokens2text(token)
        return text

    def get_num_vocabulary_size(self) -> int:
        return len(self.token_list)

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        if isinstance(integers, np.ndarray) and integers.ndim != 1:
            raise ValueError(f"Must be 1 dim ndarray, but got {integers.ndim}")
        return [self.token_list[i] for i in integers]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        return [self.token2id.get(i, self.unk_id) for i in tokens]

    @abstractmethod
    def text2tokens(self, line: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def tokens2text(self, tokens: Iterable[str]) -> str:
        raise NotImplementedError
