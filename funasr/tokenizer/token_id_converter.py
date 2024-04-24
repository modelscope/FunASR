from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Union

import numpy as np


class TokenIDConverter:
    def __init__(
        self,
        token_list: Union[Path, str, Iterable[str]],
        unk_symbol: str = "<unk>",
    ):

        if isinstance(token_list, (Path, str)):
            token_list = Path(token_list)
            self.token_list_repr = str(token_list)
            self.token_list: List[str] = []

            with token_list.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.rstrip()
                    self.token_list.append(line)

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

    def get_num_vocabulary_size(self) -> int:
        return len(self.token_list)

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        if isinstance(integers, np.ndarray) and integers.ndim != 1:
            raise ValueError(f"Must be 1 dim ndarray, but got {integers.ndim}")
        return [self.token_list[i] for i in integers]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        return [self.token2id.get(i, self.unk_id) for i in tokens]
