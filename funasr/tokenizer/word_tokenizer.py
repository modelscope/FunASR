from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union
import warnings


from funasr.tokenizer.abs_tokenizer import AbsTokenizer


class WordTokenizer(AbsTokenizer):
    def __init__(
        self,
        delimiter: str = None,
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        remove_non_linguistic_symbols: bool = False,
    ):
        self.delimiter = delimiter

        if not remove_non_linguistic_symbols and non_linguistic_symbols is not None:
            warnings.warn(
                "non_linguistic_symbols is only used " "when remove_non_linguistic_symbols = True"
            )

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

    def __repr__(self):
        return f'{self.__class__.__name__}(delimiter="{self.delimiter}")'

    def text2tokens(self, line: str) -> List[str]:
        tokens = []
        for t in line.split(self.delimiter):
            if self.remove_non_linguistic_symbols and t in self.non_linguistic_symbols:
                continue
            tokens.append(t)
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        if self.delimiter is None:
            delimiter = " "
        else:
            delimiter = self.delimiter
        return delimiter.join(tokens)
