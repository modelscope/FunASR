import logging
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union

import sentencepiece as spm
from torch.utils.data import DataLoader
from typeguard import check_argument_types

from funasr.datasets.large_datasets.dataset import Dataset
from funasr.iterators.abs_iter_factory import AbsIterFactory
from funasr.text.abs_tokenizer import AbsTokenizer


def read_symbol_table(symbol_table_file):
    if isinstance(symbol_table_file, str):
        symbol_table = {}
        with open(symbol_table_file, "r", encoding="utf8") as fin:
            for i, line in enumerate(fin):
                char = line.strip()
                symbol_table[char] = i
    else:
        assert isinstance(symbol_table_file, list)
        symbol_table = {}
        for i, char in enumerate(symbol_table_file):
            symbol_table[char] = i
    return symbol_table


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


class SentencepiecesTokenizer(AbsTokenizer):
    def __init__(self, model: Union[Path, str]):
        assert check_argument_types()
        self.model = str(model)
        self.sp = None

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def _build_sentence_piece_processor(self):
        if self.sp is None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.model)

    def text2tokens(self, line: str) -> List[str]:
        self._build_sentence_piece_processor()
        return self.sp.EncodeAsPieces(line)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        self._build_sentence_piece_processor()
        return self.sp.DecodePieces(list(tokens))


class LargeDataLoader(AbsIterFactory):
    def __init__(self, args, mode="train"):
        symbol_table = read_symbol_table(args.token_list) if args.token_list is not None else None
        seg_dict = load_seg_dict(args.seg_dict_file) if args.seg_dict_file is not None else None
        punc_dict = load_seg_dict(args.punc_dict_file) if args.punc_dict_file is not None else None
        bpe_tokenizer = load_seg_dict(args.bpemodel_file) if args.bpemodel_file is not None else None
        self.dataset_conf = args.dataset_conf
        self.frontend_conf = args.frontend_conf
        logging.info("dataloader config: {}".format(self.dataset_conf))
        batch_mode = self.dataset_conf.get("batch_mode", "padding")
        self.dataset = Dataset(args.data_list, symbol_table, seg_dict, punc_dict, bpe_tokenizer,
                               self.dataset_conf, self.frontend_conf, speed_perturb=args.speed_perturb,
                               mode=mode, batch_mode=batch_mode)

    def build_iter(self, epoch, shuffle=True):
        self.dataset.set_epoch(epoch)
        data_loader = DataLoader(self.dataset,
                                 batch_size=None,
                                 pin_memory=True,
                                 num_workers=self.dataset_conf.get("num_workers", 8))
        return data_loader
