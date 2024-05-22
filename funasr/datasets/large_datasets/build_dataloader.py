import logging
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union

import sentencepiece as spm
from torch.utils.data import DataLoader

from funasr.datasets.large_datasets.dataset import Dataset
from funasr.datasets.large_datasets.abs_iter_factory import AbsIterFactory
from funasr.tokenizer.abs_tokenizer import AbsTokenizer

from funasr.register import tables


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


@tables.register("dataset_classes", "LargeDataset")
class LargeDataLoader(AbsIterFactory):
    def __init__(self, args, mode="train"):
        symbol_table, seg_dict, punc_dict, bpe_tokenizer = None, None, None, None
        if hasattr(args, "token_list") and args.token_list is not None:
            symbol_table = read_symbol_table(args.token_list)
        if hasattr(args, "seg_dict_file") and args.seg_dict_file is not None:
            seg_dict = load_seg_dict(args.seg_dict_file)
        if hasattr(args, "punc_list") and args.punc_list is not None:
            punc_dict = read_symbol_table(args.punc_list)
        if hasattr(args, "bpemodel") and args.bpemodel is not None:
            bpe_tokenizer = SentencepiecesTokenizer(args.bpemodel)
        self.dataset_conf = args.dataset_conf
        if "frontend_conf" not in args:
            self.frontend_conf = None
        else:
            self.frontend_conf = args.frontend_conf
        self.speed_perturb = args.speed_perturb if hasattr(args, "speed_perturb") else None
        logging.info("dataloader config: {}".format(self.dataset_conf))
        batch_mode = self.dataset_conf.get("batch_mode", "padding")
        data_list = args.train_data_file if mode == "train" else args.valid_data_file
        self.dataset = Dataset(
            data_list,
            symbol_table,
            seg_dict,
            punc_dict,
            bpe_tokenizer,
            self.dataset_conf,
            self.frontend_conf,
            speed_perturb=self.speed_perturb if mode == "train" else None,
            mode=mode,
            batch_mode=batch_mode,
        )

    def build_iter(self, epoch, shuffle=True):
        self.dataset.set_epoch(epoch)
        data_loader = DataLoader(
            self.dataset,
            batch_size=None,
            pin_memory=True,
            num_workers=self.dataset_conf.get("num_workers", 8),
        )
        return data_loader
