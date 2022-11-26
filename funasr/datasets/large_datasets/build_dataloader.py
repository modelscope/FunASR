import logging

import yaml

from torch.utils.data import DataLoader
from funasr.datasets.large_datasets.dataset import Dataset
from funasr.iterators.abs_iter_factory import AbsIterFactory


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


class ArkDataLoader(AbsIterFactory):
    def __init__(self, data_list, dict_file, config_file, mode="train"):
        symbol_table = read_symbol_table(dict_file)
        with open(config_file, "r") as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        self.dataset_conf = configs["dataset_conf"]
        logging.info("dataloader config: {}".format(self.dataset_conf))
        self.dataset = Dataset(data_list, symbol_table,
                               self.dataset_conf, mode=mode)

    def build_iter(self, epoch, shuffle=True):
        self.dataset.set_epoch(epoch)
        data_loader = DataLoader(self.dataset,
                                 batch_size=None,
                                 pin_memory=True,
                                 num_workers=self.dataset_conf.get("num_workers", 8))
        return data_loader
