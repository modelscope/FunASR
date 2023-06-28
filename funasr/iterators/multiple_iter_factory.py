import logging
from typing import Callable
from typing import Collection
from typing import Iterator

import numpy as np

from funasr.iterators.abs_iter_factory import AbsIterFactory


class MultipleIterFactory(AbsIterFactory):
    def __init__(
        self,
        build_funcs: Collection[Callable[[], AbsIterFactory]],
        seed: int = 0,
        shuffle: bool = False,
    ):
        self.build_funcs = list(build_funcs)
        self.seed = seed
        self.shuffle = shuffle

    def build_iter(self, epoch: int, shuffle: bool = None) -> Iterator:
        if shuffle is None:
            shuffle = self.shuffle

        build_funcs = list(self.build_funcs)

        if shuffle:
            np.random.RandomState(epoch + self.seed).shuffle(build_funcs)

        for i, build_func in enumerate(build_funcs):
            logging.info(f"Building {i}th iter-factory...")
            iter_factory = build_func()
            assert isinstance(iter_factory, AbsIterFactory), type(iter_factory)
            yield from iter_factory.build_iter(epoch, shuffle)
