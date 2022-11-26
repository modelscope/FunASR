import collections
from pathlib import Path
from typing import Union

import numpy as np
from typeguard import check_argument_types

from funasr.fileio.read_text import load_num_sequence_text


class FloatRandomGenerateDataset(collections.abc.Mapping):
    """Generate float array from shape.txt.

    Examples:
        shape.txt
        uttA 123,83
        uttB 34,83
        >>> dataset = FloatRandomGenerateDataset("shape.txt")
        >>> array = dataset["uttA"]
        >>> assert array.shape == (123, 83)
        >>> array = dataset["uttB"]
        >>> assert array.shape == (34, 83)

    """

    def __init__(
        self,
        shape_file: Union[Path, str],
        dtype: Union[str, np.dtype] = "float32",
        loader_type: str = "csv_int",
    ):
        assert check_argument_types()
        shape_file = Path(shape_file)
        self.utt2shape = load_num_sequence_text(shape_file, loader_type)
        self.dtype = np.dtype(dtype)

    def __iter__(self):
        return iter(self.utt2shape)

    def __len__(self):
        return len(self.utt2shape)

    def __getitem__(self, item) -> np.ndarray:
        shape = self.utt2shape[item]
        return np.random.randn(*shape).astype(self.dtype)


class IntRandomGenerateDataset(collections.abc.Mapping):
    """Generate float array from shape.txt

    Examples:
        shape.txt
        uttA 123,83
        uttB 34,83
        >>> dataset = IntRandomGenerateDataset("shape.txt", low=0, high=10)
        >>> array = dataset["uttA"]
        >>> assert array.shape == (123, 83)
        >>> array = dataset["uttB"]
        >>> assert array.shape == (34, 83)

    """

    def __init__(
        self,
        shape_file: Union[Path, str],
        low: int,
        high: int = None,
        dtype: Union[str, np.dtype] = "int64",
        loader_type: str = "csv_int",
    ):
        assert check_argument_types()
        shape_file = Path(shape_file)
        self.utt2shape = load_num_sequence_text(shape_file, loader_type)
        self.dtype = np.dtype(dtype)
        self.low = low
        self.high = high

    def __iter__(self):
        return iter(self.utt2shape)

    def __len__(self):
        return len(self.utt2shape)

    def __getitem__(self, item) -> np.ndarray:
        shape = self.utt2shape[item]
        return np.random.randint(self.low, self.high, size=shape, dtype=self.dtype)
