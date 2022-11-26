"""Iterable dataset module."""
import copy
from io import StringIO
from pathlib import Path
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Iterator
from typing import Tuple
from typing import Union

import kaldiio
import numpy as np
import soundfile
import torch
from torch.utils.data.dataset import IterableDataset
from typeguard import check_argument_types

from funasr.datasets.dataset import ESPnetDataset


def load_kaldi(input):
    retval = kaldiio.load_mat(input)
    if isinstance(retval, tuple):
        assert len(retval) == 2, len(retval)
        if isinstance(retval[0], int) and isinstance(retval[1], np.ndarray):
            # sound scp case
            rate, array = retval
        elif isinstance(retval[1], int) and isinstance(retval[0], np.ndarray):
            # Extended ark format case
            array, rate = retval
        else:
            raise RuntimeError(f"Unexpected type: {type(retval[0])}, {type(retval[1])}")

        # Multichannel wave fie
        # array: (NSample, Channel) or (Nsample)

    else:
        # Normal ark case
        assert isinstance(retval, np.ndarray), type(retval)
        array = retval
    return array


DATA_TYPES = {
    "sound": lambda x: soundfile.read(x)[0],
    "kaldi_ark": load_kaldi,
    "npy": np.load,
    "text_int": lambda x: np.loadtxt(
        StringIO(x), ndmin=1, dtype=np.long, delimiter=" "
    ),
    "csv_int": lambda x: np.loadtxt(StringIO(x), ndmin=1, dtype=np.long, delimiter=","),
    "text_float": lambda x: np.loadtxt(
        StringIO(x), ndmin=1, dtype=np.float32, delimiter=" "
    ),
    "csv_float": lambda x: np.loadtxt(
        StringIO(x), ndmin=1, dtype=np.float32, delimiter=","
    ),
    "text": lambda x: x,
}


class IterableESPnetDataset(IterableDataset):
    """Pytorch Dataset class for ESPNet.

    Examples:
        >>> dataset = IterableESPnetDataset([('wav.scp', 'input', 'sound'),
        ...                                  ('token_int', 'output', 'text_int')],
        ...                                )
        >>> for uid, data in dataset:
        ...     data
        {'input': per_utt_array, 'output': per_utt_array}
    """

    def __init__(
        self,
        path_name_type_list: Collection[Tuple[str, str, str]],
        preprocess: Callable[
            [str, Dict[str, np.ndarray]], Dict[str, np.ndarray]
        ] = None,
        float_dtype: str = "float32",
        int_dtype: str = "long",
        key_file: str = None,
    ):
        assert check_argument_types()
        if len(path_name_type_list) == 0:
            raise ValueError(
                '1 or more elements are required for "path_name_type_list"'
            )

        path_name_type_list = copy.deepcopy(path_name_type_list)
        self.preprocess = preprocess

        self.float_dtype = float_dtype
        self.int_dtype = int_dtype
        self.key_file = key_file

        self.debug_info = {}
        non_iterable_list = []
        self.path_name_type_list = []

        for path, name, _type in path_name_type_list:
            if name in self.debug_info:
                raise RuntimeError(f'"{name}" is duplicated for data-key')
            self.debug_info[name] = path, _type
            if _type not in DATA_TYPES:
                non_iterable_list.append((path, name, _type))
            else:
                self.path_name_type_list.append((path, name, _type))

        if len(non_iterable_list) != 0:
            # Some types doesn't support iterable mode
            self.non_iterable_dataset = ESPnetDataset(
                path_name_type_list=non_iterable_list,
                preprocess=preprocess,
                float_dtype=float_dtype,
                int_dtype=int_dtype,
            )
        else:
            self.non_iterable_dataset = None

        if Path(Path(path_name_type_list[0][0]).parent, "utt2category").exists():
            self.apply_utt2category = True
        else:
            self.apply_utt2category = False

    def has_name(self, name) -> bool:
        return name in self.debug_info

    def names(self) -> Tuple[str, ...]:
        return tuple(self.debug_info)

    def __repr__(self):
        _mes = self.__class__.__name__
        _mes += "("
        for name, (path, _type) in self.debug_info.items():
            _mes += f'\n  {name}: {{"path": "{path}", "type": "{_type}"}}'
        _mes += f"\n  preprocess: {self.preprocess})"
        return _mes

    def __iter__(self) -> Iterator[Tuple[Union[str, int], Dict[str, np.ndarray]]]:
        if self.key_file is not None:
            uid_iter = (
                line.rstrip().split(maxsplit=1)[0]
                for line in open(self.key_file, encoding="utf-8")
            )
        elif len(self.path_name_type_list) != 0:
            uid_iter = (
                line.rstrip().split(maxsplit=1)[0]
                for line in open(self.path_name_type_list[0][0], encoding="utf-8")
            )
        else:
            uid_iter = iter(self.non_iterable_dataset)

        files = [open(lis[0], encoding="utf-8") for lis in self.path_name_type_list]

        worker_info = torch.utils.data.get_worker_info()

        linenum = 0
        count = 0
        for count, uid in enumerate(uid_iter, 1):
            # If num_workers>=1, split keys
            if worker_info is not None:
                if (count - 1) % worker_info.num_workers != worker_info.id:
                    continue

            # 1. Read a line from each file
            while True:
                keys = []
                values = []
                for f in files:
                    linenum += 1
                    try:
                        line = next(f)
                    except StopIteration:
                        raise RuntimeError(f"{uid} is not found in the files")
                    sps = line.rstrip().split(maxsplit=1)
                    if len(sps) != 2:
                        raise RuntimeError(
                            f"This line doesn't include a space:"
                            f" {f}:L{linenum}: {line})"
                        )
                    key, value = sps
                    keys.append(key)
                    values.append(value)

                for k_idx, k in enumerate(keys):
                    if k != keys[0]:
                        raise RuntimeError(
                            f"Keys are mismatched. Text files (idx={k_idx}) is "
                            f"not sorted or not having same keys at L{linenum}"
                        )

                # If the key is matched, break the loop
                if len(keys) == 0 or keys[0] == uid:
                    break

            # 2. Load the entry from each line and create a dict
            data = {}
            # 2.a. Load data streamingly
            for value, (path, name, _type) in zip(values, self.path_name_type_list):
                func = DATA_TYPES[_type]
                # Load entry
                array = func(value)
                data[name] = array
            if self.non_iterable_dataset is not None:
                # 2.b. Load data from non-iterable dataset
                _, from_non_iterable = self.non_iterable_dataset[uid]
                data.update(from_non_iterable)

            # 3. [Option] Apply preprocessing
            #   e.g. funasr.train.preprocessor:CommonPreprocessor
            if self.preprocess is not None:
                data = self.preprocess(uid, data)

            # 4. Force data-precision
            for name in data:
                value = data[name]
                if not isinstance(value, np.ndarray):
                    raise RuntimeError(
                        f"All values must be converted to np.ndarray object "
                        f'by preprocessing, but "{name}" is still {type(value)}.'
                    )

                # Cast to desired type
                if value.dtype.kind == "f":
                    value = value.astype(self.float_dtype)
                elif value.dtype.kind == "i":
                    value = value.astype(self.int_dtype)
                else:
                    raise NotImplementedError(f"Not supported dtype: {value.dtype}")
                data[name] = value

            yield uid, data

        if count == 0:
            raise RuntimeError("No iteration")
