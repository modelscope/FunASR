# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from abc import ABC
from abc import abstractmethod
import collections
import copy
import functools
import logging
import numbers
import re
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Mapping
from typing import Tuple
from typing import Union

import h5py
import humanfriendly
import kaldiio
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typeguard import check_argument_types
from typeguard import check_return_type

from funasr.fileio.npy_scp import NpyScpReader
from funasr.fileio.rand_gen_dataset import FloatRandomGenerateDataset
from funasr.fileio.rand_gen_dataset import IntRandomGenerateDataset
from funasr.fileio.read_text import load_num_sequence_text
from funasr.fileio.read_text import read_2column_text
from funasr.fileio.sound_scp import SoundScpReader
from funasr.utils.sized_dict import SizedDict


class AdapterForSoundScpReader(collections.abc.Mapping):
    def __init__(self, loader, dtype=None):
        assert check_argument_types()
        self.loader = loader
        self.dtype = dtype
        self.rate = None

    def keys(self):
        return self.loader.keys()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> np.ndarray:
        retval = self.loader[key]

        if isinstance(retval, tuple):
            assert len(retval) == 2, len(retval)
            if isinstance(retval[0], int) and isinstance(retval[1], np.ndarray):
                # sound scp case
                rate, array = retval
            elif isinstance(retval[1], int) and isinstance(retval[0], np.ndarray):
                # Extended ark format case
                array, rate = retval
            else:
                raise RuntimeError(
                    f"Unexpected type: {type(retval[0])}, {type(retval[1])}"
                )

            if self.rate is not None and self.rate != rate:
                raise RuntimeError(
                    f"Sampling rates are mismatched: {self.rate} != {rate}"
                )
            self.rate = rate
            # Multichannel wave fie
            # array: (NSample, Channel) or (Nsample)
            if self.dtype is not None:
                array = array.astype(self.dtype)

        else:
            # Normal ark case
            assert isinstance(retval, np.ndarray), type(retval)
            array = retval
            if self.dtype is not None:
                array = array.astype(self.dtype)

        assert isinstance(array, np.ndarray), type(array)
        return array


class H5FileWrapper:
    def __init__(self, path: str):
        self.path = path
        self.h5_file = h5py.File(path, "r")

    def __repr__(self) -> str:
        return str(self.h5_file)

    def __len__(self) -> int:
        return len(self.h5_file)

    def __iter__(self):
        return iter(self.h5_file)

    def __getitem__(self, key) -> np.ndarray:
        value = self.h5_file[key]
        return value[()]


def sound_loader(path, dest_sample_rate=16000, float_dtype=None):
    # The file is as follows:
    #   utterance_id_A /some/where/a.wav
    #   utterance_id_B /some/where/a.flac

    # NOTE(kamo): SoundScpReader doesn't support pipe-fashion
    # like Kaldi e.g. "cat a.wav |".
    # NOTE(kamo): The audio signal is normalized to [-1,1] range.
    loader = SoundScpReader(path, dest_sample_rate=16000, normalize=True, always_2d=False)

    # SoundScpReader.__getitem__() returns Tuple[int, ndarray],
    # but ndarray is desired, so Adapter class is inserted here
    return AdapterForSoundScpReader(loader, float_dtype)


def kaldi_loader(path, float_dtype=None, max_cache_fd: int = 0):
    loader = kaldiio.load_scp(path, max_cache_fd=max_cache_fd)
    return AdapterForSoundScpReader(loader, float_dtype)


def rand_int_loader(filepath, loader_type):
    # e.g. rand_int_3_10
    try:
        low, high = map(int, loader_type[len("rand_int_") :].split("_"))
    except ValueError:
        raise RuntimeError(f"e.g rand_int_3_10: but got {loader_type}")
    return IntRandomGenerateDataset(filepath, low, high)


DATA_TYPES = {
    "sound": dict(
        func=sound_loader,
        kwargs=["dest_sample_rate","float_dtype"],
        help="Audio format types which supported by sndfile wav, flac, etc."
        "\n\n"
        "   utterance_id_a a.wav\n"
        "   utterance_id_b b.wav\n"
        "   ...",
    ),
    "kaldi_ark": dict(
        func=kaldi_loader,
        kwargs=["max_cache_fd"],
        help="Kaldi-ark file type."
        "\n\n"
        "   utterance_id_A /some/where/a.ark:123\n"
        "   utterance_id_B /some/where/a.ark:456\n"
        "   ...",
    ),
    "npy": dict(
        func=NpyScpReader,
        kwargs=[],
        help="Npy file format."
        "\n\n"
        "   utterance_id_A /some/where/a.npy\n"
        "   utterance_id_B /some/where/b.npy\n"
        "   ...",
    ),
    "text_int": dict(
        func=functools.partial(load_num_sequence_text, loader_type="text_int"),
        kwargs=[],
        help="A text file in which is written a sequence of interger numbers "
        "separated by space."
        "\n\n"
        "   utterance_id_A 12 0 1 3\n"
        "   utterance_id_B 3 3 1\n"
        "   ...",
    ),
    "csv_int": dict(
        func=functools.partial(load_num_sequence_text, loader_type="csv_int"),
        kwargs=[],
        help="A text file in which is written a sequence of interger numbers "
        "separated by comma."
        "\n\n"
        "   utterance_id_A 100,80\n"
        "   utterance_id_B 143,80\n"
        "   ...",
    ),
    "text_float": dict(
        func=functools.partial(load_num_sequence_text, loader_type="text_float"),
        kwargs=[],
        help="A text file in which is written a sequence of float numbers "
        "separated by space."
        "\n\n"
        "   utterance_id_A 12. 3.1 3.4 4.4\n"
        "   utterance_id_B 3. 3.12 1.1\n"
        "   ...",
    ),
    "csv_float": dict(
        func=functools.partial(load_num_sequence_text, loader_type="csv_float"),
        kwargs=[],
        help="A text file in which is written a sequence of float numbers "
        "separated by comma."
        "\n\n"
        "   utterance_id_A 12.,3.1,3.4,4.4\n"
        "   utterance_id_B 3.,3.12,1.1\n"
        "   ...",
    ),
    "text": dict(
        func=read_2column_text,
        kwargs=[],
        help="Return text as is. The text must be converted to ndarray "
        "by 'preprocess'."
        "\n\n"
        "   utterance_id_A hello world\n"
        "   utterance_id_B foo bar\n"
        "   ...",
    ),
    "hdf5": dict(
        func=H5FileWrapper,
        kwargs=[],
        help="A HDF5 file which contains arrays at the first level or the second level."
        "   >>> f = h5py.File('file.h5')\n"
        "   >>> array1 = f['utterance_id_A']\n"
        "   >>> array2 = f['utterance_id_B']\n",
    ),
    "rand_float": dict(
        func=FloatRandomGenerateDataset,
        kwargs=[],
        help="Generate random float-ndarray which has the given shapes "
        "in the file."
        "\n\n"
        "   utterance_id_A 3,4\n"
        "   utterance_id_B 10,4\n"
        "   ...",
    ),
    "rand_int_\\d+_\\d+": dict(
        func=rand_int_loader,
        kwargs=["loader_type"],
        help="e.g. 'rand_int_0_10'. Generate random int-ndarray which has the given "
        "shapes in the path. "
        "Give the lower and upper value by the file type. e.g. "
        "rand_int_0_10 -> Generate integers from 0 to 10."
        "\n\n"
        "   utterance_id_A 3,4\n"
        "   utterance_id_B 10,4\n"
        "   ...",
    ),
}


class AbsDataset(Dataset, ABC):
    @abstractmethod
    def has_name(self, name) -> bool:
        raise NotImplementedError

    @abstractmethod
    def names(self) -> Tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, uid) -> Tuple[Any, Dict[str, np.ndarray]]:
        raise NotImplementedError


class ESPnetDataset(AbsDataset):
    """Pytorch Dataset class for ESPNet.

    Examples:
        >>> dataset = ESPnetDataset([('wav.scp', 'input', 'sound'),
        ...                          ('token_int', 'output', 'text_int')],
        ...                         )
        ... uttid, data = dataset['uttid']
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
        max_cache_size: Union[float, int, str] = 0.0,
        max_cache_fd: int = 0,
        dest_sample_rate: int = 16000,
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
        self.max_cache_fd = max_cache_fd
        self.dest_sample_rate = dest_sample_rate

        self.loader_dict = {}
        self.debug_info = {}
        for path, name, _type in path_name_type_list:
            if name in self.loader_dict:
                raise RuntimeError(f'"{name}" is duplicated for data-key')

            loader = self._build_loader(path, _type)
            self.loader_dict[name] = loader
            self.debug_info[name] = path, _type
            if len(self.loader_dict[name]) == 0:
                raise RuntimeError(f"{path} has no samples")

            # TODO(kamo): Should check consistency of each utt-keys?

        if isinstance(max_cache_size, str):
            max_cache_size = humanfriendly.parse_size(max_cache_size)
        self.max_cache_size = max_cache_size
        if max_cache_size > 0:
            self.cache = SizedDict(shared=True)
        else:
            self.cache = None

    def _build_loader(
        self, path: str, loader_type: str
    ) -> Mapping[str, Union[np.ndarray, torch.Tensor, str, numbers.Number]]:
        """Helper function to instantiate Loader.

        Args:
            path:  The file path
            loader_type:  loader_type. sound, npy, text_int, text_float, etc
        """
        for key, dic in DATA_TYPES.items():
            # e.g. loader_type="sound"
            # -> return DATA_TYPES["sound"]["func"](path)
            if re.match(key, loader_type):
                kwargs = {}
                for key2 in dic["kwargs"]:
                    if key2 == "loader_type":
                        kwargs["loader_type"] = loader_type
                    elif key2 == "dest_sample_rate" and loader_type=="sound":
                        kwargs["dest_sample_rate"] = self.dest_sample_rate
                    elif key2 == "float_dtype":
                        kwargs["float_dtype"] = self.float_dtype
                    elif key2 == "int_dtype":
                        kwargs["int_dtype"] = self.int_dtype
                    elif key2 == "max_cache_fd":
                        kwargs["max_cache_fd"] = self.max_cache_fd
                    else:
                        raise RuntimeError(f"Not implemented keyword argument: {key2}")

                func = dic["func"]
                try:
                    return func(path, **kwargs)
                except Exception:
                    if hasattr(func, "__name__"):
                        name = func.__name__
                    else:
                        name = str(func)
                    logging.error(f"An error happened with {name}({path})")
                    raise
        else:
            raise RuntimeError(f"Not supported: loader_type={loader_type}")

    def has_name(self, name) -> bool:
        return name in self.loader_dict

    def names(self) -> Tuple[str, ...]:
        return tuple(self.loader_dict)

    def __iter__(self):
        return iter(next(iter(self.loader_dict.values())))

    def __repr__(self):
        _mes = self.__class__.__name__
        _mes += "("
        for name, (path, _type) in self.debug_info.items():
            _mes += f'\n  {name}: {{"path": "{path}", "type": "{_type}"}}'
        _mes += f"\n  preprocess: {self.preprocess})"
        return _mes

    def __getitem__(self, uid: Union[str, int]) -> Tuple[str, Dict[str, np.ndarray]]:
        assert check_argument_types()

        # Change integer-id to string-id
        if isinstance(uid, int):
            d = next(iter(self.loader_dict.values()))
            uid = list(d)[uid]

        if self.cache is not None and uid in self.cache:
            data = self.cache[uid]
            return uid, data

        data = {}
        # 1. Load data from each loaders
        for name, loader in self.loader_dict.items():
            try:
                value = loader[uid]
                if isinstance(value, (list, tuple)):
                    value = np.array(value)
                if not isinstance(
                    value, (np.ndarray, torch.Tensor, str, numbers.Number)
                ):
                    raise TypeError(
                        f"Must be ndarray, torch.Tensor, str or Number: {type(value)}"
                    )
            except Exception:
                path, _type = self.debug_info[name]
                logging.error(
                    f"Error happened with path={path}, type={_type}, id={uid}"
                )
                raise

            # torch.Tensor is converted to ndarray
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            elif isinstance(value, numbers.Number):
                value = np.array([value])
            data[name] = value

        # 2. [Option] Apply preprocessing
        #   e.g. funasr.train.preprocessor:CommonPreprocessor
        if self.preprocess is not None:
            data = self.preprocess(uid, data)

        # 3. Force data-precision
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

        if self.cache is not None and self.cache.size < self.max_cache_size:
            self.cache[uid] = data

        retval = uid, data
        assert check_return_type(retval)
        return retval
