# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import collections
import copy
import logging
import numbers
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Mapping
from typing import Tuple
from typing import Union

import kaldiio
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typeguard import check_argument_types
from typeguard import check_return_type

from funasr.fileio.npy_scp import NpyScpReader
from funasr.fileio.sound_scp import SoundScpReader


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


def sound_loader(path, dest_sample_rate=16000, float_dtype=None):
    # The file is as follows:
    #   utterance_id_A /some/where/a.wav
    #   utterance_id_B /some/where/a.flac

    # NOTE(kamo): SoundScpReader doesn't support pipe-fashion
    # like Kaldi e.g. "cat a.wav |".
    # NOTE(kamo): The audio signal is normalized to [-1,1] range.
    loader = SoundScpReader(path, dest_sample_rate, normalize=True, always_2d=False)

    # SoundScpReader.__getitem__() returns Tuple[int, ndarray],
    # but ndarray is desired, so Adapter class is inserted here
    return AdapterForSoundScpReader(loader, float_dtype)


def kaldi_loader(path, float_dtype=None, max_cache_fd: int = 0):
    loader = kaldiio.load_scp(path, max_cache_fd=max_cache_fd)
    return AdapterForSoundScpReader(loader, float_dtype)


class ESPnetDataset(Dataset):
    """
        Pytorch Dataset class for FunASR, modified from ESPnet
    """

    def __init__(
            self,
            path_name_type_list: Collection[Tuple[str, str, str]],
            preprocess: Callable[
                [str, Dict[str, np.ndarray]], Dict[str, np.ndarray]
            ] = None,
            float_dtype: str = "float32",
            int_dtype: str = "long",
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

    def _build_loader(
            self, path: str, loader_type: str
    ) -> Mapping[str, Union[np.ndarray, torch.Tensor, str, numbers.Number]]:
        """Helper function to instantiate Loader.

        Args:
            path:  The file path
            loader_type:  loader_type. sound, npy, text, etc
        """
        if loader_type == "sound":
            loader = SoundScpReader(path, self.dest_sample_rate, normalize=True, always_2d=False)
            return AdapterForSoundScpReader(loader, self.float_dtype)
        elif loader_type == "kaldi_ark":
            loader = kaldiio.load_scp(path)
            return AdapterForSoundScpReader(loader, self.float_dtype)
        elif loader_type == "npy":
            return NpyScpReader()
        elif loader_type == "text":
            text_loader = {}
            with open(path, "r", encoding="utf-8") as f:
                for linenum, line in enumerate(f, 1):
                    sps = line.rstrip().split(maxsplit=1)
                    if len(sps) == 1:
                        k, v = sps[0], ""
                    else:
                        k, v = sps
                    if k in text_loader:
                        raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
                    text_loader[k] = v
            return text_loader
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

        retval = uid, data
        assert check_return_type(retval)
        return retval
