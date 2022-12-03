# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from espnet/espnet.
"""Iterable dataset module."""
import copy
from io import StringIO
from pathlib import Path
from typing import Callable, Collection, Dict, Iterator, Tuple, Union

import kaldiio
import numpy as np
import soundfile
import torch
from funasr.datasets.dataset import ESPnetDataset
from torch.utils.data.dataset import IterableDataset
from typeguard import check_argument_types

from funasr.utils import wav_utils


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
            raise RuntimeError(
                f'Unexpected type: {type(retval[0])}, {type(retval[1])}')

        # Multichannel wave fie
        # array: (NSample, Channel) or (Nsample)

    else:
        # Normal ark case
        assert isinstance(retval, np.ndarray), type(retval)
        array = retval
    return array


DATA_TYPES = {
    'sound':
    lambda x: soundfile.read(x)[0],
    'kaldi_ark':
    load_kaldi,
    'npy':
    np.load,
    'text_int':
    lambda x: np.loadtxt(StringIO(x), ndmin=1, dtype=np.long, delimiter=' '),
    'csv_int':
    lambda x: np.loadtxt(StringIO(x), ndmin=1, dtype=np.long, delimiter=','),
    'text_float':
    lambda x: np.loadtxt(StringIO(x), ndmin=1, dtype=np.float32, delimiter=' '
                         ),
    'csv_float':
    lambda x: np.loadtxt(StringIO(x), ndmin=1, dtype=np.float32, delimiter=','
                         ),
    'text':
    lambda x: x,
}


class IterableESPnetDatasetModelScope(IterableDataset):
    """Pytorch Dataset class for ESPNet.

    Examples:
        >>> dataset = IterableESPnetDataset([('wav.scp', 'input', 'sound'),
        ...                                  ('token_int', 'output', 'text_int')],
        ...                                )
        >>> for uid, data in dataset:
        ...     data
        {'input': per_utt_array, 'output': per_utt_array}
    """
    def __init__(self,
                 path_name_type_list: Collection[Tuple[any, str, str]],
                 preprocess: Callable[[str, Dict[str, np.ndarray]],
                                      Dict[str, np.ndarray]] = None,
                 float_dtype: str = 'float32',
                 int_dtype: str = 'long',
                 key_file: str = None,
                 sample_rate: Union[dict, int] = 16000):
        assert check_argument_types()
        if len(path_name_type_list) == 0:
            raise ValueError(
                '1 or more elements are required for "path_name_type_list"')

        self.preprocess = preprocess

        self.float_dtype = float_dtype
        self.int_dtype = int_dtype
        self.key_file = key_file
        self.sample_rate = sample_rate

        self.debug_info = {}
        non_iterable_list = []
        self.path_name_type_list = []

        path_list = path_name_type_list[0]
        name = path_name_type_list[1]
        _type = path_name_type_list[2]
        if name in self.debug_info:
            raise RuntimeError(f'"{name}" is duplicated for data-key')
        self.debug_info[name] = path_list, _type
        #        for path, name, _type in path_name_type_list:
        for path in path_list:
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

        self.apply_utt2category = False

    def has_name(self, name) -> bool:
        return name in self.debug_info

    def names(self) -> Tuple[str, ...]:
        return tuple(self.debug_info)

    def __repr__(self):
        _mes = self.__class__.__name__
        _mes += '('
        for name, (path, _type) in self.debug_info.items():
            _mes += f'\n  {name}: {{"path": "{path}", "type": "{_type}"}}'
        _mes += f'\n  preprocess: {self.preprocess})'
        return _mes

    def __iter__(
            self) -> Iterator[Tuple[Union[str, int], Dict[str, np.ndarray]]]:
        torch.set_printoptions(profile='default')
        count = len(self.path_name_type_list)
        for idx in range(count):
            # 2. Load the entry from each line and create a dict
            data = {}
            # 2.a. Load data streamingly

            # value:  /home/fsc/code/MaaS/MaaS-lib-nls-asr/data/test/audios/asr_example.wav
            value = self.path_name_type_list[idx][0]['file']
            uid = self.path_name_type_list[idx][0]['key']
            # name:  speech
            name = self.path_name_type_list[idx][1]
            _type = self.path_name_type_list[idx][2]
            func = DATA_TYPES[_type]
            array = func(value)

            # 2.b. audio resample
            if _type == 'sound':
                audio_sr: int = 16000
                model_sr: int = 16000
                if isinstance(self.sample_rate, int):
                    model_sr = self.sample_rate
                else:
                    if 'audio_sr' in self.sample_rate:
                        audio_sr = self.sample_rate['audio_sr']
                    if 'model_sr' in self.sample_rate:
                        model_sr = self.sample_rate['model_sr']
                array = wav_utils.torch_resample(array, audio_sr, model_sr)

            # array:  [ 1.25122070e-03  ... ]
            data[name] = array

            # 3. [Option] Apply preprocessing
            #   e.g. espnet2.train.preprocessor:CommonPreprocessor
            if self.preprocess is not None:
                data = self.preprocess(uid, data)
                # data:  {'speech': array([ 1.25122070e-03 ... 6.10351562e-03])}

            # 4. Force data-precision
            for name in data:
                # value is np.ndarray data
                value = data[name]
                if not isinstance(value, np.ndarray):
                    raise RuntimeError(
                        f'All values must be converted to np.ndarray object '
                        f'by preprocessing, but "{name}" is still {type(value)}.'
                    )

                # Cast to desired type
                if value.dtype.kind == 'f':
                    value = value.astype(self.float_dtype)
                elif value.dtype.kind == 'i':
                    value = value.astype(self.int_dtype)
                else:
                    raise NotImplementedError(
                        f'Not supported dtype: {value.dtype}')
                data[name] = value

            yield uid, data

        if count == 0:
            raise RuntimeError('No iteration')


class IterableESPnetBytesModelScope(IterableDataset):
    """Pytorch audio bytes class for ESPNet.

    Examples:
        >>> dataset = IterableESPnetBytes([('audio bytes', 'input', 'sound'),
        ...                                ('token_int', 'output', 'text_int')],
        ...                                )
        >>> for uid, data in dataset:
        ...     data
        {'input': per_utt_array, 'output': per_utt_array}
    """
    def __init__(self,
                 path_name_type_list: Collection[Tuple[any, str, str]],
                 preprocess: Callable[[str, Dict[str, np.ndarray]],
                                      Dict[str, np.ndarray]] = None,
                 float_dtype: str = 'float32',
                 int_dtype: str = 'long',
                 key_file: str = None,
                 sample_rate: Union[dict, int] = 16000):
        assert check_argument_types()
        if len(path_name_type_list) == 0:
            raise ValueError(
                '1 or more elements are required for "path_name_type_list"')

        self.preprocess = preprocess

        self.float_dtype = float_dtype
        self.int_dtype = int_dtype
        self.key_file = key_file
        self.sample_rate = sample_rate

        self.debug_info = {}
        non_iterable_list = []
        self.path_name_type_list = []

        audio_data = path_name_type_list[0]
        name = path_name_type_list[1]
        _type = path_name_type_list[2]
        if name in self.debug_info:
            raise RuntimeError(f'"{name}" is duplicated for data-key')
        self.debug_info[name] = audio_data, _type
        self.path_name_type_list.append((audio_data, name, _type))

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

        self.apply_utt2category = False

        if float_dtype == 'float32':
            self.np_dtype = np.float32

    def has_name(self, name) -> bool:
        return name in self.debug_info

    def names(self) -> Tuple[str, ...]:
        return tuple(self.debug_info)

    def __repr__(self):
        _mes = self.__class__.__name__
        _mes += '('
        for name, (path, _type) in self.debug_info.items():
            _mes += f'\n  {name}: {{"path": "{path}", "type": "{_type}"}}'
        _mes += f'\n  preprocess: {self.preprocess})'
        return _mes

    def __iter__(
            self) -> Iterator[Tuple[Union[str, int], Dict[str, np.ndarray]]]:

        torch.set_printoptions(profile='default')
        # 2. Load the entry from each line and create a dict
        data = {}
        # 2.a. Load data streamingly

        value = self.path_name_type_list[0][0]
        uid = 'pcm_data'
        # name:  speech
        name = self.path_name_type_list[0][1]
        _type = self.path_name_type_list[0][2]
        func = DATA_TYPES[_type]
        # array:  [ 1.25122070e-03  ... ]
        #        data[name] = np.frombuffer(value, dtype=self.np_dtype)

        # 2.b. byte(PCM16) to float32
        middle_data = np.frombuffer(value, dtype=np.int16)
        middle_data = np.asarray(middle_data)
        if middle_data.dtype.kind not in 'iu':
            raise TypeError("'middle_data' must be an array of integers")
        dtype = np.dtype('float32')
        if dtype.kind != 'f':
            raise TypeError("'dtype' must be a floating point type")

        i = np.iinfo(middle_data.dtype)
        abs_max = 2**(i.bits - 1)
        offset = i.min + abs_max
        array = np.frombuffer((middle_data.astype(dtype) - offset) / abs_max,
                              dtype=self.np_dtype)

        # 2.c. audio resample
        if _type == 'sound':
            audio_sr: int = 16000
            model_sr: int = 16000
            if isinstance(self.sample_rate, int):
                model_sr = self.sample_rate
            else:
                if 'audio_sr' in self.sample_rate:
                    audio_sr = self.sample_rate['audio_sr']
                if 'model_sr' in self.sample_rate:
                    model_sr = self.sample_rate['model_sr']
            array = wav_utils.torch_resample(array, audio_sr, model_sr)

        data[name] = array

        # 3. [Option] Apply preprocessing
        #   e.g. espnet2.train.preprocessor:CommonPreprocessor
        if self.preprocess is not None:
            data = self.preprocess(uid, data)
            # data:  {'speech': array([ 1.25122070e-03 ... 6.10351562e-03])}

        # 4. Force data-precision
        for name in data:
            # value is np.ndarray data
            value = data[name]
            if not isinstance(value, np.ndarray):
                raise RuntimeError(
                    f'All values must be converted to np.ndarray object '
                    f'by preprocessing, but "{name}" is still {type(value)}.')

            # Cast to desired type
            if value.dtype.kind == 'f':
                value = value.astype(self.float_dtype)
            elif value.dtype.kind == 'i':
                value = value.astype(self.int_dtype)
            else:
                raise NotImplementedError(
                    f'Not supported dtype: {value.dtype}')
            data[name] = value

        yield uid, data
