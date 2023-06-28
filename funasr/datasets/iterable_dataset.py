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
from typing import List

import kaldiio
import numpy as np
import torch
import torchaudio
import soundfile
from torch.utils.data.dataset import IterableDataset
import os.path

from funasr.datasets.dataset import ESPnetDataset


SUPPORT_AUDIO_TYPE_SETS = ['flac', 'mp3', 'ogg', 'opus', 'wav', 'pcm']

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


def load_bytes(input):
    middle_data = np.frombuffer(input, dtype=np.int16)
    middle_data = np.asarray(middle_data)
    if middle_data.dtype.kind not in 'iu':
        raise TypeError("'middle_data' must be an array of integers")
    dtype = np.dtype('float32')
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(middle_data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    array = np.frombuffer((middle_data.astype(dtype) - offset) / abs_max, dtype=np.float32)
    return array

def load_pcm(input):
    with open(input,"rb") as f:
        bytes = f.read()
    return load_bytes(bytes)

def load_wav(input):
    try:
        return torchaudio.load(input)[0].numpy()
    except:
        waveform, _ = soundfile.read(input, dtype='float32')
        if waveform.ndim == 2:
            waveform = waveform[:, 0]
        return np.expand_dims(waveform, axis=0)

DATA_TYPES = {
    "sound": load_wav,
    "pcm": load_pcm,
    "kaldi_ark": load_kaldi,
    "bytes": load_bytes,
    "waveform": lambda x: x,
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
            path_name_type_list: Collection[Tuple[any, str, str]],
            preprocess: Callable[
                [str, Dict[str, np.ndarray]], Dict[str, np.ndarray]
            ] = None,
            float_dtype: str = "float32",
            fs: dict = None,
            mc: bool = False,
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
        self.fs = fs
        self.mc = mc

        self.debug_info = {}
        non_iterable_list = []
        self.path_name_type_list = []

        if not isinstance(path_name_type_list[0], (Tuple, List)):
            path = path_name_type_list[0]
            name = path_name_type_list[1]
            _type = path_name_type_list[2]
            self.debug_info[name] = path, _type
            if _type not in DATA_TYPES:
                non_iterable_list.append((path, name, _type))
            else:
                self.path_name_type_list.append((path, name, _type))
        else:
            for path, name, _type in path_name_type_list:
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
        count = 0
        if len(self.path_name_type_list) != 0 and (self.path_name_type_list[0][2] == "bytes" or self.path_name_type_list[0][2] == "waveform"):
            linenum = len(self.path_name_type_list)
            data = {}
            for i in range(linenum):
                value = self.path_name_type_list[i][0]
                uid = 'utt_id'
                name = self.path_name_type_list[i][1]
                _type = self.path_name_type_list[i][2]
                func = DATA_TYPES[_type]
                array = func(value)
                if self.fs is not None and (name == "speech" or name == "ref_speech"):
                    audio_fs = self.fs["audio_fs"]
                    model_fs = self.fs["model_fs"]
                    if audio_fs is not None and model_fs is not None:
                        array = torch.from_numpy(array)
                        array = array.unsqueeze(0)
                        array = torchaudio.transforms.Resample(orig_freq=audio_fs,
                                                       new_freq=model_fs)(array)
                        array = array.squeeze(0).numpy()

                data[name] = array

                if self.preprocess is not None:
                    data = self.preprocess(uid, data)
                for name in data:
                    count += 1
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

        elif len(self.path_name_type_list) != 0 and self.path_name_type_list[0][2] == "sound" and not self.path_name_type_list[0][0].lower().endswith(".scp"):
            linenum = len(self.path_name_type_list)
            data = {}
            for i in range(linenum):
                value = self.path_name_type_list[i][0]
                uid = os.path.basename(self.path_name_type_list[i][0]).split(".")[0]
                name = self.path_name_type_list[i][1]
                _type = self.path_name_type_list[i][2]
                if _type == "sound":
                   audio_type = os.path.basename(value).lower()
                   if audio_type.rfind(".pcm") >= 0:
                       _type = "pcm"
                func = DATA_TYPES[_type]
                array = func(value)
                if self.fs is not None and (name == "speech" or name == "ref_speech"):
                    audio_fs = self.fs["audio_fs"]
                    model_fs = self.fs["model_fs"]
                    if audio_fs is not None and model_fs is not None:
                        array = torch.from_numpy(array)
                        array = torchaudio.transforms.Resample(orig_freq=audio_fs,
                                                               new_freq=model_fs)(array)
                        array = array.numpy()
                        
                if _type == "sound":
                    if self.mc:
                        data[name] = array.transpose((1, 0))
                    else:
                        data[name] = array[0]
                else:
                    data[name] = array

                if self.preprocess is not None:
                    data = self.preprocess(uid, data)
                for name in data:
                    count += 1
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

        else:
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
                    if _type == "sound":
                        audio_type = os.path.basename(value).lower()
                        if audio_type.rfind(".pcm") >= 0:
                            _type = "pcm"
                    func = DATA_TYPES[_type]
                    # Load entry
                    array = func(value)
                    if self.fs is not None and name == "speech":
                        audio_fs = self.fs["audio_fs"]
                        model_fs = self.fs["model_fs"]
                        if audio_fs is not None and model_fs is not None:
                            array = torch.from_numpy(array)
                            array = torchaudio.transforms.Resample(orig_freq=audio_fs,
                                                                   new_freq=model_fs)(array)
                            array = array.numpy()
                    if _type == "sound":
                        if self.mc:
                            data[name] = array.transpose((1, 0))
                        else:
                            data[name] = array[0]
                    else:
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

