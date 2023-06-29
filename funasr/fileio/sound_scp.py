import collections.abc
from pathlib import Path
from typing import List, Tuple, Union

import random
import numpy as np
import soundfile
import librosa

import torch
import torchaudio

from funasr.fileio.read_text import read_2column_text

def soundfile_read(
    wavs: Union[str, List[str]],
    dtype=None,
    always_2d: bool = False,
    concat_axis: int = 1,
    start: int = 0,
    end: int = None,
    return_subtype: bool = False,
) -> Tuple[np.array, int]:
    if isinstance(wavs, str):
        wavs = [wavs]

    arrays = []
    subtypes = []
    prev_rate = None
    prev_wav = None
    for wav in wavs:
        with soundfile.SoundFile(wav) as f:
            f.seek(start)
            if end is not None:
                frames = end - start
            else:
                frames = -1
            if dtype == "float16":
                array = f.read(
                    frames,
                    dtype="float32",
                    always_2d=always_2d,
                ).astype(dtype)
            else:
                array = f.read(frames, dtype=dtype, always_2d=always_2d)
            rate = f.samplerate
            subtype = f.subtype
            subtypes.append(subtype)

        if len(wavs) > 1 and array.ndim == 1 and concat_axis == 1:
            # array: (Time, Channel)
            array = array[:, None]

        if prev_wav is not None:
            if prev_rate != rate:
                raise RuntimeError(
                    f"'{prev_wav}' and '{wav}' have mismatched sampling rate: "
                    f"{prev_rate} != {rate}"
                )

            dim1 = arrays[0].shape[1 - concat_axis]
            dim2 = array.shape[1 - concat_axis]
            if dim1 != dim2:
                raise RuntimeError(
                    "Shapes must match with "
                    f"{1 - concat_axis} axis, but gut {dim1} and {dim2}"
                )

        prev_rate = rate
        prev_wav = wav
        arrays.append(array)

    if len(arrays) == 1:
        array = arrays[0]
    else:
        array = np.concatenate(arrays, axis=concat_axis)

    if return_subtype:
        return array, rate, subtypes
    else:
        return array, rate


class SoundScpReader(collections.abc.Mapping):
    """Reader class for 'wav.scp'.

    Examples:
        key1 /some/path/a.wav
        key2 /some/path/b.wav
        key3 /some/path/c.wav
        key4 /some/path/d.wav
        ...

        >>> reader = SoundScpReader('wav.scp')
        >>> rate, array = reader['key1']

    """

    def __init__(
        self,
        fname,
        dtype=np.int16,
        always_2d: bool = False,
        normalize: bool = False,
        dest_sample_rate: int = 16000,
        speed_perturb: Union[list, tuple] = None,
    ):
        self.fname = fname
        self.dtype = dtype
        self.always_2d = always_2d
        self.normalize = normalize
        self.data = read_2column_text(fname)
        self.dest_sample_rate = dest_sample_rate
        self.speed_perturb = speed_perturb

    def __getitem__(self, key):
        wav = self.data[key]
        if self.normalize:
            # soundfile.read normalizes data to [-1,1] if dtype is not given
            array, rate = librosa.load(
                wav, sr=self.dest_sample_rate, mono=self.always_2d
            )
        else:
            array, rate = librosa.load(
                wav, sr=self.dest_sample_rate, mono=self.always_2d, dtype=self.dtype
            )

        if self.speed_perturb is not None:
            speed = random.choice(self.speed_perturb)
            if speed != 1.0:
                array, _ = torchaudio.sox_effects.apply_effects_tensor(
                    torch.tensor(array).view(1, -1), rate,
                    [['speed', str(speed)], ['rate', str(rate)]])
                array = array.view(-1).numpy()

        if array.ndim==2:
            array=array.transpose((1, 0))

        return rate, array

    def get_path(self, key):
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()


class SoundScpWriter:
    """Writer class for 'wav.scp'

    Examples:
        key1 /some/path/a.wav
        key2 /some/path/b.wav
        key3 /some/path/c.wav
        key4 /some/path/d.wav
        ...

        >>> writer = SoundScpWriter('./data/', './data/feat.scp')
        >>> writer['aa'] = 16000, numpy_array
        >>> writer['bb'] = 16000, numpy_array

    """

    def __init__(
        self,
        outdir: Union[Path, str],
        scpfile: Union[Path, str],
        format="wav",
        dtype=None,
    ):
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.format = format
        self.dtype = dtype

        self.data = {}

    def __setitem__(self, key: str, value):
        rate, signal = value
        assert isinstance(rate, int), type(rate)
        assert isinstance(signal, np.ndarray), type(signal)
        if signal.ndim not in (1, 2):
            raise RuntimeError(f"Input signal must be 1 or 2 dimension: {signal.ndim}")
        if signal.ndim == 1:
            signal = signal[:, None]

        wav = self.dir / f"{key}.{self.format}"
        wav.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(str(wav), signal, rate)

        self.fscp.write(f"{key} {wav}\n")

        # Store the file path
        self.data[key] = str(wav)

    def get_path(self, key):
        return self.data[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()
