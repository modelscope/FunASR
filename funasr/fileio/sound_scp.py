import collections.abc
from pathlib import Path
from typing import Union

import numpy as np
import soundfile
import librosa
from typeguard import check_argument_types

from funasr.fileio.read_text import read_2column_text


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
    ):
        assert check_argument_types()
        self.fname = fname
        self.dtype = dtype
        self.always_2d = always_2d
        self.normalize = normalize
        self.data = read_2column_text(fname)
        self.dest_sample_rate = dest_sample_rate

    def __getitem__(self, key):
        wav = self.data[key]
        if self.normalize:
            # soundfile.read normalizes data to [-1,1] if dtype is not given
            array, rate = librosa.load(
                wav, sr=self.dest_sample_rate, mono=not self.always_2d
            )
        else:
            array, rate = librosa.load(
                wav, sr=self.dest_sample_rate, mono=not self.always_2d, dtype=self.dtype
            )

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
        assert check_argument_types()
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
