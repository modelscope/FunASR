#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
# Modified from 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)

import io
import os
import torch
import requests
import tempfile
import contextlib
import numpy as np
import librosa as sf
from typing import Union
from pathlib import Path
from typing import Generator, Union
from abc import ABCMeta, abstractmethod
import torchaudio.compliance.kaldi as Kaldi

from funasr.models.transformer.utils.nets_utils import pad_list


def check_audio_list(audio: list):
    audio_dur = 0
    for i in range(len(audio)):
        seg = audio[i]
        assert seg[1] >= seg[0], "modelscope error: Wrong time stamps."
        assert isinstance(seg[2], np.ndarray), "modelscope error: Wrong data type."
        assert (
            int(seg[1] * 16000) - int(seg[0] * 16000) == seg[2].shape[0]
        ), "modelscope error: audio data in list is inconsistent with time length."
        if i > 0:
            assert seg[0] >= audio[i - 1][1], "modelscope error: Wrong time stamps."
        audio_dur += seg[1] - seg[0]
    return audio_dur
    # assert audio_dur > 5, 'modelscope error: The effective audio duration is too short.'


def sv_preprocess(inputs: Union[np.ndarray, list]):
    output = []
    for i in range(len(inputs)):
        if isinstance(inputs[i], str):
            file_bytes = File.read(inputs[i])
            data, fs = sf.load(io.BytesIO(file_bytes), dtype="float32")
            if len(data.shape) == 2:
                data = data[:, 0]
            data = torch.from_numpy(data).unsqueeze(0)
            data = data.squeeze(0)
        elif isinstance(inputs[i], np.ndarray):
            assert len(inputs[i].shape) == 1, "modelscope error: Input array should be [N, T]"
            data = inputs[i]
            if data.dtype in ["int16", "int32", "int64"]:
                data = (data / (1 << 15)).astype("float32")
            else:
                data = data.astype("float32")
            data = torch.from_numpy(data)
        else:
            raise ValueError(
                "modelscope error: The input type is restricted to audio address and nump array."
            )
        output.append(data)
    return output


def sv_chunk(vad_segments: list, fs=16000) -> list:
    config = {
        "seg_dur": 1.5,
        "seg_shift": 0.75,
    }

    def seg_chunk(seg_data):
        seg_st = seg_data[0]
        data = seg_data[2]
        chunk_len = int(config["seg_dur"] * fs)
        chunk_shift = int(config["seg_shift"] * fs)
        last_chunk_ed = 0
        seg_res = []
        for chunk_st in range(0, data.shape[0], chunk_shift):
            chunk_ed = min(chunk_st + chunk_len, data.shape[0])
            if chunk_ed <= last_chunk_ed:
                break
            last_chunk_ed = chunk_ed
            chunk_st = max(0, chunk_ed - chunk_len)
            chunk_data = data[chunk_st:chunk_ed]
            if chunk_data.shape[0] < chunk_len:
                chunk_data = np.pad(chunk_data, (0, chunk_len - chunk_data.shape[0]), "constant")
            seg_res.append([chunk_st / fs + seg_st, chunk_ed / fs + seg_st, chunk_data])
        return seg_res

    segs = []
    for i, s in enumerate(vad_segments):
        segs.extend(seg_chunk(s))

    return segs


def extract_feature(audio):
    features = []
    feature_times = []
    feature_lengths = []
    for au in audio:
        feature = Kaldi.fbank(au.unsqueeze(0), num_mel_bins=80)
        feature = feature - feature.mean(dim=0, keepdim=True)
        features.append(feature)
        feature_times.append(au.shape[0])
        feature_lengths.append(feature.shape[0])
    # padding for batch inference
    features_padded = pad_list(features, pad_value=0)
    # features = torch.cat(features)
    return features_padded, feature_lengths, feature_times


def postprocess(
    segments: list, vad_segments: list, labels: np.ndarray, embeddings: np.ndarray
) -> list:
    assert len(segments) == len(labels)
    labels = correct_labels(labels)
    distribute_res = []
    for i in range(len(segments)):
        distribute_res.append([segments[i][0], segments[i][1], labels[i]])
    # merge the same speakers chronologically
    distribute_res = merge_seque(distribute_res)

    # accquire speaker center
    spk_embs = []
    for i in range(labels.max() + 1):
        spk_emb = embeddings[labels == i].mean(0)
        spk_embs.append(spk_emb)
    spk_embs = np.stack(spk_embs)

    def is_overlapped(t1, t2):
        if t1 > t2 + 1e-4:
            return True
        return False

    # distribute the overlap region
    for i in range(1, len(distribute_res)):
        if is_overlapped(distribute_res[i - 1][1], distribute_res[i][0]):
            p = (distribute_res[i][0] + distribute_res[i - 1][1]) / 2
            distribute_res[i][0] = p
            distribute_res[i - 1][1] = p

    # smooth the result
    distribute_res = smooth(distribute_res)

    return distribute_res


def correct_labels(labels):
    labels_id = 0
    id2id = {}
    new_labels = []
    for i in labels:
        if i not in id2id:
            id2id[i] = labels_id
            labels_id += 1
        new_labels.append(id2id[i])
    return np.array(new_labels)


def merge_seque(distribute_res):
    res = [distribute_res[0]]
    for i in range(1, len(distribute_res)):
        if distribute_res[i][2] != res[-1][2] or distribute_res[i][0] > res[-1][1]:
            res.append(distribute_res[i])
        else:
            res[-1][1] = distribute_res[i][1]
    return res


def smooth(res, mindur=0.7):
    # if only one segment, return directly
    if len(res) < 2:
        return res
    # short segments are assigned to nearest speakers.
    for i in range(len(res)):
        res[i][0] = round(res[i][0], 2)
        res[i][1] = round(res[i][1], 2)
        if res[i][1] - res[i][0] < mindur:
            if i == 0:
                res[i][2] = res[i + 1][2]
            elif i == len(res) - 1:
                res[i][2] = res[i - 1][2]
            elif res[i][0] - res[i - 1][1] <= res[i + 1][0] - res[i][1]:
                res[i][2] = res[i - 1][2]
            else:
                res[i][2] = res[i + 1][2]
    # merge the speakers
    res = merge_seque(res)

    return res


def distribute_spk(sentence_list, sd_time_list):
    sd_time_list = [(spk_st * 1000, spk_ed * 1000, spk) for spk_st, spk_ed, spk in sd_time_list]
    for d in sentence_list:
        sentence_start = d['start']
        sentence_end = d['end']
        sentence_spk = 0
        max_overlap = 0
        for spk_st, spk_ed, spk in sd_time_list:
            overlap = max(min(sentence_end, spk_ed) - max(sentence_start, spk_st), 0)
            if overlap > max_overlap:
                max_overlap = overlap
                sentence_spk = spk
            if overlap > 0 and sentence_spk == spk:
                max_overlap += overlap
        d['spk'] = int(sentence_spk)
    return sentence_list


class Storage(metaclass=ABCMeta):
    """Abstract class of storage.

    All backends need to implement two apis: ``read()`` and ``read_text()``.
    ``read()`` reads the file as a byte stream and ``read_text()`` reads
    the file as texts.
    """

    @abstractmethod
    def read(self, filepath: str):
        pass

    @abstractmethod
    def read_text(self, filepath: str):
        pass

    @abstractmethod
    def write(self, obj: bytes, filepath: Union[str, Path]) -> None:
        pass

    @abstractmethod
    def write_text(self, obj: str, filepath: Union[str, Path], encoding: str = "utf-8") -> None:
        pass


class LocalStorage(Storage):
    """Local hard disk storage"""

    def read(self, filepath: Union[str, Path]) -> bytes:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        with open(filepath, "rb") as f:
            content = f.read()
        return content

    def read_text(self, filepath: Union[str, Path], encoding: str = "utf-8") -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        with open(filepath, "r", encoding=encoding) as f:
            value_buf = f.read()
        return value_buf

    def write(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'wb' mode.

        Note:
            ``write`` will create a directory if the directory of ``filepath``
            does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        dirname = os.path.dirname(filepath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        with open(filepath, "wb") as f:
            f.write(obj)

    def write_text(self, obj: str, filepath: Union[str, Path], encoding: str = "utf-8") -> None:
        """Write data to a given ``filepath`` with 'w' mode.

        Note:
            ``write_text`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.
        """
        dirname = os.path.dirname(filepath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        with open(filepath, "w", encoding=encoding) as f:
            f.write(obj)

    @contextlib.contextmanager
    def as_local_path(self, filepath: Union[str, Path]) -> Generator[Union[str, Path], None, None]:
        """Only for unified API and do nothing."""
        yield filepath


class HTTPStorage(Storage):
    """HTTP and HTTPS storage."""

    def read(self, url):
        # TODO @wenmeng.zwm add progress bar if file is too large
        r = requests.get(url)
        r.raise_for_status()
        return r.content

    def read_text(self, url):
        r = requests.get(url)
        r.raise_for_status()
        return r.text

    @contextlib.contextmanager
    def as_local_path(self, filepath: str) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath``.

        ``as_local_path`` is decorated by :meth:`contextlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str): Download a file from ``filepath``.

        Examples:
            >>> storage = HTTPStorage()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with storage.get_local_path('http://path/to/file') as path:
            ...     # do something here
        """
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.read(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def write(self, obj: bytes, url: Union[str, Path]) -> None:
        raise NotImplementedError("write is not supported by HTTP Storage")

    def write_text(self, obj: str, url: Union[str, Path], encoding: str = "utf-8") -> None:
        raise NotImplementedError("write_text is not supported by HTTP Storage")


class OSSStorage(Storage):
    """OSS storage."""

    def __init__(self, oss_config_file=None):
        # read from config file or env var
        raise NotImplementedError("OSSStorage.__init__ to be implemented in the future")

    def read(self, filepath):
        raise NotImplementedError("OSSStorage.read to be implemented in the future")

    def read_text(self, filepath, encoding="utf-8"):
        raise NotImplementedError("OSSStorage.read_text to be implemented in the future")

    @contextlib.contextmanager
    def as_local_path(self, filepath: str) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath``.

        ``as_local_path`` is decorated by :meth:`contextlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str): Download a file from ``filepath``.

        Examples:
            >>> storage = OSSStorage()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with storage.get_local_path('http://path/to/file') as path:
            ...     # do something here
        """
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.read(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def write(self, obj: bytes, filepath: Union[str, Path]) -> None:
        raise NotImplementedError("OSSStorage.write to be implemented in the future")

    def write_text(self, obj: str, filepath: Union[str, Path], encoding: str = "utf-8") -> None:
        raise NotImplementedError("OSSStorage.write_text to be implemented in the future")


G_STORAGES = {}


class File(object):
    _prefix_to_storage: dict = {
        "oss": OSSStorage,
        "http": HTTPStorage,
        "https": HTTPStorage,
        "local": LocalStorage,
    }

    @staticmethod
    def _get_storage(uri):
        assert isinstance(uri, str), f"uri should be str type, but got {type(uri)}"

        if "://" not in uri:
            # local path
            storage_type = "local"
        else:
            prefix, _ = uri.split("://")
            storage_type = prefix

        assert storage_type in File._prefix_to_storage, (
            f"Unsupported uri {uri}, valid prefixs: " f"{list(File._prefix_to_storage.keys())}"
        )

        if storage_type not in G_STORAGES:
            G_STORAGES[storage_type] = File._prefix_to_storage[storage_type]()

        return G_STORAGES[storage_type]

    @staticmethod
    def read(uri: str) -> bytes:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        storage = File._get_storage(uri)
        return storage.read(uri)

    @staticmethod
    def read_text(uri: Union[str, Path], encoding: str = "utf-8") -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        storage = File._get_storage(uri)
        return storage.read_text(uri)

    @staticmethod
    def write(obj: bytes, uri: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'wb' mode.

        Note:
            ``write`` will create a directory if the directory of ``filepath``
            does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        storage = File._get_storage(uri)
        return storage.write(obj, uri)

    @staticmethod
    def write_text(obj: str, uri: str, encoding: str = "utf-8") -> None:
        """Write data to a given ``filepath`` with 'w' mode.

        Note:
            ``write_text`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.
        """
        storage = File._get_storage(uri)
        return storage.write_text(obj, uri)

    @contextlib.contextmanager
    def as_local_path(uri: str) -> Generator[Union[str, Path], None, None]:
        """Only for unified API and do nothing."""
        storage = File._get_storage(uri)
        with storage.as_local_path(uri) as local_path:
            yield local_path
