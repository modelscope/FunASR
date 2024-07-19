import os
import torch
import json
import torch.distributed as dist
import numpy as np
import kaldiio
import librosa
import torchaudio
import time
import logging
from torch.nn.utils.rnn import pad_sequence

try:
    from funasr.download.file import download_from_url
except:
    print("urllib is not installed, if you infer from url, please install it first.")
import pdb
import subprocess
from subprocess import CalledProcessError, run


def is_ffmpeg_installed():
    try:
        output = subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
        return "ffmpeg version" in output.decode("utf-8")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


use_ffmpeg = False
if is_ffmpeg_installed():
    use_ffmpeg = True
else:
    print(
        "Notice: ffmpeg is not installed. torchaudio is used to load audio\n"
        "If you want to use ffmpeg backend to load audio, please install it by:"
        "\n\tsudo apt install ffmpeg # ubuntu"
        "\n\t# brew install ffmpeg # mac"
    )


def load_audio_text_image_video(
    data_or_path_or_list,
    fs: int = 16000,
    audio_fs: int = 16000,
    data_type="sound",
    tokenizer=None,
    **kwargs,
):
    if isinstance(data_or_path_or_list, (list, tuple)):
        if data_type is not None and isinstance(data_type, (list, tuple)):
            data_types = [data_type] * len(data_or_path_or_list)
            data_or_path_or_list_ret = [[] for d in data_type]
            for i, (data_type_i, data_or_path_or_list_i) in enumerate(
                zip(data_types, data_or_path_or_list)
            ):
                for j, (data_type_j, data_or_path_or_list_j) in enumerate(
                    zip(data_type_i, data_or_path_or_list_i)
                ):
                    data_or_path_or_list_j = load_audio_text_image_video(
                        data_or_path_or_list_j,
                        fs=fs,
                        audio_fs=audio_fs,
                        data_type=data_type_j,
                        tokenizer=tokenizer,
                        **kwargs,
                    )
                    data_or_path_or_list_ret[j].append(data_or_path_or_list_j)

            return data_or_path_or_list_ret
        else:
            return [
                load_audio_text_image_video(
                    audio, fs=fs, audio_fs=audio_fs, data_type=data_type, **kwargs
                )
                for audio in data_or_path_or_list
            ]
    if isinstance(data_or_path_or_list, str) and data_or_path_or_list.startswith(
            ("http://", "https://")
    ):  # download url to local file
        data_or_path_or_list = download_from_url(data_or_path_or_list)

    if isinstance(data_or_path_or_list, str) and os.path.exists(data_or_path_or_list):  # local file
        if data_type is None or data_type == "sound":
            # if use_ffmpeg:
            #     data_or_path_or_list = _load_audio_ffmpeg(data_or_path_or_list, sr=fs)
            #     data_or_path_or_list = torch.from_numpy(data_or_path_or_list).squeeze()  # [n_samples,]
            # else:
            #     data_or_path_or_list, audio_fs = torchaudio.load(data_or_path_or_list)
            #     if kwargs.get("reduce_channels", True):
            #         data_or_path_or_list = data_or_path_or_list.mean(0)
            try:
                data_or_path_or_list, audio_fs = torchaudio.load(data_or_path_or_list)
                if kwargs.get("reduce_channels", True):
                    data_or_path_or_list = data_or_path_or_list.mean(0)
            except:
                data_or_path_or_list = _load_audio_ffmpeg(data_or_path_or_list, sr=fs)
                data_or_path_or_list = torch.from_numpy(
                    data_or_path_or_list
                ).squeeze()  # [n_samples,]
        elif data_type == "text" and tokenizer is not None:
            data_or_path_or_list = tokenizer.encode(data_or_path_or_list)
        elif data_type == "image":  # undo
            pass
        elif data_type == "video":  # undo
            pass

        # if data_in is a file or url, set is_final=True
        if "cache" in kwargs:
            kwargs["cache"]["is_final"] = True
            kwargs["cache"]["is_streaming_input"] = False
    elif isinstance(data_or_path_or_list, str) and data_type == "text" and tokenizer is not None:
        data_or_path_or_list = tokenizer.encode(data_or_path_or_list)
    elif isinstance(data_or_path_or_list, np.ndarray):  # audio sample point
        data_or_path_or_list = torch.from_numpy(data_or_path_or_list).squeeze()  # [n_samples,]
    elif isinstance(data_or_path_or_list, str) and data_type == "kaldi_ark":
        data_mat = kaldiio.load_mat(data_or_path_or_list)
        if isinstance(data_mat, tuple):
            audio_fs, mat = data_mat
        else:
            mat = data_mat
        if mat.dtype == "int16" or mat.dtype == "int32":
            mat = mat.astype(np.float64)
            mat = mat / 32768
        if mat.ndim == 2:
            mat = mat[:, 0]
        data_or_path_or_list = mat
    else:
        pass
        # print(f"unsupport data type: {data_or_path_or_list}, return raw data")

    if audio_fs != fs and data_type != "text":
        resampler = torchaudio.transforms.Resample(audio_fs, fs)
        data_or_path_or_list = resampler(data_or_path_or_list[None, :])[0, :]
    return data_or_path_or_list


def load_bytes(input):
    middle_data = np.frombuffer(input, dtype=np.int16)
    middle_data = np.asarray(middle_data)
    if middle_data.dtype.kind not in "iu":
        raise TypeError("'middle_data' must be an array of integers")
    dtype = np.dtype("float32")
    if dtype.kind != "f":
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(middle_data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    array = np.frombuffer((middle_data.astype(dtype) - offset) / abs_max, dtype=np.float32)
    return array


def extract_fbank(data, data_len=None, data_type: str = "sound", frontend=None, **kwargs):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
        if len(data.shape) < 2:
            data = data[None, :]  # data: [batch, N]
        data_len = [data.shape[1]] if data_len is None else data_len
    elif isinstance(data, torch.Tensor):
        if len(data.shape) < 2:
            data = data[None, :]  # data: [batch, N]
        data_len = [data.shape[1]] if data_len is None else data_len
    elif isinstance(data, (list, tuple)):
        data_list, data_len = [], []
        for data_i in data:
            if isinstance(data_i, np.ndarray):
                data_i = torch.from_numpy(data_i)
            data_list.append(data_i)
            data_len.append(data_i.shape[0])
        data = pad_sequence(data_list, batch_first=True)  # data: [batch, N]

    data, data_len = frontend(data, data_len, **kwargs)

    if isinstance(data_len, (list, tuple)):
        data_len = torch.tensor([data_len])
    return data.to(torch.float32), data_len.to(torch.int32)


def _load_audio_ffmpeg(file: str, sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
