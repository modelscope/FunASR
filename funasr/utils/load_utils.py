import os
import torch
import json
from io import BytesIO
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
import subprocess
from subprocess import CalledProcessError, run

try:
    from pydub import AudioSegment
except:
    pass


def is_ffmpeg_installed():
    """Is ffmpeg installed."""
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
    """Load audio/text/image/video data from various input formats.

    Args:
        data_or_path_or_list: File path, URL, numpy array, torch Tensor, bytes, or list.
        fs (int): Target sample rate (default 16000).
        audio_fs (int): Source audio sample rate.
        data_type (str): Input type ("sound", "text", "fbank").

    Returns:
        torch.Tensor or list: Loaded and resampled audio tensor(s).
    """
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

    # Fail fast with a clear error if an audio file path does not exist, instead of
    # silently passing the string downstream (which later crashes with a cryptic
    # "expected Tensor ... but got str" deep inside the model).
    if (
        isinstance(data_or_path_or_list, str)
        and data_type in (None, "sound")
        and not data_or_path_or_list.startswith(("http://", "https://"))
        and not os.path.exists(data_or_path_or_list)
    ):
        raise FileNotFoundError(
            f"Audio file not found: {data_or_path_or_list!r}. Pass a valid local file "
            f"path, URL, numpy array, torch.Tensor, or bytes."
        )
    if (isinstance(data_or_path_or_list, str) and os.path.exists(data_or_path_or_list)) or hasattr(data_or_path_or_list, 'read'):  # local file or bytes io
        if data_type is None or data_type == "sound":
            if hasattr(data_or_path_or_list, "read") and hasattr(data_or_path_or_list, "seek"):
                data_or_path_or_list.seek(0)
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
                try:
                    if hasattr(data_or_path_or_list, "seek"):
                        data_or_path_or_list.seek(0)
                    import soundfile as sf
                    data_np, audio_fs = sf.read(data_or_path_or_list, dtype="float32")
                    data_or_path_or_list = torch.from_numpy(data_np).squeeze()
                    if data_or_path_or_list.ndim > 1 and kwargs.get("reduce_channels", True):
                        data_or_path_or_list = data_or_path_or_list.mean(-1)
                except:
                    if hasattr(data_or_path_or_list, "seek"):
                        data_or_path_or_list.seek(0)
                    data_or_path_or_list = _load_audio_ffmpeg(data_or_path_or_list, sr=fs)
                    data_or_path_or_list = torch.from_numpy(
                        data_or_path_or_list
                    ).squeeze()  # [n_samples,]
        elif data_type == "text" and tokenizer is not None:
            with open(data_or_path_or_list, "r") as f:
                data_or_path_or_list = tokenizer.encode(f.read().strip())
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
        data_or_path_or_list = torch.from_numpy(data_or_path_or_list)  # .squeeze()  # [n_samples,]
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


def _mp3_header_fields(data: bytes, offset: int):
    """Return MPEG audio header fields, including free-format bitrate index zero."""
    if offset + 4 > len(data):
        return None

    header = int.from_bytes(data[offset : offset + 4], "big")
    if header & 0xFFE00000 != 0xFFE00000:
        return None

    version_id = (header >> 19) & 0x3
    layer_id = (header >> 17) & 0x3
    bitrate_index = (header >> 12) & 0xF
    sample_rate_index = (header >> 10) & 0x3
    padding = (header >> 9) & 0x1
    if (
        version_id == 1
        or layer_id == 0
        or bitrate_index == 15
        or sample_rate_index == 3
    ):
        return None
    return version_id, layer_id, bitrate_index, sample_rate_index, padding


def _mp3_frame_length(data: bytes, offset: int) -> int:
    """Return a fixed-bitrate MPEG audio frame length, or zero if unavailable."""
    fields = _mp3_header_fields(data, offset)
    if fields is None:
        return 0
    version_id, layer_id, bitrate_index, sample_rate_index, padding = fields
    if bitrate_index == 0:
        return 0

    mpeg1_bitrates = {
        3: (32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448),
        2: (32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384),
        1: (32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    }
    mpeg2_bitrates = {
        3: (32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256),
        2: (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
        1: (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
    }
    sample_rates = {
        3: (44100, 48000, 32000),
        2: (22050, 24000, 16000),
        0: (11025, 12000, 8000),
    }
    bitrate_table = mpeg1_bitrates if version_id == 3 else mpeg2_bitrates
    bitrate = bitrate_table[layer_id][bitrate_index - 1] * 1000
    sample_rate = sample_rates[version_id][sample_rate_index]

    if layer_id == 3:
        return (12 * bitrate // sample_rate + padding) * 4
    coefficient = 144 if version_id == 3 or layer_id == 2 else 72
    return coefficient * bitrate // sample_rate + padding


def _has_consecutive_mp3_frames(data: bytes) -> bool:
    """Avoid mistaking raw PCM that starts with one sync-like sample for MP3."""
    fields = _mp3_header_fields(data, 0)
    if fields is None:
        return False
    version_id, layer_id, bitrate_index, sample_rate_index, _ = fields
    if bitrate_index == 0:
        signature = version_id, layer_id, sample_rate_index
        padding_slot = 4 if layer_id == 3 else 1
        first_padding = fields[4] * padding_slot
        for second_offset in range(24, min(len(data) - 3, 8192)):
            next_fields = _mp3_header_fields(data, second_offset)
            if next_fields is not None and (
                next_fields[0], next_fields[1], next_fields[3]
            ) == signature and next_fields[2] == 0:
                base_frame_length = second_offset - first_padding
                third_offset = (
                    second_offset
                    + base_frame_length
                    + next_fields[4] * padding_slot
                )
                third_fields = _mp3_header_fields(data, third_offset)
                if third_fields is not None and (
                    third_fields[0], third_fields[1], third_fields[3]
                ) == signature and third_fields[2] == 0:
                    return True
        return False

    first_frame_length = _mp3_frame_length(data, 0)
    return first_frame_length > 0 and _mp3_frame_length(data, first_frame_length) > 0


def _is_audio_container(data: bytes) -> bool:
    """Return True if *data* starts with a recognised container-format magic header.

    Raw PCM byte streams have no header, so they will return False and the
    expensive pydub/ffmpeg validation round-trip can be skipped entirely.
    """
    if len(data) < 4:
        return False
    # WAV  – RIFF....WAVE
    if (
        len(data) >= 12
        and data[:4] in (b"RIFF", b"RIFX", b"RF64", b"BW64")
        and data[8:12] == b"WAVE"
    ):
        return True
    # MP3  – ID3 tag or at least two structurally valid MPEG audio frames
    has_mpeg_sync = data[0] == 0xFF and (data[1] & 0xE0) == 0xE0
    if data[:3] == b"ID3" or (has_mpeg_sync and _has_consecutive_mp3_frames(data)):
        return True
    # OGG
    if data[:4] == b"OggS":
        return True
    # FLAC
    if data[:4] == b"fLaC":
        return True
    # MP4 / M4A / AAC  – 'ftyp' box at offset 4
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return True
    # WebM / MKV
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return True
    return False


def load_bytes(input):
    """Convert raw PCM or container-formatted audio bytes to a waveform.

    Args:
        input (bytes): Raw int16 PCM or encoded audio-file bytes.

    Returns:
        numpy.ndarray: Mono float32 samples at 16 kHz.
    """
    if _is_audio_container(input):
        try:
            waveform = load_audio_text_image_video(BytesIO(input), fs=16000)
        except Exception as exc:
            raise RuntimeError(
                "Failed to decode container-formatted audio bytes. Verify that the input is "
                "a complete supported audio file and that torchaudio, soundfile, or ffmpeg "
                "is available."
            ) from exc
        else:
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.detach().cpu().numpy()
            return np.asarray(waveform, dtype=np.float32)

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


def validate_frame_rate(
    input,
    fs: int = 16000,
):

    # 将文件读取为字节流
    """Validate frame rate.
    
        Args:
            input: Input audio/text data.
            fs: TODO.
        """
    byte_data = BytesIO(input)

    # 使用 pydub 加载音频
    try:
        audio = AudioSegment.from_file(byte_data)
    except:
        raise RuntimeError(
            "You are decoding the pcm data, please install pydub first. via `pip install pydub`."
        )

    # 确保采样率为 16000 Hz
    if audio.frame_rate != fs:
        audio = audio.set_frame_rate(fs)

        # 将重新采样后的音频导出为字节流
        output = BytesIO()
        audio.export(output, format="wav")
        output.seek(0)

        # 获取重新采样后的字节流数据
        input = output.read()

    return input


def extract_fbank(data, data_len=None, data_type: str = "sound", frontend=None, **kwargs):
    """Extract filter-bank features from audio data.

    Args:
        data: Audio samples (list of numpy arrays or tensors).
        data_len: Lengths of each sample.
        data_type (str): Input type ("sound", "fbank").
        frontend: Frontend instance for feature extraction.

    Returns:
        tuple: (features_tensor, feature_lengths, feature_times)
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
        if len(data.shape) < 2:
            data = data[None, :]  # data: [batch, N]
        elif data.shape[0] > 1:
            data = data.mean(dim=0, keepdim=True)  # convert stereo/multi-channel to mono
        data_len = [data.shape[1]] if data_len is None else data_len
    elif isinstance(data, torch.Tensor):
        if len(data.shape) < 2:
            data = data[None, :]  # data: [batch, N]
        elif data.shape[0] > 1:
            data = data.mean(dim=0, keepdim=True)  # convert stereo/multi-channel to mono
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


def _load_audio_ffmpeg(file, sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str or file-like object
        The audio file or byte stream to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    pcm_params = []
    stdin_data = None
    if hasattr(file, "read"):
        if hasattr(file, "seek"):
            file.seek(0)
        stdin_data = file.read()
        input_source = "pipe:0"
    else:
        input_source = os.fspath(file)
    if isinstance(input_source, str) and input_source.lower().endswith('.pcm'):
        pcm_params = [
            "-f", "s16le",
            "-ar", str(sr),
            "-ac", "1"
        ]

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        *pcm_params,  # PCM files need input format specified before -i since PCM is raw data without headers
        "-i", input_source,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, input=stdin_data, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
