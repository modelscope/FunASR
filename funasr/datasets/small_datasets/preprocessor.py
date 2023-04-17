from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List
from typing import Union

import numpy as np
import scipy.signal
import soundfile
from typeguard import check_argument_types
from typeguard import check_return_type

from funasr.text.build_tokenizer import build_tokenizer
from funasr.text.cleaner import TextCleaner
from funasr.text.token_id_converter import TokenIDConverter


class AbsPreprocessor(ABC):
    def __init__(self, train: bool):
        self.train = train

    @abstractmethod
    def __call__(
            self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError


def forward_segment(text, dic):
    word_list = []
    i = 0
    while i < len(text):
        longest_word = text[i]
        for j in range(i + 1, len(text) + 1):
            word = text[i:j]
            if word in dic:
                if len(word) > len(longest_word):
                    longest_word = word
        word_list.append(longest_word)
        i += len(longest_word)
    return word_list


def seg_tokenize(txt, seg_dict):
    out_txt = ""
    for word in txt:
        if word in seg_dict:
            out_txt += seg_dict[word] + " "
        else:
            out_txt += "<unk>" + " "
    return out_txt.strip().split()


def seg_tokenize_wo_pattern(txt, seg_dict):
    out_txt = ""
    for word in txt:
        if word in seg_dict:
            out_txt += seg_dict[word] + " "
        else:
            out_txt += "<unk>" + " "
    return out_txt.strip().split()


def framing(
        x,
        frame_length: int = 512,
        frame_shift: int = 256,
        centered: bool = True,
        padded: bool = True,
):
    if x.size == 0:
        raise ValueError("Input array size is zero")
    if frame_length < 1:
        raise ValueError("frame_length must be a positive integer")
    if frame_length > x.shape[-1]:
        raise ValueError("frame_length is greater than input length")
    if 0 >= frame_shift:
        raise ValueError("frame_shift must be greater than 0")

    if centered:
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [
            (frame_length // 2, frame_length // 2)
        ]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = frame_length + (nseg-1)*nstep,
        #  with integer nseg
        nadd = (-(x.shape[-1] - frame_length) % frame_shift) % frame_length
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [(0, nadd)]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    # Created strided array of data segments
    if frame_length == 1 and frame_length == frame_shift:
        result = x[..., None]
    else:
        shape = x.shape[:-1] + (
            (x.shape[-1] - frame_length) // frame_shift + 1,
            frame_length,
        )
        strides = x.strides[:-1] + (frame_shift * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return result


def detect_non_silence(
        x: np.ndarray,
        threshold: float = 0.01,
        frame_length: int = 1024,
        frame_shift: int = 512,
        window: str = "boxcar",
) -> np.ndarray:
    """Power based voice activity detection.

    Args:
        x: (Channel, Time)
    >>> x = np.random.randn(1000)
    >>> detect = detect_non_silence(x)
    >>> assert x.shape == detect.shape
    >>> assert detect.dtype == np.bool
    """
    if x.shape[-1] < frame_length:
        return np.full(x.shape, fill_value=True, dtype=np.bool)

    if x.dtype.kind == "i":
        x = x.astype(np.float64)
    # framed_w: (C, T, F)
    framed_w = framing(
        x,
        frame_length=frame_length,
        frame_shift=frame_shift,
        centered=False,
        padded=True,
    )
    framed_w *= scipy.signal.get_window(window, frame_length).astype(framed_w.dtype)
    # power: (C, T)
    power = (framed_w ** 2).mean(axis=-1)
    # mean_power: (C, 1)
    mean_power = np.mean(power, axis=-1, keepdims=True)
    if np.all(mean_power == 0):
        return np.full(x.shape, fill_value=True, dtype=np.bool)
    # detect_frames: (C, T)
    detect_frames = power / mean_power > threshold
    # detects: (C, T, F)
    detects = np.broadcast_to(
        detect_frames[..., None], detect_frames.shape + (frame_shift,)
    )
    # detects: (C, TF)
    detects = detects.reshape(*detect_frames.shape[:-1], -1)
    # detects: (C, TF)
    return np.pad(
        detects,
        [(0, 0)] * (x.ndim - 1) + [(0, x.shape[-1] - detects.shape[-1])],
        mode="edge",
    )


class CommonPreprocessor(AbsPreprocessor):
    def __init__(
            self,
            train: bool,
            token_type: str = None,
            token_list: Union[Path, str, Iterable[str]] = None,
            bpemodel: Union[Path, str, Iterable[str]] = None,
            text_cleaner: Collection[str] = None,
            g2p_type: str = None,
            unk_symbol: str = "<unk>",
            space_symbol: str = "<space>",
            non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
            delimiter: str = None,
            rir_scp: str = None,
            rir_apply_prob: float = 1.0,
            noise_scp: str = None,
            noise_apply_prob: float = 1.0,
            noise_db_range: str = "3_10",
            speech_volume_normalize: float = None,
            speech_name: str = "speech",
            text_name: str = "text",
            split_with_space: bool = False,
            seg_dict_file: str = None,
    ):
        super().__init__(train)
        self.train = train
        self.speech_name = speech_name
        self.text_name = text_name
        self.speech_volume_normalize = speech_volume_normalize
        self.rir_apply_prob = rir_apply_prob
        self.noise_apply_prob = noise_apply_prob
        self.split_with_space = split_with_space
        self.seg_dict = None
        if seg_dict_file is not None:
            self.seg_dict = {}
            with open(seg_dict_file) as f:
                lines = f.readlines()
            for line in lines:
                s = line.strip().split()
                key = s[0]
                value = s[1:]
                self.seg_dict[key] = " ".join(value)

        if token_type is not None:
            if token_list is None:
                raise ValueError("token_list is required if token_type is not None")
            self.text_cleaner = TextCleaner(text_cleaner)

            self.tokenizer = build_tokenizer(
                token_type=token_type,
                bpemodel=bpemodel,
                delimiter=delimiter,
                space_symbol=space_symbol,
                non_linguistic_symbols=non_linguistic_symbols,
                g2p_type=g2p_type,
            )
            self.token_id_converter = TokenIDConverter(
                token_list=token_list,
                unk_symbol=unk_symbol,
            )
        else:
            self.text_cleaner = None
            self.tokenizer = None
            self.token_id_converter = None

        if train and rir_scp is not None:
            self.rirs = []
            with open(rir_scp, "r", encoding="utf-8") as f:
                for line in f:
                    sps = line.strip().split(None, 1)
                    if len(sps) == 1:
                        self.rirs.append(sps[0])
                    else:
                        self.rirs.append(sps[1])
        else:
            self.rirs = None

        if train and noise_scp is not None:
            self.noises = []
            with open(noise_scp, "r", encoding="utf-8") as f:
                for line in f:
                    sps = line.strip().split(None, 1)
                    if len(sps) == 1:
                        self.noises.append(sps[0])
                    else:
                        self.noises.append(sps[1])
            sps = noise_db_range.split("_")
            if len(sps) == 1:
                self.noise_db_low, self.noise_db_high = float(sps[0])
            elif len(sps) == 2:
                self.noise_db_low, self.noise_db_high = float(sps[0]), float(sps[1])
            else:
                raise ValueError(
                    "Format error: '{noise_db_range}' e.g. -3_4 -> [-3db,4db]"
                )
        else:
            self.noises = None

    def _speech_process(
            self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        assert check_argument_types()
        if self.speech_name in data:
            if self.train and (self.rirs is not None or self.noises is not None):
                speech = data[self.speech_name]
                nsamples = len(speech)

                # speech: (Nmic, Time)
                if speech.ndim == 1:
                    speech = speech[None, :]
                else:
                    speech = speech.T
                # Calc power on non shlence region
                power = (speech[detect_non_silence(speech)] ** 2).mean()

                # 1. Convolve RIR
                if self.rirs is not None and self.rir_apply_prob >= np.random.random():
                    rir_path = np.random.choice(self.rirs)
                    if rir_path is not None:
                        rir, _ = soundfile.read(
                            rir_path, dtype=np.float64, always_2d=True
                        )

                        # rir: (Nmic, Time)
                        rir = rir.T

                        # speech: (Nmic, Time)
                        # Note that this operation doesn't change the signal length
                        speech = scipy.signal.convolve(speech, rir, mode="full")[
                                 :, : speech.shape[1]
                                 ]
                        # Reverse mean power to the original power
                        power2 = (speech[detect_non_silence(speech)] ** 2).mean()
                        speech = np.sqrt(power / max(power2, 1e-10)) * speech

                # 2. Add Noise
                if (
                        self.noises is not None
                        and self.noise_apply_prob >= np.random.random()
                ):
                    noise_path = np.random.choice(self.noises)
                    if noise_path is not None:
                        noise_db = np.random.uniform(
                            self.noise_db_low, self.noise_db_high
                        )
                        with soundfile.SoundFile(noise_path) as f:
                            if f.frames == nsamples:
                                noise = f.read(dtype=np.float64, always_2d=True)
                            elif f.frames < nsamples:
                                offset = np.random.randint(0, nsamples - f.frames)
                                # noise: (Time, Nmic)
                                noise = f.read(dtype=np.float64, always_2d=True)
                                # Repeat noise
                                noise = np.pad(
                                    noise,
                                    [(offset, nsamples - f.frames - offset), (0, 0)],
                                    mode="wrap",
                                )
                            else:
                                offset = np.random.randint(0, f.frames - nsamples)
                                f.seek(offset)
                                # noise: (Time, Nmic)
                                noise = f.read(
                                    nsamples, dtype=np.float64, always_2d=True
                                )
                                if len(noise) != nsamples:
                                    raise RuntimeError(f"Something wrong: {noise_path}")
                        # noise: (Nmic, Time)
                        noise = noise.T

                        noise_power = (noise ** 2).mean()
                        scale = (
                                10 ** (-noise_db / 20)
                                * np.sqrt(power)
                                / np.sqrt(max(noise_power, 1e-10))
                        )
                        speech = speech + scale * noise

                speech = speech.T
                ma = np.max(np.abs(speech))
                if ma > 1.0:
                    speech /= ma
                data[self.speech_name] = speech

            if self.speech_volume_normalize is not None:
                speech = data[self.speech_name]
                ma = np.max(np.abs(speech))
                data[self.speech_name] = speech * self.speech_volume_normalize / ma
        assert check_return_type(data)
        return data

    def _text_process(
            self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if self.text_name in data and self.tokenizer is not None:
            text = data[self.text_name]
            text = self.text_cleaner(text)
            if self.split_with_space:
                tokens = text.strip().split(" ")
                if self.seg_dict is not None:
                    tokens = forward_segment("".join(tokens), self.seg_dict)
                    tokens = seg_tokenize(tokens, self.seg_dict)
            else:
                tokens = self.tokenizer.text2tokens(text)
            text_ints = self.token_id_converter.tokens2ids(tokens)
            data[self.text_name] = np.array(text_ints, dtype=np.int64)
        assert check_return_type(data)
        return data

    def __call__(
            self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        assert check_argument_types()

        data = self._speech_process(data)
        data = self._text_process(data)
        return data


## FIXME
class LMPreprocessor(CommonPreprocessor):
    def __init__(
            self,
            train: bool,
            token_type: str = None,
            token_list: Union[Path, str, Iterable[str]] = None,
            bpemodel: Union[Path, str, Iterable[str]] = None,
            text_cleaner: Collection[str] = None,
            g2p_type: str = None,
            unk_symbol: str = "<unk>",
            space_symbol: str = "<space>",
            non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
            delimiter: str = None,
            rir_scp: str = None,
            rir_apply_prob: float = 1.0,
            noise_scp: str = None,
            noise_apply_prob: float = 1.0,
            noise_db_range: str = "3_10",
            speech_volume_normalize: float = None,
            speech_name: str = "speech",
            text_name: str = "text",
            split_with_space: bool = False,
            seg_dict_file: str = None,
    ):
        super().__init__(train,
                         token_type,
                         token_list,
                         bpemodel,
                         text_cleaner,
                         g2p_type,
                         unk_symbol,
                         space_symbol,
                         non_linguistic_symbols,
                         delimiter,
                         rir_scp,
                         rir_apply_prob,
                         noise_scp,
                         noise_apply_prob,
                         noise_db_range,
                         speech_volume_normalize,
                         speech_name,
                         text_name,
                         split_with_space,
                         seg_dict_file,
                         )

    def _text_process(
            self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if self.text_name in data and self.tokenizer is not None:
            text = data[self.text_name]
            text = self.text_cleaner(text)
            if self.split_with_space:
                tokens = text.strip().split(" ")
                if self.seg_dict is not None:
                    tokens = seg_tokenize_wo_pattern(tokens, self.seg_dict)
            else:
                tokens = self.tokenizer.text2tokens(text)
            text_ints = self.token_id_converter.tokens2ids(tokens)
            data[self.text_name] = np.array(text_ints, dtype=np.int64)
        assert check_return_type(data)
        return data


class CommonPreprocessor_multi(AbsPreprocessor):
    def __init__(
            self,
            train: bool,
            token_type: str = None,
            token_list: Union[Path, str, Iterable[str]] = None,
            bpemodel: Union[Path, str, Iterable[str]] = None,
            text_cleaner: Collection[str] = None,
            g2p_type: str = None,
            unk_symbol: str = "<unk>",
            space_symbol: str = "<space>",
            non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
            delimiter: str = None,
            speech_name: str = "speech",
            text_name: List[str] = ["text"],
    ):
        super().__init__(train)
        self.train = train
        self.speech_name = speech_name
        self.text_name = text_name

        if token_type is not None:
            if token_list is None:
                raise ValueError("token_list is required if token_type is not None")
            self.text_cleaner = TextCleaner(text_cleaner)

            self.tokenizer = build_tokenizer(
                token_type=token_type,
                bpemodel=bpemodel,
                delimiter=delimiter,
                space_symbol=space_symbol,
                non_linguistic_symbols=non_linguistic_symbols,
                g2p_type=g2p_type,
            )
            self.token_id_converter = TokenIDConverter(
                token_list=token_list,
                unk_symbol=unk_symbol,
            )
        else:
            self.text_cleaner = None
            self.tokenizer = None
            self.token_id_converter = None

    def _text_process(
            self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        for text_n in self.text_name:
            if text_n in data and self.tokenizer is not None:
                text = data[text_n]
                text = self.text_cleaner(text)
                tokens = self.tokenizer.text2tokens(text)
                text_ints = self.token_id_converter.tokens2ids(tokens)
                data[text_n] = np.array(text_ints, dtype=np.int64)
        assert check_return_type(data)
        return data

    def __call__(
            self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        assert check_argument_types()

        if self.speech_name in data:
            # Nothing now: candidates:
            # - STFT
            # - Fbank
            # - CMVN
            # - Data augmentation
            pass

        data = self._text_process(data)
        return data


class MutliTokenizerCommonPreprocessor(CommonPreprocessor):
    def __init__(
            self,
            train: bool,
            token_type: List[str] = [None],
            token_list: List[Union[Path, str, Iterable[str]]] = [None],
            bpemodel: List[Union[Path, str, Iterable[str]]] = [None],
            text_cleaner: Collection[str] = None,
            g2p_type: str = None,
            unk_symbol: str = "<unk>",
            space_symbol: str = "<space>",
            non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
            delimiter: str = None,
            rir_scp: str = None,
            rir_apply_prob: float = 1.0,
            noise_scp: str = None,
            noise_apply_prob: float = 1.0,
            noise_db_range: str = "3_10",
            speech_volume_normalize: float = None,
            speech_name: str = "speech",
            text_name: List[str] = ["text"],
    ):
        # TODO(jiatong): sync with Kamo and Jing on interface for preprocessor
        super().__init__(
            train=train,
            token_type=token_type[0],
            token_list=token_list[0],
            bpemodel=bpemodel[0],
            text_cleaner=text_cleaner,
            g2p_type=g2p_type,
            unk_symbol=unk_symbol,
            space_symbol=space_symbol,
            non_linguistic_symbols=non_linguistic_symbols,
            delimiter=delimiter,
            speech_name=speech_name,
            text_name=text_name[0],
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            speech_volume_normalize=speech_volume_normalize,
        )

        assert (
                len(token_type) == len(token_list) == len(bpemodel) == len(text_name)
        ), "token_type, token_list, bpemodel, or processing text_name mismatched"
        self.num_tokenizer = len(token_type)
        self.tokenizer = []
        self.token_id_converter = []

        for i in range(self.num_tokenizer):
            if token_type[i] is not None:
                if token_list[i] is None:
                    raise ValueError("token_list is required if token_type is not None")

                self.tokenizer.append(
                    build_tokenizer(
                        token_type=token_type[i],
                        bpemodel=bpemodel[i],
                        delimiter=delimiter,
                        space_symbol=space_symbol,
                        non_linguistic_symbols=non_linguistic_symbols,
                        g2p_type=g2p_type,
                    )
                )
                self.token_id_converter.append(
                    TokenIDConverter(
                        token_list=token_list[i],
                        unk_symbol=unk_symbol,
                    )
                )
            else:
                self.tokenizer.append(None)
                self.token_id_converter.append(None)

        self.text_cleaner = TextCleaner(text_cleaner)
        self.text_name = text_name  # override the text_name from CommonPreprocessor

    def _text_process(
            self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        for i in range(self.num_tokenizer):
            text_name = self.text_name[i]
            if text_name in data and self.tokenizer[i] is not None:
                text = data[text_name]
                text = self.text_cleaner(text)
                tokens = self.tokenizer[i].text2tokens(text)
                text_ints = self.token_id_converter[i].tokens2ids(tokens)
                data[text_name] = np.array(text_ints, dtype=np.int64)
        assert check_return_type(data)
        return data


class CodeMixTokenizerCommonPreprocessor(CommonPreprocessor):
    def __init__(
            self,
            train: bool,
            token_type: str = None,
            token_list: Union[Path, str, Iterable[str]] = None,
            bpemodel: Union[Path, str, Iterable[str]] = None,
            text_cleaner: Collection[str] = None,
            g2p_type: str = None,
            unk_symbol: str = "<unk>",
            space_symbol: str = "<space>",
            non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
            delimiter: str = None,
            rir_scp: str = None,
            rir_apply_prob: float = 1.0,
            noise_scp: str = None,
            noise_apply_prob: float = 1.0,
            noise_db_range: str = "3_10",
            speech_volume_normalize: float = None,
            speech_name: str = "speech",
            text_name: str = "text",
            split_text_name: str = "split_text",
            split_with_space: bool = False,
            seg_dict_file: str = None,
    ):
        super().__init__(
            train=train,
            # Force to use word.
            token_type="word",
            token_list=token_list,
            bpemodel=bpemodel,
            text_cleaner=text_cleaner,
            g2p_type=g2p_type,
            unk_symbol=unk_symbol,
            space_symbol=space_symbol,
            non_linguistic_symbols=non_linguistic_symbols,
            delimiter=delimiter,
            speech_name=speech_name,
            text_name=text_name,
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            speech_volume_normalize=speech_volume_normalize,
            split_with_space=split_with_space,
            seg_dict_file=seg_dict_file,
        )
        # The data field name for split text.
        self.split_text_name = split_text_name

    @classmethod
    def split_words(cls, text: str):
        words = []
        segs = text.split()
        for seg in segs:
            # There is no space in seg.
            current_word = ""
            for c in seg:
                if len(c.encode()) == 1:
                    # This is an ASCII char.
                    current_word += c
                else:
                    # This is a Chinese char.
                    if len(current_word) > 0:
                        words.append(current_word)
                        current_word = ""
                    words.append(c)
            if len(current_word) > 0:
                words.append(current_word)
        return words

    def __call__(
            self, uid: str, data: Dict[str, Union[list, str, np.ndarray]]
    ) -> Dict[str, Union[list, np.ndarray]]:
        assert check_argument_types()
        # Split words.
        if isinstance(data[self.text_name], str):
            split_text = self.split_words(data[self.text_name])
        else:
            split_text = data[self.text_name]
        data[self.text_name] = " ".join(split_text)
        data = self._speech_process(data)
        data = self._text_process(data)
        data[self.split_text_name] = split_text
        return data

    def pop_split_text_data(self, data: Dict[str, Union[str, np.ndarray]]):
        result = data[self.split_text_name]
        del data[self.split_text_name]
        return result


class PuncTrainTokenizerCommonPreprocessor(CommonPreprocessor):
    def __init__(
            self,
            train: bool,
            token_type: List[str] = [None],
            token_list: List[Union[Path, str, Iterable[str]]] = [None],
            bpemodel: List[Union[Path, str, Iterable[str]]] = [None],
            text_cleaner: Collection[str] = None,
            g2p_type: str = None,
            unk_symbol: str = "<unk>",
            space_symbol: str = "<space>",
            non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
            delimiter: str = None,
            rir_scp: str = None,
            rir_apply_prob: float = 1.0,
            noise_scp: str = None,
            noise_apply_prob: float = 1.0,
            noise_db_range: str = "3_10",
            speech_volume_normalize: float = None,
            speech_name: str = "speech",
            text_name: List[str] = ["text"],
            vad_name: str = "vad_indexes",
    ):
        # TODO(jiatong): sync with Kamo and Jing on interface for preprocessor
        super().__init__(
            train=train,
            token_type=token_type[0],
            token_list=token_list[0],
            bpemodel=bpemodel[0],
            text_cleaner=text_cleaner,
            g2p_type=g2p_type,
            unk_symbol=unk_symbol,
            space_symbol=space_symbol,
            non_linguistic_symbols=non_linguistic_symbols,
            delimiter=delimiter,
            speech_name=speech_name,
            text_name=text_name[0],
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            speech_volume_normalize=speech_volume_normalize,
        )

        assert (
                len(token_type) == len(token_list) == len(bpemodel) == len(text_name)
        ), "token_type, token_list, bpemodel, or processing text_name mismatched"
        self.num_tokenizer = len(token_type)
        self.tokenizer = []
        self.token_id_converter = []

        for i in range(self.num_tokenizer):
            if token_type[i] is not None:
                if token_list[i] is None:
                    raise ValueError("token_list is required if token_type is not None")

                self.tokenizer.append(
                    build_tokenizer(
                        token_type=token_type[i],
                        bpemodel=bpemodel[i],
                        delimiter=delimiter,
                        space_symbol=space_symbol,
                        non_linguistic_symbols=non_linguistic_symbols,
                        g2p_type=g2p_type,
                    )
                )
                self.token_id_converter.append(
                    TokenIDConverter(
                        token_list=token_list[i],
                        unk_symbol=unk_symbol,
                    )
                )
            else:
                self.tokenizer.append(None)
                self.token_id_converter.append(None)

        self.text_cleaner = TextCleaner(text_cleaner)
        self.text_name = text_name  # override the text_name from CommonPreprocessor
        self.vad_name = vad_name

    def _text_process(
            self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        for i in range(self.num_tokenizer):
            text_name = self.text_name[i]
            if text_name in data and self.tokenizer[i] is not None:
                text = data[text_name]
                text = self.text_cleaner(text)
                tokens = self.tokenizer[i].text2tokens(text)
                if "vad:" in tokens[-1]:
                    vad = tokens[-1][4:]
                    tokens = tokens[:-1]
                    if len(vad) == 0:
                        vad = -1
                    else:
                        vad = int(vad)
                    data[self.vad_name] = np.array([vad], dtype=np.int64)
                text_ints = self.token_id_converter[i].tokens2ids(tokens)
                data[text_name] = np.array(text_ints, dtype=np.int64)


def split_to_mini_sentence(words: list, word_limit: int = 20):
    assert word_limit > 1
    if len(words) <= word_limit:
        return [words]
    sentences = []
    length = len(words)
    sentence_len = length // word_limit
    for i in range(sentence_len):
        sentences.append(words[i * word_limit:(i + 1) * word_limit])
    if length % word_limit > 0:
        sentences.append(words[sentence_len * word_limit:])
    return sentences


def build_preprocess(args):
    if args.task_name == "asr":
        pass
    else:
        raise ValueError(f"Not supported task={args.task_name}")
