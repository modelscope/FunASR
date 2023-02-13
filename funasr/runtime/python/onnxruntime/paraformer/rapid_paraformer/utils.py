# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import functools
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Set, Tuple, Union

import numpy as np
import yaml
from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions, get_available_providers, get_device)
from typeguard import check_argument_types

from .kaldifeat import compute_fbank_feats
import warnings

root_dir = Path(__file__).resolve().parent

logger_initialized = {}


class TokenIDConverter():
    def __init__(self, token_list: Union[List, str],
                 ):
        check_argument_types()

        # self.token_list = self.load_token(token_path)
        self.token_list = token_list
        self.unk_symbol = token_list[-1]

    # @staticmethod
    # def load_token(file_path: Union[Path, str]) -> List:
    #     if not Path(file_path).exists():
    #         raise TokenIDConverterError(f'The {file_path} does not exist.')
    #
    #     with open(str(file_path), 'rb') as f:
    #         token_list = pickle.load(f)
    #
    #     if len(token_list) != len(set(token_list)):
    #         raise TokenIDConverterError('The Token exists duplicated symbol.')
    #     return token_list

    def get_num_vocabulary_size(self) -> int:
        return len(self.token_list)

    def ids2tokens(self,
                   integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        if isinstance(integers, np.ndarray) and integers.ndim != 1:
            raise TokenIDConverterError(
                f"Must be 1 dim ndarray, but got {integers.ndim}")
        return [self.token_list[i] for i in integers]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        token2id = {v: i for i, v in enumerate(self.token_list)}
        if self.unk_symbol not in token2id:
            raise TokenIDConverterError(
                f"Unknown symbol '{self.unk_symbol}' doesn't exist in the token_list"
            )
        unk_id = token2id[self.unk_symbol]
        return [token2id.get(i, unk_id) for i in tokens]


class CharTokenizer():
    def __init__(
        self,
        symbol_value: Union[Path, str, Iterable[str]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
    ):
        check_argument_types()

        self.space_symbol = space_symbol
        self.non_linguistic_symbols = self.load_symbols(symbol_value)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols

    @staticmethod
    def load_symbols(value: Union[Path, str, Iterable[str]] = None) -> Set:
        if value is None:
            return set()

        if isinstance(value, Iterable[str]):
            return set(value)

        file_path = Path(value)
        if not file_path.exists():
            logging.warning("%s doesn't exist.", file_path)
            return set()

        with file_path.open("r", encoding="utf-8") as f:
            return set(line.rstrip() for line in f)

    def text2tokens(self, line: Union[str, list]) -> List[str]:
        tokens = []
        while len(line) != 0:
            for w in self.non_linguistic_symbols:
                if line.startswith(w):
                    if not self.remove_non_linguistic_symbols:
                        tokens.append(line[: len(w)])
                    line = line[len(w):]
                    break
            else:
                t = line[0]
                if t == " ":
                    t = "<space>"
                tokens.append(t)
                line = line[1:]
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        tokens = [t if t != self.space_symbol else " " for t in tokens]
        return "".join(tokens)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'space_symbol="{self.space_symbol}"'
            f'non_linguistic_symbols="{self.non_linguistic_symbols}"'
            f")"
        )


class WavFrontend():
    """Conventional frontend structure for ASR.
    """

    def __init__(
            self,
            cmvn_file: str = None,
            fs: int = 16000,
            window: str = 'hamming',
            n_mels: int = 80,
            frame_length: int = 25,
            frame_shift: int = 10,
            filter_length_min: int = -1,
            filter_length_max: float = -1,
            lfr_m: int = 1,
            lfr_n: int = 1,
            dither: float = 1.0
    ) -> None:
        check_argument_types()

        self.fs = fs
        self.window = window
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.filter_length_min = filter_length_min
        self.filter_length_max = filter_length_max
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn_file = cmvn_file
        self.dither = dither

        if self.cmvn_file:
            self.cmvn = self.load_cmvn()

    def fbank(self,
              input_content: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        waveform_len = input_content.shape[1]
        waveform = input_content[0][:waveform_len]
        waveform = waveform * (1 << 15)
        mat = compute_fbank_feats(waveform,
                                  num_mel_bins=self.n_mels,
                                  frame_length=self.frame_length,
                                  frame_shift=self.frame_shift,
                                  dither=self.dither,
                                  energy_floor=0.0,
                                  sample_frequency=self.fs,
                                  window_type=self.window)
        feat = mat.astype(np.float32)
        feat_len = np.array(mat.shape[0]).astype(np.int32)
        return feat, feat_len

    def lfr_cmvn(self, feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.lfr_m != 1 or self.lfr_n != 1:
            feat = self.apply_lfr(feat, self.lfr_m, self.lfr_n)

        if self.cmvn_file:
            feat = self.apply_cmvn(feat)

        feat_len = np.array(feat.shape[0]).astype(np.int32)
        return feat, feat_len

    @staticmethod
    def apply_lfr(inputs: np.ndarray, lfr_m: int, lfr_n: int) -> np.ndarray:
        LFR_inputs = []

        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / lfr_n))
        left_padding = np.tile(inputs[0], ((lfr_m - 1) // 2, 1))
        inputs = np.vstack((left_padding, inputs))
        T = T + (lfr_m - 1) // 2
        for i in range(T_lfr):
            if lfr_m <= T - i * lfr_n:
                LFR_inputs.append(
                    (inputs[i * lfr_n:i * lfr_n + lfr_m]).reshape(1, -1))
            else:
                # process last LFR frame
                num_padding = lfr_m - (T - i * lfr_n)
                frame = inputs[i * lfr_n:].reshape(-1)
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))

                LFR_inputs.append(frame)
        LFR_outputs = np.vstack(LFR_inputs).astype(np.float32)
        return LFR_outputs

    def apply_cmvn(self, inputs: np.ndarray) -> np.ndarray:
        """
        Apply CMVN with mvn data
        """
        frame, dim = inputs.shape
        means = np.tile(self.cmvn[0:1, :dim], (frame, 1))
        vars = np.tile(self.cmvn[1:2, :dim], (frame, 1))
        inputs = (inputs + means) * vars
        return inputs

    def load_cmvn(self,) -> np.ndarray:
        with open(self.cmvn_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        means_list = []
        vars_list = []
        for i in range(len(lines)):
            line_item = lines[i].split()
            if line_item[0] == '<AddShift>':
                line_item = lines[i + 1].split()
                if line_item[0] == '<LearnRateCoef>':
                    add_shift_line = line_item[3:(len(line_item) - 1)]
                    means_list = list(add_shift_line)
                    continue
            elif line_item[0] == '<Rescale>':
                line_item = lines[i + 1].split()
                if line_item[0] == '<LearnRateCoef>':
                    rescale_line = line_item[3:(len(line_item) - 1)]
                    vars_list = list(rescale_line)
                    continue

        means = np.array(means_list).astype(np.float64)
        vars = np.array(vars_list).astype(np.float64)
        cmvn = np.array([means, vars])
        return cmvn


class Hypothesis(NamedTuple):
    """Hypothesis data type."""

    yseq: np.ndarray
    score: Union[float, np.ndarray] = 0
    scores: Dict[str, Union[float, np.ndarray]] = dict()
    states: Dict[str, Any] = dict()

    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
            scores={k: float(v) for k, v in self.scores.items()},
        )._asdict()


class TokenIDConverterError(Exception):
    pass


class ONNXRuntimeError(Exception):
    pass


class OrtInferSession():
    def __init__(self, model_file, device_id=-1):
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_ep = 'CUDAExecutionProvider'
        cuda_provider_options = {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": "true",
        }
        cpu_ep = 'CPUExecutionProvider'
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        EP_list = []
        if device_id != -1 and get_device() == 'GPU' \
                and cuda_ep in get_available_providers():
            EP_list = [(cuda_ep, cuda_provider_options)]
        EP_list.append((cpu_ep, cpu_provider_options))

        self._verify_model(model_file)
        self.session = InferenceSession(model_file,
                                        sess_options=sess_opt,
                                        providers=EP_list)

        if device_id != -1 and cuda_ep not in self.session.get_providers():
            warnings.warn(f'{cuda_ep} is not avaiable for current env, the inference part is automatically shifted to be executed under {cpu_ep}.\n'
                          'Please ensure the installed onnxruntime-gpu version matches your cuda and cudnn version, '
                          'you can check their relations from the offical web site: '
                          'https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html',
                          RuntimeWarning)

    def __call__(self,
                 input_content: List[Union[np.ndarray, np.ndarray]]) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(None, input_dict)
        except Exception as e:
            raise ONNXRuntimeError('ONNXRuntime inferece failed.') from e

    def get_input_names(self, ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(self,):
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = 'character'):
        return self.meta_dict[key].splitlines()

    def have_key(self, key: str = 'character') -> bool:
        self.meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in self.meta_dict.keys():
            return True
        return False

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f'{model_path} does not exists.')
        if not model_path.is_file():
            raise FileExistsError(f'{model_path} is not a file.')


def read_yaml(yaml_path: Union[str, Path]) -> Dict:
    if not Path(yaml_path).exists():
        raise FileExistsError(f'The {yaml_path} does not exist.')

    with open(str(yaml_path), 'rb') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data


@functools.lru_cache()
def get_logger(name='rapdi_paraformer'):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added.
    Args:
        name (str): Logger name.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt="%Y/%m/%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger_initialized[name] = True
    logger.propagate = False
    return logger
