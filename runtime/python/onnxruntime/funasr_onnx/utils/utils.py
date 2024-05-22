# -*- encoding: utf-8 -*-

import functools
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Set, Tuple, Union

import re
import numpy as np
import yaml

try:
    from onnxruntime import (
        GraphOptimizationLevel,
        InferenceSession,
        SessionOptions,
        get_available_providers,
        get_device,
    )
except:
    print("please pip3 install onnxruntime")
import jieba
import warnings

root_dir = Path(__file__).resolve().parent

logger_initialized = {}


def pad_list(xs, pad_value, max_len=None):
    n_batch = len(xs)
    if max_len is None:
        max_len = max(x.size(0) for x in xs)
    # pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    # numpy format
    pad = (np.zeros((n_batch, max_len)) + pad_value).astype(np.int32)
    for i in range(n_batch):
        pad[i, : xs[i].shape[0]] = xs[i]

    return pad


"""
def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask
"""


class TokenIDConverter:
    def __init__(
        self,
        token_list: Union[List, str],
    ):

        self.token_list = token_list
        self.unk_symbol = token_list[-1]
        self.token2id = {v: i for i, v in enumerate(self.token_list)}
        self.unk_id = self.token2id[self.unk_symbol]

    def get_num_vocabulary_size(self) -> int:
        return len(self.token_list)

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        if isinstance(integers, np.ndarray) and integers.ndim != 1:
            raise TokenIDConverterError(f"Must be 1 dim ndarray, but got {integers.ndim}")
        return [self.token_list[i] for i in integers]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:

        return [self.token2id.get(i, self.unk_id) for i in tokens]


class CharTokenizer:
    def __init__(
        self,
        symbol_value: Union[Path, str, Iterable[str]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
    ):

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
                    line = line[len(w) :]
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


class OrtInferSession:
    def __init__(self, model_file, device_id=-1, intra_op_num_threads=4):
        device_id = str(device_id)
        sess_opt = SessionOptions()
        sess_opt.intra_op_num_threads = intra_op_num_threads
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_ep = "CUDAExecutionProvider"
        cuda_provider_options = {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": "true",
        }
        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        EP_list = []
        if device_id != "-1" and get_device() == "GPU" and cuda_ep in get_available_providers():
            EP_list = [(cuda_ep, cuda_provider_options)]
        EP_list.append((cpu_ep, cpu_provider_options))

        self._verify_model(model_file)
        self.session = InferenceSession(model_file, sess_options=sess_opt, providers=EP_list)

        if device_id != "-1" and cuda_ep not in self.session.get_providers():
            warnings.warn(
                f"{cuda_ep} is not avaiable for current env, the inference part is automatically shifted to be executed under {cpu_ep}.\n"
                "Please ensure the installed onnxruntime-gpu version matches your cuda and cudnn version, "
                "you can check their relations from the offical web site: "
                "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
                RuntimeWarning,
            )

    def __call__(self, input_content: List[Union[np.ndarray, np.ndarray]]) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(self.get_output_names(), input_dict)
        except Exception as e:
            raise ONNXRuntimeError("ONNXRuntime inferece failed.") from e

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(
        self,
    ):
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character"):
        return self.meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        self.meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in self.meta_dict.keys():
            return True
        return False

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")


def split_to_mini_sentence(words: list, word_limit: int = 20):
    assert word_limit > 1
    if len(words) <= word_limit:
        return [words]
    sentences = []
    length = len(words)
    sentence_len = length // word_limit
    for i in range(sentence_len):
        sentences.append(words[i * word_limit : (i + 1) * word_limit])
    if length % word_limit > 0:
        sentences.append(words[sentence_len * word_limit :])
    return sentences


def code_mix_split_words(text: str):
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


def isEnglish(text: str):
    if re.search("^[a-zA-Z']+$", text):
        return True
    else:
        return False


def join_chinese_and_english(input_list):
    line = ""
    for token in input_list:
        if isEnglish(token):
            line = line + " " + token
        else:
            line = line + token

    line = line.strip()
    return line


def code_mix_split_words_jieba(seg_dict_file: str):
    jieba.load_userdict(seg_dict_file)

    def _fn(text: str):
        input_list = text.split()
        token_list_all = []
        langauge_list = []
        token_list_tmp = []
        language_flag = None
        for token in input_list:
            if isEnglish(token) and language_flag == "Chinese":
                token_list_all.append(token_list_tmp)
                langauge_list.append("Chinese")
                token_list_tmp = []
            elif not isEnglish(token) and language_flag == "English":
                token_list_all.append(token_list_tmp)
                langauge_list.append("English")
                token_list_tmp = []

            token_list_tmp.append(token)

            if isEnglish(token):
                language_flag = "English"
            else:
                language_flag = "Chinese"

        if token_list_tmp:
            token_list_all.append(token_list_tmp)
            langauge_list.append(language_flag)

        result_list = []
        for token_list_tmp, language_flag in zip(token_list_all, langauge_list):
            if language_flag == "English":
                result_list.extend(token_list_tmp)
            else:
                seg_list = jieba.cut(join_chinese_and_english(token_list_tmp), HMM=False)
                result_list.extend(seg_list)

        return result_list

    return _fn


def read_yaml(yaml_path: Union[str, Path]) -> Dict:
    if not Path(yaml_path).exists():
        raise FileExistsError(f"The {yaml_path} does not exist.")

    with open(str(yaml_path), "rb") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data


@functools.lru_cache()
def get_logger(name="funasr_onnx"):
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
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger_initialized[name] = True
    logger.propagate = False
    logging.basicConfig(level=logging.ERROR)
    return logger
