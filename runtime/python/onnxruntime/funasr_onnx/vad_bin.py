# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os.path
from pathlib import Path
from typing import List, Union, Tuple

import copy
import librosa
import numpy as np

from .utils.utils import ONNXRuntimeError, OrtInferSession, get_logger, read_yaml
from .utils.frontend import WavFrontend, WavFrontendOnline
from .utils.e2e_vad import E2EVadModel

logging = get_logger()


class Fsmn_vad:
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = None,
        batch_size: int = 1,
        device_id: Union[str, int] = "-1",
        quantize: bool = False,
        intra_op_num_threads: int = 4,
        max_end_sil: int = None,
        cache_dir: str = None,
        **kwargs,
    ):

        if not Path(model_dir).exists():
            try:
                from modelscope.hub.snapshot_download import snapshot_download
            except:
                raise "You are exporting model from modelscope, please install modelscope and try it again. To install modelscope, you could:\n" "\npip3 install -U modelscope\n" "For the users in China, you could install with the command:\n" "\npip3 install -U modelscope -i https://mirror.sjtu.edu.cn/pypi/web/simple"
            try:
                model_dir = snapshot_download(model_dir, cache_dir=cache_dir)
            except:
                raise "model_dir must be model_name in modelscope or local path downloaded from modelscope, but is {}".format(
                    model_dir
                )

        model_file = os.path.join(model_dir, "model.onnx")
        if quantize:
            model_file = os.path.join(model_dir, "model_quant.onnx")
        if not os.path.exists(model_file):
            print(".onnx is not exist, begin to export onnx")
            try:
                from funasr import AutoModel
            except:
                raise "You are exporting onnx, please install funasr and try it again. To install funasr, you could:\n" "\npip3 install -U funasr\n" "For the users in China, you could install with the command:\n" "\npip3 install -U funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple"

            model = AutoModel(model=model_dir)
            model_dir = model.export(type="onnx", quantize=quantize, **kwargs)
        config_file = os.path.join(model_dir, "config.yaml")
        cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)

        self.frontend = WavFrontend(cmvn_file=cmvn_file, **config["frontend_conf"])
        self.ort_infer = OrtInferSession(
            model_file, device_id, intra_op_num_threads=intra_op_num_threads
        )
        self.batch_size = batch_size
        self.vad_scorer = E2EVadModel(config["model_conf"])
        self.max_end_sil = (
            max_end_sil if max_end_sil is not None else config["model_conf"]["max_end_silence_time"]
        )
        self.encoder_conf = config["encoder_conf"]

    def prepare_cache(self, in_cache: list = []):
        if len(in_cache) > 0:
            return in_cache
        fsmn_layers = self.encoder_conf["fsmn_layers"]
        proj_dim = self.encoder_conf["proj_dim"]
        lorder = self.encoder_conf["lorder"]
        for i in range(fsmn_layers):
            cache = np.zeros((1, proj_dim, lorder - 1, 1)).astype(np.float32)
            in_cache.append(cache)
        return in_cache

    def __call__(self, audio_in: Union[str, np.ndarray, List[str]], **kwargs) -> List:
        waveform_list = self.load_data(audio_in, self.frontend.opts.frame_opts.samp_freq)
        waveform_nums = len(waveform_list)
        is_final = kwargs.get("kwargs", False)

        segments = [[]] * self.batch_size
        for beg_idx in range(0, waveform_nums, self.batch_size):

            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            waveform = waveform_list[beg_idx:end_idx]
            feats, feats_len = self.extract_feat(waveform)
            waveform = np.array(waveform)
            param_dict = kwargs.get("param_dict", dict())
            in_cache = param_dict.get("in_cache", list())
            in_cache = self.prepare_cache(in_cache)
            try:
                t_offset = 0
                step = int(min(feats_len.max(), 6000))
                for t_offset in range(0, int(feats_len), min(step, feats_len - t_offset)):
                    if t_offset + step >= feats_len - 1:
                        step = feats_len - t_offset
                        is_final = True
                    else:
                        is_final = False
                    feats_package = feats[:, t_offset : int(t_offset + step), :]
                    waveform_package = waveform[
                        :,
                        t_offset
                        * 160 : min(waveform.shape[-1], (int(t_offset + step) - 1) * 160 + 400),
                    ]

                    inputs = [feats_package]
                    # inputs = [feats]
                    inputs.extend(in_cache)
                    scores, out_caches = self.infer(inputs)
                    in_cache = out_caches
                    segments_part = self.vad_scorer(
                        scores,
                        waveform_package,
                        is_final=is_final,
                        max_end_sil=self.max_end_sil,
                        online=False,
                    )
                    # segments = self.vad_scorer(scores, waveform[0][None, :], is_final=is_final, max_end_sil=self.max_end_sil)

                    if segments_part:
                        for batch_num in range(0, self.batch_size):
                            segments[batch_num] += segments_part[batch_num]

            except ONNXRuntimeError:
                # logging.warning(traceback.format_exc())
                logging.warning("input wav is silence or noise")
                segments = ""

        return segments

    def load_data(self, wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List:
        def load_wav(path: str) -> np.ndarray:
            waveform, _ = librosa.load(path, sr=fs)
            return waveform

        if isinstance(wav_content, np.ndarray):
            return [wav_content]

        if isinstance(wav_content, str):
            return [load_wav(wav_content)]

        if isinstance(wav_content, list):
            return [load_wav(path) for path in wav_content]

        raise TypeError(f"The type of {wav_content} is not in [str, np.ndarray, list]")

    def extract_feat(self, waveform_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_len = [], []
        for waveform in waveform_list:
            speech, _ = self.frontend.fbank(waveform)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feats.append(feat)
            feats_len.append(feat_len)

        feats = self.pad_feats(feats, np.max(feats_len))
        feats_len = np.array(feats_len).astype(np.int32)
        return feats, feats_len

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, "constant", constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def infer(self, feats: List) -> Tuple[np.ndarray, np.ndarray]:

        outputs = self.ort_infer(feats)
        scores, out_caches = outputs[0], outputs[1:]
        return scores, out_caches


class Fsmn_vad_online:
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = None,
        batch_size: int = 1,
        device_id: Union[str, int] = "-1",
        quantize: bool = False,
        intra_op_num_threads: int = 4,
        max_end_sil: int = None,
        cache_dir: str = None,
        **kwargs,
    ):
        if not Path(model_dir).exists():
            try:
                from modelscope.hub.snapshot_download import snapshot_download
            except:
                raise "You are exporting model from modelscope, please install modelscope and try it again. To install modelscope, you could:\n" "\npip3 install -U modelscope\n" "For the users in China, you could install with the command:\n" "\npip3 install -U modelscope -i https://mirror.sjtu.edu.cn/pypi/web/simple"
            try:
                model_dir = snapshot_download(model_dir, cache_dir=cache_dir)
            except:
                raise "model_dir must be model_name in modelscope or local path downloaded from modelscope, but is {}".format(
                    model_dir
                )

        model_file = os.path.join(model_dir, "model.onnx")
        if quantize:
            model_file = os.path.join(model_dir, "model_quant.onnx")
        if not os.path.exists(model_file):
            print(".onnx is not exist, begin to export onnx")
            try:
                from funasr import AutoModel
            except:
                raise "You are exporting onnx, please install funasr and try it again. To install funasr, you could:\n" "\npip3 install -U funasr\n" "For the users in China, you could install with the command:\n" "\npip3 install -U funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple"

            model = AutoModel(model=model_dir)
            model_dir = model.export(type="onnx", quantize=quantize, **kwargs)

        config_file = os.path.join(model_dir, "config.yaml")
        cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)

        self.frontend = WavFrontendOnline(cmvn_file=cmvn_file, **config["frontend_conf"])
        self.ort_infer = OrtInferSession(
            model_file, device_id, intra_op_num_threads=intra_op_num_threads
        )
        self.batch_size = batch_size
        self.vad_scorer = E2EVadModel(config["model_conf"])
        self.max_end_sil = (
            max_end_sil if max_end_sil is not None else config["model_conf"]["max_end_silence_time"]
        )
        self.encoder_conf = config["encoder_conf"]

    def prepare_cache(self, in_cache: list = []):
        if len(in_cache) > 0:
            return in_cache
        fsmn_layers = self.encoder_conf["fsmn_layers"]
        proj_dim = self.encoder_conf["proj_dim"]
        lorder = self.encoder_conf["lorder"]
        for i in range(fsmn_layers):
            cache = np.zeros((1, proj_dim, lorder - 1, 1)).astype(np.float32)
            in_cache.append(cache)
        return in_cache

    def __call__(self, audio_in: np.ndarray, **kwargs) -> List:
        waveforms = np.expand_dims(audio_in, axis=0)

        param_dict = kwargs.get("param_dict", dict())
        is_final = param_dict.get("is_final", False)
        feats, feats_len = self.extract_feat(waveforms, is_final)
        segments = []
        if feats.size != 0:
            in_cache = param_dict.get("in_cache", list())
            in_cache = self.prepare_cache(in_cache)
            try:
                inputs = [feats]
                inputs.extend(in_cache)
                scores, out_caches = self.infer(inputs)
                param_dict["in_cache"] = out_caches
                waveforms = self.frontend.get_waveforms()
                segments = self.vad_scorer(
                    scores, waveforms, is_final=is_final, max_end_sil=self.max_end_sil, online=True
                )

            except ONNXRuntimeError:
                # logging.warning(traceback.format_exc())
                logging.warning("input wav is silence or noise")
                segments = []
        return segments

    def load_data(self, wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List:
        def load_wav(path: str) -> np.ndarray:
            waveform, _ = librosa.load(path, sr=fs)
            return waveform

        if isinstance(wav_content, np.ndarray):
            return [wav_content]

        if isinstance(wav_content, str):
            return [load_wav(wav_content)]

        if isinstance(wav_content, list):
            return [load_wav(path) for path in wav_content]

        raise TypeError(f"The type of {wav_content} is not in [str, np.ndarray, list]")

    def extract_feat(
        self, waveforms: np.ndarray, is_final: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        waveforms_lens = np.zeros(waveforms.shape[0]).astype(np.int32)
        for idx, waveform in enumerate(waveforms):
            waveforms_lens[idx] = waveform.shape[-1]

        feats, feats_len = self.frontend.extract_fbank(waveforms, waveforms_lens, is_final)
        # feats.append(feat)
        # feats_len.append(feat_len)

        # feats = self.pad_feats(feats, np.max(feats_len))
        # feats_len = np.array(feats_len).astype(np.int32)
        return feats.astype(np.float32), feats_len.astype(np.int32)

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, "constant", constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def infer(self, feats: List) -> Tuple[np.ndarray, np.ndarray]:

        outputs = self.ort_infer(feats)
        scores, out_caches = outputs[0], outputs[1:]
        return scores, out_caches
