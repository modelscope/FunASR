#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)


import torch
import os.path
import librosa
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple

from .utils.utils import (
    CharTokenizer,
    get_logger,
    read_yaml,
)
from .utils.frontend import WavFrontend

logging = get_logger()


class SenseVoiceSmallTorchScript:
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = None,
        batch_size: int = 1,
        device_id: Union[str, int] = "-1",
        plot_timestamp_to: str = "",
        quantize: bool = False,
        intra_op_num_threads: int = 4,
        cache_dir: str = None,
        **kwargs,
    ):
        if quantize:
            model_file = os.path.join(model_dir, "model_quant.torchscript")
        else:
            model_file = os.path.join(model_dir, "model.torchscript")

        config_file = os.path.join(model_dir, "config.yaml")
        cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)
        # token_list = os.path.join(model_dir, "tokens.json")
        # with open(token_list, "r", encoding="utf-8") as f:
        #     token_list = json.load(f)

        # self.converter = TokenIDConverter(token_list)
        self.tokenizer = CharTokenizer()
        config["frontend_conf"]['cmvn_file'] = cmvn_file
        self.frontend = WavFrontend(**config["frontend_conf"])
        self.ort_infer = torch.jit.load(model_file)
        self.batch_size = batch_size
        self.blank_id = 0

    def __call__(self, 
                 wav_content: Union[str, np.ndarray, List[str]], 
                 language: List, 
                 textnorm: List,
                 tokenizer=None,
                 **kwargs) -> List:
        waveform_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
        waveform_nums = len(waveform_list)
        asr_res = []
        for beg_idx in range(0, waveform_nums, self.batch_size):
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
            ctc_logits, encoder_out_lens = self.ort_infer(torch.Tensor(feats), 
                                                          torch.Tensor(feats_len), 
                                                          torch.tensor(language),
                                                          torch.tensor(textnorm)
                                                          )
            # support batch_size=1 only currently
            x = ctc_logits[0, : encoder_out_lens[0].item(), :]
            yseq = x.argmax(dim=-1)
            yseq = torch.unique_consecutive(yseq, dim=-1)

            mask = yseq != self.blank_id
            token_int = yseq[mask].tolist()
            
            if tokenizer is not None:
                asr_res.append(tokenizer.tokens2text(token_int))
            else:
                asr_res.append(token_int)
        return asr_res

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

