#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union

import numpy as np
import torch
from scipy.ndimage import median_filter
from torch.nn import functional as F
from typeguard import check_argument_types

from funasr.models.frontend.wav_frontend import WavFrontendMel23
from funasr.tasks.diar import DiarTask
from funasr.build_utils.build_model_from_file import build_model_from_file
from funasr.torch_utils.device_funcs import to_device
from funasr.utils.misc import statistic_model_parameters


class Speech2DiarizationEEND:
    """Speech2Diarlization class

    Examples:
        >>> import soundfile
        >>> import numpy as np
        >>> speech2diar = Speech2DiarizationEEND("diar_sond_config.yml", "diar_sond.pb")
        >>> profile = np.load("profiles.npy")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2diar(audio, profile)
        {"spk1": [(int, int), ...], ...}

    """

    def __init__(
            self,
            diar_train_config: Union[Path, str] = None,
            diar_model_file: Union[Path, str] = None,
            device: str = "cpu",
            dtype: str = "float32",
    ):
        assert check_argument_types()

        # 1. Build Diarization model
        diar_model, diar_train_args = build_model_from_file(
            config_file=diar_train_config,
            model_file=diar_model_file,
            device=device,
            task_name="diar",
            mode="eend-ola",
        )
        frontend = None
        if diar_train_args.frontend is not None and diar_train_args.frontend_conf is not None:
            frontend = WavFrontendMel23(**diar_train_args.frontend_conf)

        # set up seed for eda
        np.random.seed(diar_train_args.seed)
        torch.manual_seed(diar_train_args.seed)
        torch.cuda.manual_seed(diar_train_args.seed)
        os.environ['PYTORCH_SEED'] = str(diar_train_args.seed)
        logging.info("diar_model: {}".format(diar_model))
        logging.info("diar_train_args: {}".format(diar_train_args))
        diar_model.to(dtype=getattr(torch, dtype)).eval()

        self.diar_model = diar_model
        self.diar_train_args = diar_train_args
        self.device = device
        self.dtype = dtype
        self.frontend = frontend

    @torch.no_grad()
    def __call__(
            self,
            speech: Union[torch.Tensor, np.ndarray],
            speech_lengths: Union[torch.Tensor, np.ndarray] = None
    ):
        """Inference

        Args:
            speech: Input speech data
        Returns:
            diarization results

        """
        assert check_argument_types()
        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        if self.frontend is not None:
            feats, feats_len = self.frontend.forward(speech, speech_lengths)
            feats = to_device(feats, device=self.device)
            feats_len = feats_len.int()
            self.diar_model.frontend = None
        else:
            feats = speech
            feats_len = speech_lengths
        batch = {"speech": feats, "speech_lengths": feats_len}
        batch = to_device(batch, device=self.device)
        results = self.diar_model.estimate_sequential(**batch)

        return results

    @staticmethod
    def from_pretrained(
            model_tag: Optional[str] = None,
            **kwargs: Optional[Any],
    ):
        """Build Speech2Diarization instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2Diarization: Speech2Diarization instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return Speech2DiarizationEEND(**kwargs)


class Speech2DiarizationSOND:
    """Speech2Xvector class

    Examples:
        >>> import soundfile
        >>> import numpy as np
        >>> speech2diar = Speech2DiarizationSOND("diar_sond_config.yml", "diar_sond.pb")
        >>> profile = np.load("profiles.npy")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2diar(audio, profile)
        {"spk1": [(int, int), ...], ...}

    """

    def __init__(
            self,
            diar_train_config: Union[Path, str] = None,
            diar_model_file: Union[Path, str] = None,
            device: Union[str, torch.device] = "cpu",
            batch_size: int = 1,
            dtype: str = "float32",
            streaming: bool = False,
            smooth_size: int = 83,
            dur_threshold: float = 10,
    ):
        assert check_argument_types()

        # TODO: 1. Build Diarization model
        diar_model, diar_train_args = build_model_from_file(
            config_file=diar_train_config,
            model_file=diar_model_file,
            device=device,
            task_name="diar",
            mode="sond",
        )
        logging.info("diar_model: {}".format(diar_model))
        logging.info("model parameter number: {}".format(statistic_model_parameters(diar_model)))
        logging.info("diar_train_args: {}".format(diar_train_args))
        diar_model.to(dtype=getattr(torch, dtype)).eval()

        self.diar_model = diar_model
        self.diar_train_args = diar_train_args
        self.token_list = diar_train_args.token_list
        self.smooth_size = smooth_size
        self.dur_threshold = dur_threshold
        self.device = device
        self.dtype = dtype

    def smooth_multi_labels(self, multi_label):
        multi_label = median_filter(multi_label, (self.smooth_size, 1), mode="constant", cval=0.0).astype(int)
        return multi_label

    @staticmethod
    def calc_spk_turns(label_arr, spk_list):
        turn_list = []
        length = label_arr.shape[0]
        n_spk = label_arr.shape[1]
        for k in range(n_spk):
            if spk_list[k] == "None":
                continue
            in_utt = False
            start = 0
            for i in range(length):
                if label_arr[i, k] == 1 and in_utt is False:
                    start = i
                    in_utt = True
                if label_arr[i, k] == 0 and in_utt is True:
                    turn_list.append([spk_list[k], start, i - start])
                    in_utt = False
            if in_utt:
                turn_list.append([spk_list[k], start, length - start])
        return turn_list

    @staticmethod
    def seq2arr(seq, vec_dim=8):
        def int2vec(x, vec_dim=8, dtype=np.int):
            b = ('{:0' + str(vec_dim) + 'b}').format(x)
            # little-endian order: lower bit first
            return (np.array(list(b)[::-1]) == '1').astype(dtype)

        # process oov
        seq = np.array([int(x) for x in seq])
        new_seq = []
        for i, x in enumerate(seq):
            if x < 2 ** vec_dim:
                new_seq.append(x)
            else:
                idx_list = np.where(seq < 2 ** vec_dim)[0]
                if len(idx_list) > 0:
                    idx = np.abs(idx_list - i).argmin()
                    new_seq.append(seq[idx_list[idx]])
                else:
                    new_seq.append(0)
        return np.row_stack([int2vec(x, vec_dim) for x in new_seq])

    def post_processing(self, raw_logits: torch.Tensor, spk_num: int, output_format: str = "speaker_turn"):
        logits_idx = raw_logits.argmax(-1)  # B, T, vocab_size -> B, T
        # upsampling outputs to match inputs
        ut = logits_idx.shape[1] * self.diar_model.encoder.time_ds_ratio
        logits_idx = F.upsample(
            logits_idx.unsqueeze(1).float(),
            size=(ut,),
            mode="nearest",
        ).squeeze(1).long()
        logits_idx = logits_idx[0].tolist()
        pse_labels = [self.token_list[x] for x in logits_idx]
        if output_format == "pse_labels":
            return pse_labels, None

        multi_labels = self.seq2arr(pse_labels, spk_num)[:, :spk_num]  # remove padding speakers
        multi_labels = self.smooth_multi_labels(multi_labels)
        if output_format == "binary_labels":
            return multi_labels, None

        spk_list = ["spk{}".format(i + 1) for i in range(spk_num)]
        spk_turns = self.calc_spk_turns(multi_labels, spk_list)
        results = OrderedDict()
        for spk, st, dur in spk_turns:
            if spk not in results:
                results[spk] = []
            if dur > self.dur_threshold:
                results[spk].append((st, st + dur))

        # sort segments in start time ascending
        for spk in results:
            results[spk] = sorted(results[spk], key=lambda x: x[0])

        return results, pse_labels

    @torch.no_grad()
    def __call__(
            self,
            speech: Union[torch.Tensor, np.ndarray],
            profile: Union[torch.Tensor, np.ndarray],
            output_format: str = "speaker_turn"
    ):
        """Inference

        Args:
            speech: Input speech data
            profile: Speaker profiles
        Returns:
            diarization results for each speaker

        """
        assert check_argument_types()
        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)
        if isinstance(profile, np.ndarray):
            profile = torch.tensor(profile)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        profile = profile.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        speech_lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        profile_lengths = profile.new_full([1], dtype=torch.long, fill_value=profile.size(1))
        batch = {"speech": speech, "speech_lengths": speech_lengths,
                 "profile": profile, "profile_lengths": profile_lengths}
        # a. To device
        batch = to_device(batch, device=self.device)

        logits = self.diar_model.prediction_forward(**batch)
        results, pse_labels = self.post_processing(logits, profile.shape[1], output_format)

        return results, pse_labels

    @staticmethod
    def from_pretrained(
            model_tag: Optional[str] = None,
            **kwargs: Optional[Any],
    ):
        """Build Speech2Xvector instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2Xvector: Speech2Xvector instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return Speech2DiarizationSOND(**kwargs)
