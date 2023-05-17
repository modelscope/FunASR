# -*- encoding: utf-8 -*-
#!/usr/bin/env python3
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
from kaldiio import WriteHelper
from typeguard import check_argument_types
from typeguard import check_return_type

from funasr.utils.cli_utils import get_commandline_args
from funasr.tasks.sv import SVTask
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import config_argparse
from funasr.utils.types import str2bool
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_none
from funasr.utils.misc import statistic_model_parameters

class Speech2Xvector:
    """Speech2Xvector class

    Examples:
        >>> import soundfile
        >>> speech2xvector = Speech2Xvector("sv_config.yml", "sv.pb")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2xvector(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
            self,
            sv_train_config: Union[Path, str] = None,
            sv_model_file: Union[Path, str] = None,
            device: str = "cpu",
            batch_size: int = 1,
            dtype: str = "float32",
            streaming: bool = False,
            embedding_node: str = "resnet1_dense",
    ):
        assert check_argument_types()

        # TODO: 1. Build SV model
        sv_model, sv_train_args = SVTask.build_model_from_file(
            config_file=sv_train_config,
            model_file=sv_model_file,
            device=device
        )
        logging.info("sv_model: {}".format(sv_model))
        logging.info("model parameter number: {}".format(statistic_model_parameters(sv_model)))
        logging.info("sv_train_args: {}".format(sv_train_args))
        sv_model.to(dtype=getattr(torch, dtype)).eval()

        self.sv_model = sv_model
        self.sv_train_args = sv_train_args
        self.device = device
        self.dtype = dtype
        self.embedding_node = embedding_node

    @torch.no_grad()
    def calculate_embedding(self, speech: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, ilens = self.sv_model.encode(**batch)

        # c. Forward Pooling
        pooling = self.sv_model.pooling_layer(enc)

        # d. Forward Decoder
        outputs, embeddings = self.sv_model.decoder(pooling)

        if self.embedding_node not in embeddings:
            raise ValueError("Required embedding node {} not in {}".format(
                self.embedding_node, embeddings.keys()))

        return embeddings[self.embedding_node]

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray],
            ref_speech: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        """Inference

        Args:
            speech: Input speech data
            ref_speech: Reference speech to compare
        Returns:
            embedding, ref_embedding, similarity_score

        """
        assert check_argument_types()
        self.sv_model.eval()
        embedding = self.calculate_embedding(speech)
        ref_emb, score = None, None
        if ref_speech is not None:
            ref_emb = self.calculate_embedding(ref_speech)
            score = torch.cosine_similarity(embedding, ref_emb)

        results = (embedding, ref_emb, score)
        assert check_return_type(results)
        return results

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

        return Speech2Xvector(**kwargs)




