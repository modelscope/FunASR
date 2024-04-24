#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import logging

import torch

from funasr.models.transformer.model import Transformer
from funasr.register import tables


@tables.register("model_classes", "SANM")
class SANM(Transformer):
    """
    Author: Zhifu Gao, Shiliang Zhang, Ming Lei, Ian McLoughlin
    San-m: Memory equipped self-attention for end-to-end speech recognition
    https://arxiv.org/abs/2006.01713
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
