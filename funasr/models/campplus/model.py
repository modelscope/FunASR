# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import time
import torch
import logging
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from typing import Union, Dict, List, Tuple, Optional

from funasr.utils.load_utils import load_audio_text_image_video
from funasr.utils.datadir_writer import DatadirWriter
from funasr.register import tables
from funasr.models.campplus.components import DenseLayer, StatsPool, TDNNLayer, CAMDenseTDNNBlock, TransitLayer, \
    BasicResBlock, get_nonlinear, FCM
from funasr.models.campplus.utils import extract_feature


@tables.register("model_classes", "CAMPPlus")
class CAMPPlus(nn.Module):
    def __init__(self,
                 feat_dim=80,
                 embedding_size=192,
                 growth_rate=32,
                 bn_size=4,
                 init_channels=128,
                 config_str='batchnorm-relu',
                 memory_efficient=True,
                 output_level='segment',
                 **kwargs,):
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.output_level = output_level

        self.xvector = nn.Sequential(
            OrderedDict([

                ('tdnn',
                 TDNNLayer(channels,
                           init_channels,
                           5,
                           stride=2,
                           dilation=1,
                           padding=-1,
                           config_str=config_str)),
            ]))
        channels = init_channels
        for i, (num_layers, kernel_size,
                dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            block = CAMDenseTDNNBlock(num_layers=num_layers,
                                      in_channels=channels,
                                      out_channels=growth_rate,
                                      bn_channels=bn_size * growth_rate,
                                      kernel_size=kernel_size,
                                      dilation=dilation,
                                      config_str=config_str,
                                      memory_efficient=memory_efficient)
            self.xvector.add_module('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                'transit%d' % (i + 1),
                TransitLayer(channels,
                             channels // 2,
                             bias=False,
                             config_str=config_str))
            channels //= 2

        self.xvector.add_module(
            'out_nonlinear', get_nonlinear(config_str, channels))

        if self.output_level == 'segment':
            self.xvector.add_module('stats', StatsPool())
            self.xvector.add_module(
                'dense',
                DenseLayer(
                    channels * 2, embedding_size, config_str='batchnorm_'))
        else:
            assert self.output_level == 'frame', '`output_level` should be set to \'segment\' or \'frame\'. '

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        if self.output_level == 'frame':
            x = x.transpose(1, 2)
        return x

    def generate(self,
                 data_in,
                 data_lengths=None,
                 key: list=None,
                 tokenizer=None,
                 frontend=None,
                 **kwargs,
                 ):
        # extract fbank feats
        meta_data = {}
        time1 = time.perf_counter()
        audio_sample_list = load_audio_text_image_video(data_in, fs=16000, audio_fs=kwargs.get("fs", 16000), data_type="sound")
        time2 = time.perf_counter()
        meta_data["load_data"] = f"{time2 - time1:0.3f}"
        speech, speech_lengths = extract_feature(audio_sample_list)
        time3 = time.perf_counter()
        meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
        meta_data["batch_data_time"] = np.array(speech_lengths).sum().item() / 16000.0
        # import pdb; pdb.set_trace()
        results = []
        embeddings = self.forward(speech)
        for embedding in embeddings:
            results.append({"spk_embedding":embedding})
        return results, meta_data