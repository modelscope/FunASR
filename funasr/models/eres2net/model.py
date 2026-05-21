#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
# Modified from 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)

import os
import time
import logging
import torch
import numpy as np

from funasr.register import tables
from funasr.models.campplus.utils import extract_feature
from funasr.utils.load_utils import load_audio_text_image_video
from funasr.models.eres2net.eres2netv2 import ERes2NetV2


@tables.register("model_classes", "ERes2NetV2")
@tables.register("model_classes", "iic/speech_eres2netv2_sv_zh-cn_16k-common")
class ERes2NetV2SV(torch.nn.Module):
    """ERes2NetV2: Enhanced Res2Net v2 for Speaker Verification.

    Improved speaker embedding model based on Res2Net architecture with
    multi-scale feature aggregation. Provides 192-dim speaker embeddings
    for speaker verification and diarization.

    Better than CAM++ for short-duration audio (< 3s) speaker feature extraction.

    Output: {"spk_embedding": Tensor of shape (1, 192)}
    """

    def __init__(
        self,
        feat_dim=80,
        embedding_size=192,
        m_channels=64,
        baseWidth=26,
        scale=2,
        expansion=2,
        num_blocks=[3, 4, 6, 3],
        pooling_func="TSTP",
        two_emb_layer=False,
        **kwargs,
    ):
        """Initialize ERes2NetV2SV.
        
            Args:
                feat_dim: Size/dimension parameter.
                embedding_size: Size/dimension parameter.
                m_channels: TODO.
                baseWidth: TODO.
                scale: TODO.
                expansion: TODO.
                num_blocks: TODO.
                pooling_func: TODO.
                two_emb_layer: TODO.
                **kwargs: Additional keyword arguments.
            """
        super().__init__()
        self.model = ERes2NetV2(
            feat_dim=feat_dim,
            embedding_size=embedding_size,
            m_channels=m_channels,
            baseWidth=baseWidth,
            scale=scale,
            expansion=expansion,
            num_blocks=num_blocks,
            pooling_func=pooling_func,
            two_emb_layer=two_emb_layer,
        )
        self.embedding_size = embedding_size

        model_path = kwargs.get("model_path", None)
        init_param = kwargs.get("init_param", None)
        if init_param is None and model_path is not None:
            ckpt = os.path.join(model_path, "pretrained_eres2netv2.ckpt")
            if os.path.exists(ckpt):
                init_param = ckpt
        if init_param is not None and os.path.exists(init_param):
            self._load_pretrained(init_param)

    def _load_pretrained(self, path):
        """Internal: load pretrained.
        
            Args:
                path: TODO.
            """
        state_dict = torch.load(path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logging.warning(f"ERes2NetV2 missing keys: {missing[:5]}...")
        logging.info(f"ERes2NetV2 loaded pretrained weights from {path}")

    def forward(self, x):
        """Forward pass for training.
        
            Args:
                x: TODO.
            """
        return self.model(x)

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        """Run inference on input data.
        
            Args:
                data_in: Input data (audio samples, file paths, or text).
                data_lengths: Lengths of each input sample in the batch.
                key: Sample identifiers.
                tokenizer: Tokenizer instance for text encoding/decoding.
                frontend: Audio frontend for feature extraction.
                **kwargs: Additional keyword arguments.
            """
        meta_data = {}
        time1 = time.perf_counter()
        audio_sample_list = load_audio_text_image_video(
            data_in, fs=16000, audio_fs=kwargs.get("fs", 16000), data_type="sound"
        )
        time2 = time.perf_counter()
        meta_data["load_data"] = f"{time2 - time1:0.3f}"
        speech, speech_lengths, speech_times = extract_feature(audio_sample_list)
        speech = speech.to(device=kwargs["device"])
        time3 = time.perf_counter()
        meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
        meta_data["batch_data_time"] = np.array(speech_times).sum().item() / 16000.0
        results = [{"spk_embedding": self.forward(speech.to(torch.float32))}]
        return results, meta_data
