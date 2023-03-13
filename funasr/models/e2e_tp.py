import logging
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import numpy as np
from typeguard import check_argument_types

from funasr.models.encoder.abs_encoder import AbsEncoder
from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.predictor.cif import mae_loss
from funasr.modules.add_sos_eos import add_sos_eos
from funasr.modules.nets_utils import make_pad_mask, pad_list
from funasr.torch_utils.device_funcs import force_gatherable
from funasr.train.abs_espnet_model import AbsESPnetModel
from funasr.models.predictor.cif import CifPredictorV3


if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class TimestampPredictor(AbsESPnetModel):
    """
    Author: Speech Lab, Alibaba Group, China
    """

    def __init__(
            self,
            frontend: Optional[AbsFrontend],
            encoder: AbsEncoder,
            predictor: CifPredictorV3,
            predictor_bias: int = 0,
            token_list=None,
    ):
        assert check_argument_types()

        super().__init__()
        # note that eos is the same as sos (equivalent ID)

        self.frontend = frontend
        self.encoder = encoder
        self.encoder.interctc_use_conditioning = False

        self.predictor = predictor
        self.predictor_bias = predictor_bias
        self.criterion_pre = mae_loss()
        self.token_list = token_list
    
    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
                speech.shape[0]
                == speech_lengths.shape[0]
                == text.shape[0]
                == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]
        # for data-parallel
        text = text[:, : text_lengths.max()]
        speech = speech[:, :speech_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
            encoder_out.device)
        if self.predictor_bias == 1:
            _, text = add_sos_eos(text, 1, 2, -1)
            text_lengths = text_lengths + self.predictor_bias
        _, _, _, _, pre_token_length2 = self.predictor(encoder_out, text, encoder_out_mask, ignore_id=-1)

        # loss_pre = self.criterion_pre(ys_pad_lens.type_as(pre_token_length), pre_token_length)
        loss_pre = self.criterion_pre(text_lengths.type_as(pre_token_length2), pre_token_length2)

        loss = loss_pre
        stats = dict()

        # Collect Attn branch stats
        stats["loss_pre"] = loss_pre.detach().cpu() if loss_pre is not None else None
        stats["loss"] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
            self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        return encoder_out, encoder_out_lens
    
    def _extract_feats(
            self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]
        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def calc_predictor_timestamp(self, encoder_out, encoder_out_lens, token_num):
        encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
            encoder_out.device)
        ds_alphas, ds_cif_peak, us_alphas, us_cif_peak = self.predictor.get_upsample_timestamp(encoder_out,
                                                                                               encoder_out_mask,
                                                                                               token_num)
        return ds_alphas, ds_cif_peak, us_alphas, us_cif_peak

    def collect_feats(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
