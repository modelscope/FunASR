import logging
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from funasr.models.e2e_asr_common import ErrorCalculator
from funasr.modules.nets_utils import th_accuracy
from funasr.modules.add_sos_eos import add_sos_eos
from funasr.losses.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from funasr.models.ctc import CTC
from funasr.models.decoder.abs_decoder import AbsDecoder
from funasr.models.encoder.abs_encoder import AbsEncoder
from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.postencoder.abs_postencoder import AbsPostEncoder
from funasr.models.preencoder.abs_preencoder import AbsPreEncoder
from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.layers.abs_normalize import AbsNormalize
from funasr.torch_utils.device_funcs import force_gatherable
from funasr.train.abs_espnet_model import AbsESPnetModel
from funasr.modules.streaming_utils.chunk_utilis import sequence_mask
from funasr.models.predictor.cif import mae_loss
from funasr.export.utils.torch_function import sequence_mask
from funasr.models.encoder.sanm_encoder import SANMEncoder
from funasr.models.encoder.conformer_encoder import ConformerEncoder
from funasr.export.models.encoder.sanm_encoder import SANMEncoder as SANMEncoder_export
from funasr.export.models.encoder.conformer_encoder import ConformerEncoder as ConformerEncoder_export
from funasr.models.predictor.cif import CifPredictorV2
from funasr.export.models.predictor.cif import CifPredictorV2 as CifPredictorV2_export
from funasr.export.utils.torch_function import MakePadMask
from funasr.models.decoder.sanm_decoder import FsmnDecoderSCAMAOpt
from funasr.export.models.decoder.sanm_decoder import FsmnDecoderSCAMAOpt as FsmnDecoderSCAMAOpt_export



import torch
import torch.nn as nn

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

class UniASR(nn.Module):
    def __init__(
            self,
            model,
            max_seq_len=512,
            feats_dim=560,
            model_name='model',
            **kwargs,
    ):
        super().__init__()
        onnx = True
        if "onnx" in kwargs:
            onnx = kwargs["onnx"]
        self.feats_dim = feats_dim
        self.model_name = model_name
        # if isinstance(model.encoder, SANMEncoder):
        #     self.encoder = SANMEncoder_export(model.encoder, onnx=onnx)
        # elif isinstance(model.encoder, ConformerEncoder):
        #     self.encoder = ConformerEncoder_export(model.encoder, onnx=onnx)
        self.encoder = SANMEncoder_export(model.encoder, onnx=onnx)

        if isinstance(model.predictor, CifPredictorV2):
            self.predictor = CifPredictorV2_export(model.predictor)
        if onnx:
            self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        else:
            self.make_pad_mask = sequence_mask(max_seq_len, flip=False)
        
        if isinstance(model.decoder, FsmnDecoderSCAMAOpt):
            self.decoder = FsmnDecoderSCAMAOpt_export(model.decoder, onnx=onnx)

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
    ):
        # a. To device
        batch = {"speech": speech, "speech_lengths": speech_lengths}
        enc, enc_len = self.encoder(**batch)

        mask = self.make_pad_mask(enc_len)[:, None, :]
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(enc, mask)
        pre_token_length = pre_token_length.floor().type(torch.int32)
        decoder_out, _ = self.decoder(enc, enc_len, pre_acoustic_embeds, pre_token_length)
        decoder_out = torch.log_softmax(decoder_out, dim=-1)

        return decoder_out, pre_token_length
      

    def get_dummy_inputs(self):
        speech = torch.randn(2, 30, self.feats_dim)
        speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
        return (speech, speech_lengths)

    def get_dummy_inputs_txt(self, txt_file: str = "/mnt/workspace/data_fbank/0207/12345.wav.fea.txt"):
        import numpy as np
        fbank = np.loadtxt(txt_file)
        fbank_lengths = np.array([fbank.shape[0], ], dtype=np.int32)
        speech = torch.from_numpy(fbank[None, :, :].astype(np.float32))
        speech_lengths = torch.from_numpy(fbank_lengths.astype(np.int32))
        return (speech, speech_lengths)

    def get_input_names(self):
        return ['speech', 'speech_lengths']

    def get_output_names(self):
        return ['logits', 'token_num']

    def get_dynamic_axes(self):
        return {
            'speech': {
                0: 'batch_size',
                1: 'feats_length'
            },
            'speech_lengths': {
                0: 'batch_size',
            },
            'logits': {
                0: 'batch_size',
                1: 'logits_length'
            },
        }