from typing import Any
from typing import List
from typing import Tuple

import torch
import torch.nn as nn

from funasr.modules.embedding import SinusoidalPositionEncoder
from funasr.punctuation.sanm_encoder import SANMVadEncoder as Encoder
from funasr.punctuation.abs_model import AbsPunctuation
from funasr.punctuation.sanm_encoder import SANMVadEncoder
from funasr.export.models.encoder.sanm_encoder import SANMVadEncoder as SANMVadEncoder_export

class VadRealtimeTransformer(AbsPunctuation):

    def __init__(
        self,
        model,
        max_seq_len=512,
        model_name='punc_model',
        **kwargs,
    ):
        super().__init__()
        onnx = False
        if "onnx" in kwargs:
            onnx = kwargs["onnx"]

        self.embed = model.embed
        if isinstance(model.encoder, SANMVadEncoder):
            self.encoder = SANMVadEncoder_export(model.encoder, onnx=onnx)
        else:
            assert False, "Only support samn encode."
        # self.encoder = model.encoder
        self.decoder = model.decoder



    def forward(self, input: torch.Tensor, text_lengths: torch.Tensor,
                vad_indexes: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Compute loss value from buffer sequences.

        Args:
            input (torch.Tensor): Input ids. (batch, len)
            hidden (torch.Tensor): Target ids. (batch, len)

        """
        x = self.embed(input)
        # mask = self._target_mask(input)
        h, _, _ = self.encoder(x, text_lengths, vad_indexes)
        y = self.decoder(h)
        return y

    def with_vad(self):
        return True

    def get_dummy_inputs(self):
        length = 120
        text_indexes = torch.randint(0, self.embed.num_embeddings, (1, length))
        text_lengths = torch.tensor([length], dtype=torch.int32)
        vad_mask = torch.ones(length, length)[None, None, :, :]
        return (text_indexes, text_lengths, vad_mask)

    def get_input_names(self):
        return ['input', 'text_lengths', 'vad_mask']

    def get_output_names(self):
        return ['logits']

    def get_dynamic_axes(self):
        return {
            'input': {
                1: 'feats_length'
            },
            'vad_mask': {
                2: 'feats_length1',
                3: 'feats_length2'
            },
            'logits': {
                1: 'logits_length'
            },
        }
