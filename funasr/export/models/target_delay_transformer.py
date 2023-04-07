from typing import Tuple

import torch
import torch.nn as nn

from funasr.models.encoder.sanm_encoder import SANMEncoder
from funasr.export.models.encoder.sanm_encoder import SANMEncoder as SANMEncoder_export
from funasr.models.encoder.sanm_encoder import SANMVadEncoder
from funasr.export.models.encoder.sanm_encoder import SANMVadEncoder as SANMVadEncoder_export

class CT_Transformer(nn.Module):

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
        self.decoder = model.decoder
        # self.model = model
        self.feats_dim = self.embed.embedding_dim
        self.num_embeddings = self.embed.num_embeddings
        self.model_name = model_name

        if isinstance(model.encoder, SANMEncoder):
            self.encoder = SANMEncoder_export(model.encoder, onnx=onnx)
        else:
            assert False, "Only support samn encode."

    def forward(self, inputs: torch.Tensor, text_lengths: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Compute loss value from buffer sequences.

        Args:
            input (torch.Tensor): Input ids. (batch, len)
            hidden (torch.Tensor): Target ids. (batch, len)

        """
        x = self.embed(inputs)
        # mask = self._target_mask(input)
        h, _ = self.encoder(x, text_lengths)
        y = self.decoder(h)
        return y

    def get_dummy_inputs(self):
        length = 120
        text_indexes = torch.randint(0, self.embed.num_embeddings, (2, length))
        text_lengths = torch.tensor([length-20, length], dtype=torch.int32)
        return (text_indexes, text_lengths)

    def get_input_names(self):
        return ['inputs', 'text_lengths']

    def get_output_names(self):
        return ['logits']

    def get_dynamic_axes(self):
        return {
            'inputs': {
                0: 'batch_size',
                1: 'feats_length'
            },
            'text_lengths': {
                0: 'batch_size',
            },
            'logits': {
                0: 'batch_size',
                1: 'logits_length'
            },
        }


class CT_Transformer_VadRealtime(nn.Module):

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
        self.decoder = model.decoder
        self.model_name = model_name



    def forward(self, inputs: torch.Tensor,
                text_lengths: torch.Tensor,
                vad_indexes: torch.Tensor,
                sub_masks: torch.Tensor,
                ) -> Tuple[torch.Tensor, None]:
        """Compute loss value from buffer sequences.

        Args:
            input (torch.Tensor): Input ids. (batch, len)
            hidden (torch.Tensor): Target ids. (batch, len)

        """
        x = self.embed(inputs)
        # mask = self._target_mask(input)
        h, _ = self.encoder(x, text_lengths, vad_indexes, sub_masks)
        y = self.decoder(h)
        return y

    def with_vad(self):
        return True

    def get_dummy_inputs(self):
        length = 120
        text_indexes = torch.randint(0, self.embed.num_embeddings, (1, length))
        text_lengths = torch.tensor([length], dtype=torch.int32)
        vad_mask = torch.ones(length, length, dtype=torch.float32)[None, None, :, :]
        sub_masks = torch.ones(length, length, dtype=torch.float32)
        sub_masks = torch.tril(sub_masks).type(torch.float32)
        return (text_indexes, text_lengths, vad_mask, sub_masks[None, None, :, :])

    def get_input_names(self):
        return ['inputs', 'text_lengths', 'vad_masks', 'sub_masks']

    def get_output_names(self):
        return ['logits']

    def get_dynamic_axes(self):
        return {
            'inputs': {
                1: 'feats_length'
            },
            'vad_masks': {
                2: 'feats_length1',
                3: 'feats_length2'
            },
            'sub_masks': {
                2: 'feats_length1',
                3: 'feats_length2'
            },
            'logits': {
                1: 'logits_length'
            },
        }
