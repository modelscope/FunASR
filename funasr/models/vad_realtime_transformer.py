from typing import Any
from typing import List
from typing import Tuple

import torch
import torch.nn as nn

from funasr.modules.embedding import SinusoidalPositionEncoder
from funasr.models.encoder.sanm_encoder import SANMVadEncoder as Encoder
from funasr.train.abs_model import AbsPunctuation


class VadRealtimeTransformer(AbsPunctuation):
    """
    Author: Speech Lab, Alibaba Group, China
    CT-Transformer: Controllable time-delay transformer for real-time punctuation prediction and disfluency detection
    https://arxiv.org/pdf/2003.01309.pdf
    """
    def __init__(
        self,
        vocab_size: int,
        punc_size: int,
        pos_enc: str = None,
        embed_unit: int = 128,
        att_unit: int = 256,
        head: int = 2,
        unit: int = 1024,
        layer: int = 4,
        dropout_rate: float = 0.5,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
    ):
        super().__init__()
        if pos_enc == "sinusoidal":
            #            pos_enc_class = PositionalEncoding
            pos_enc_class = SinusoidalPositionEncoder
        elif pos_enc is None:

            def pos_enc_class(*args, **kwargs):
                return nn.Sequential()  # indentity

        else:
            raise ValueError(f"unknown pos-enc option: {pos_enc}")

        self.embed = nn.Embedding(vocab_size, embed_unit)
        self.encoder = Encoder(
            input_size=embed_unit,
            output_size=att_unit,
            attention_heads=head,
            linear_units=unit,
            num_blocks=layer,
            dropout_rate=dropout_rate,
            input_layer="pe",
            # pos_enc_class=pos_enc_class,
            padding_idx=0,
            kernel_size=kernel_size,
            sanm_shfit=sanm_shfit,
        )
        self.decoder = nn.Linear(att_unit, punc_size)


#    def _target_mask(self, ys_in_pad):
#        ys_mask = ys_in_pad != 0
#        m = subsequent_n_mask(ys_mask.size(-1), 5, device=ys_mask.device).unsqueeze(0)
#        return ys_mask.unsqueeze(-2) & m

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
        return y, None

    def with_vad(self):
        return True

    def score(self, y: torch.Tensor, state: Any, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Score new token.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                torch.float32 scores for next token (vocab_size)
                and next state for ys

        """
        y = y.unsqueeze(0)
        h, _, cache = self.encoder.forward_one_step(self.embed(y), self._target_mask(y), cache=state)
        h = self.decoder(h[:, -1])
        logp = h.log_softmax(dim=-1).squeeze(0)
        return logp, cache

    def batch_score(self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, vocab_size)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.encoder.encoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [torch.stack([states[b][i] for b in range(n_batch)]) for i in range(n_layers)]

        # batch decoding
        h, _, states = self.encoder.forward_one_step(self.embed(ys), self._target_mask(ys), cache=batch_state)
        h = self.decoder(h[:, -1])
        logp = h.log_softmax(dim=-1)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list
