# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
from typing import Any, List, Tuple

import torch
from torch import nn
import whisper

from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.register import tables


@tables.register("decoder_classes", "OpenAIWhisperDecoderWarp")
class OpenAIWhisperDecoderWarp(nn.Module):
    """Transformer-based Speech-to-Text Decoder from OpenAI's Whisper Model:

    URL: https://github.com/openai/whisper
    """

    def __init__(
        self,
        dropout_rate: float = 0.0,
        whisper_model: str = "small",
        download_dir: str = None,
        use_padmask: bool = False,
    ):
        super().__init__()

        assert whisper_model in whisper.available_models()
        _model = whisper.load_model(whisper_model, download_root=download_dir, device="cpu")
        self.decoders = copy.deepcopy(_model.decoder)
        attention_dim = self.decoders.token_embedding.embedding_dim

        # note that originally Whisper doesn't use dropouts
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.decoders.train()
        del _model
        self.use_padmask = use_padmask

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        tgt, memory = ys_in_pad, hs_pad
        tgt = self.decoders.token_embedding(tgt) + self.decoders.positional_embedding[: tgt.size(1)]
        tgt = self.dropout(tgt)

        x = tgt.to(memory.dtype)

        if self.use_padmask:
            memory_mask = (~make_pad_mask(hlens)[:, None, :]).to(memory.device)
        else:
            memory_mask = None

        for layer, block in enumerate(self.decoders.blocks):
            x = block(
                x,
                memory,
                mask=self.decoders.mask,
                memory_mask=memory_mask,
                is_pad_mask=False,
                is_pad_memory_mask=True,
            )

            if layer < len(self.decoders.blocks) - 1:
                x = self.dropout(x)

        x = self.decoders.ln(x)
        x = (x @ torch.transpose(self.decoders.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return x, ys_in_lens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        NOTE (Shih-Lun):
            cache implementation is ignored for now
            for simplicity & correctness
        """
        x = self.decoders.token_embedding(tgt) + self.decoders.positional_embedding[: tgt.size(1)]
        x = self.dropout(x)
        x = x.to(memory.dtype)

        for layer, block in enumerate(self.decoders.blocks):
            x = block(x, memory, mask=self.decoders.mask)
            if layer < len(self.decoders.blocks) - 1:
                x = self.dropout(x)

        x = self.decoders.ln(x)
        y = x[:, -1]
        y = (y @ torch.transpose(self.decoders.token_embedding.weight.to(x.dtype), 0, 1)).float()
        y = torch.log_softmax(y, dim=-1)

        return y, None

    def score(self, ys, state, x):
        """Score."""
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), torch.empty(0), x.unsqueeze(0), cache=state  # dummy mask
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # batch decoding, dummy mask is passed
        logp, states = self.forward_one_step(ys, torch.empty(0), xs, cache=None)

        return logp, None
