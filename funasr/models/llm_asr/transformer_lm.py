from typing import Any
from typing import List
from typing import Tuple

import torch
import torch.nn as nn

from funasr.models.transformer.embedding import PositionalEncoding, ScaledPositionalEncoding, RelPositionalEncoding, LegacyRelPositionalEncoding
from funasr.models.llm_asr.transformer_encoder import TransformerEncoder_s0 as Encoder
from funasr.models.transformer.utils.mask import subsequent_mask
from funasr.models.transformer.utils.nets_utils import make_pad_mask
import logging
from distutils.version import LooseVersion
from contextlib import contextmanager
if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class TransformerEmbedLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pos_enc: str = None,
        embed_unit: int = 128,
        att_unit: int = 256,
        head: int = 2,
        unit: int = 1024,
        layer: int = 4,
        dropout_rate: float = 0.5,
        attention_dropout_rate: float = 0.0,
        pe_type: str = "split",
        bidirectional_inputs: bool = False,
        text_vocab_size: int = 4000,
        input_aug_conf: dict = None,
        output_aug_conf: dict = None,
        codec_groups: int = 4,
        selfattention_layer_type: str = "selfattn",
        input_normalize: bool = False,
        use_decoder: bool = True,
        encoder_type: str = "transformer",
        **kwargs
    ):
        super().__init__()
        if pos_enc == "sinusoidal":
            pos_enc_class = PositionalEncoding
        elif pos_enc == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc == "legacy_rel_pos":
            assert selfattention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        elif pos_enc is None:

            def pos_enc_class(*args, **kwargs):
                return nn.Sequential()  # indentity

        else:
            raise ValueError(f"unknown pos-enc option: {pos_enc}")

        self.embed_unit = embed_unit
        self.pe_type = pe_type
        self.encoder_type = encoder_type
        if encoder_type == "llama":
            raise NotImplementedError("llama encoder has not been implemented")
            # from cosyvoice.nets.encoder.llama_encoder import LlamaEncoder
            # # set causal to false, using mask to control causal mode.
            # self.encoder = LlamaEncoder(
            #     input_size=embed_unit,
            #     output_size=att_unit,
            #     attention_heads=head,
            #     num_blocks=layer,
            #     dropout_rate=dropout_rate,
            #     attention_dropout_rate=attention_dropout_rate,
            #     causal=False,
            #     linear_units=unit,
            # )
        else:
            self.encoder = Encoder(
                idim=embed_unit,
                attention_dim=att_unit,
                attention_heads=head,
                linear_units=unit,
                num_blocks=layer,
                dropout_rate=dropout_rate,
                positional_dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                input_layer="none" if pe_type == "split" else "linear",
                pos_enc_class=pos_enc_class,
                selfattention_layer_type=selfattention_layer_type,
            )
        if use_decoder:
            self.decoder = nn.Linear(att_unit, vocab_size)
        else:
            self.decoder = None
        self.attn_unit = att_unit
        self.pos_enc_func = None
        if pe_type == "split":
            assert pos_enc == "sinusoidal" or pos_enc == "abs_pos" or pos_enc == "scaled_abs_pos", \
                "Different positional embedding for inputs and outputs " \
                "only supports sinusoidal, abs_pos and scaled_abs_pos."
            self.pos_enc_func = pos_enc_class(embed_unit, 0.1)
            self.input_layer = torch.nn.Linear(embed_unit, att_unit)
        self.bidirectional_inputs = bidirectional_inputs
        self.text_vocab_size = text_vocab_size
        self.codec_groups = codec_groups
        self.input_aug = None
        if input_aug_conf is not None:
            from funasr.models.specaug.specaug import SpecAug
            self.input_aug = SpecAug(**input_aug_conf)

        self.output_aug = None
        if output_aug_conf is not None:
            from funasr.models.specaug.specaug import SpecAug
            self.output_aug = SpecAug(**output_aug_conf)

        self.normalize = None
        if input_normalize:
            from funasr.models.normalize.utterance_mvn import UtteranceMVN
            self.normalize = UtteranceMVN()

        self.first_pack_mask_conf: dict = kwargs.get("first_pack_mask_conf", None)

    def output_size(self):
        return self.attn_unit

    def _target_mask(self, lengths):
        ys_mask = ~make_pad_mask(lengths)
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    def clac_first_package_mask(self, mask, input_lengths, cond_lengths):
        device = mask.device
        mask_type = self.first_pack_mask_conf.get("mask_type", "first_pack")
        fp_token_len = self.first_pack_mask_conf["fp_token_len"]
        fp_text_len = self.first_pack_mask_conf["fp_text_len"]
        # NOTE: fp_text_len excluding sos, xvec, only including text
        # NOTE: cond_lengths including sos, xvec and text
        if mask_type == "streaming":
            for i, (seq_len, cond_len) in enumerate(zip(input_lengths, cond_lengths)):
                # 1 for task_id
                token_len = seq_len - cond_len - 1
                if token_len > 0:
                    target_text_len = torch.ceil(torch.arange(1, token_len+1, device=device) / fp_token_len) * fp_text_len
                    # 2 for sos and xvec, M -> M x 1
                    target_text_len = torch.minimum(target_text_len + 2, cond_len).unsqueeze(1)
                    # 1 x N
                    pos_range = torch.arange(0, cond_len, device=device).unsqueeze(0)
                    # M x N
                    text_mask = pos_range < target_text_len
                    # 1 for <task_id>
                    mask[i, cond_len+1:seq_len, :cond_len] = mask[i, cond_len+1:seq_len, :cond_len] * text_mask
        else:
            for i, (seq_len, cond_len) in enumerate(zip(input_lengths, cond_lengths)):
                mask_token_end = min(cond_len+1+fp_token_len, seq_len)
                mask[i, cond_len+1:mask_token_end, fp_text_len+2:cond_len] = 0

        return mask

    def forward(
            self,
            input: torch.Tensor,
            input_lengths: torch.Tensor,
            cond_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute LM loss value from buffer sequences.

        Args:
            input (torch.Tensor): Input ids. (batch, len, dim)
            input_lengths (torch.Tensor): length of input. (batch,)
            cond_lengths (torch.Tensor): length of conditions (including sos, excluding taskid). (batch,)

        """
        mask = self._target_mask(input_lengths).to(input.device)
        if self.first_pack_mask_conf is not None:
            mask = self.clac_first_package_mask(mask, input_lengths, cond_lengths)
        if self.bidirectional_inputs:
            for i, length in enumerate(cond_lengths):
                mask[i, :length, :length] = True
        pos_emb = None
        if self.pe_type == "split":
            pos_emb = torch.zeros((input.shape[0], input.shape[1]*2-1, self.attn_unit)).to(input)
        kk = self.codec_groups
        # with torch.no_grad():
        with autocast(False):
            for i, length in enumerate(cond_lengths):
                # perform specaug for each frame including multi-group.
                raw_feat = input[i:i + 1, 1:length].clone()
                bb, tt, dd = raw_feat.shape
                raw_feat = raw_feat.reshape(bb, tt // kk, kk, dd).reshape(bb, tt // kk, kk * dd)

                if self.input_aug is not None and self.training:
                    raw_feat = self.input_aug(raw_feat, (cond_lengths[i:i+1] - 1) // kk)[0]

                if self.normalize is not None:
                    raw_feat = self.normalize(raw_feat, None)[0]

                input[i:i + 1, 1:length] = raw_feat.reshape(bb, tt//kk, kk, dd).reshape(bb, tt, dd)

                if self.output_aug is not None and self.training:
                    raw_feat = input[i:i + 1, length+1:].clone()
                    aug_feat = self.output_aug(raw_feat, input_lengths[i:i+1] - length - 2)[0]
                    input[i:i + 1, length + 1:] = aug_feat

                # add positional encoding
                if self.pe_type == "split" and self.pos_enc_func is not None:
                    posed_input = self.pos_enc_func(input[i:i + 1, :length].clone())
                    if isinstance(posed_input, tuple):
                        pos_emb[i:i+1, :length*2-1] = posed_input[1]
                        posed_input = posed_input[0]
                    input[i:i + 1, :length] = posed_input

                    posed_output = self.pos_enc_func(input[i:i + 1, length + 1:].clone())
                    if isinstance(posed_output, tuple):
                        pos_emb[i:i+1, length*2: length*2+posed_output[1].shape[1]] = posed_output[1]
                        posed_output = posed_output[0]
                    input[i:i + 1, length + 1:] = posed_output

        if self.pe_type == "split":
            input = self.input_layer(input)
            if isinstance(self.pos_enc_func, (RelPositionalEncoding, LegacyRelPositionalEncoding)):
                input = (input, pos_emb)
        # logging.info(f"shapes {input.shape} {mask.shape} {input_lengths}")
        h, _ = self.encoder(input, mask)
        if self.decoder is None:
            return h, h

        y = self.decoder(h)
        return y, h

    def init_state(self, x: torch.Tensor):
        return None

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token.

        Args:
            y (torch.Tensor): 2D torch.float prefix embeddings.
            state: Scorer state for prefix tokens
            x (torch.Tensor): encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                torch.float32 scores for next token (vocab_size)
                and next state for ys

        """
        # this implementation is much faster than the blow!!
        mask = torch.tril(torch.ones((1, y.shape[0], y.shape[0]), device=y.device)).to(torch.bool)
        y_emb = y.unsqueeze(0).to(x.device)
        # lengths = y_emb.new_full([1], dtype=torch.long, fill_value=y_emb.size(1))
        # mask = self._target_mask(lengths).to(y_emb.device)
        # x includes <sos>, feat, <task_id>
        input_length = x.shape[0] - 1
        if self.bidirectional_inputs:
            mask[:1, :input_length, :input_length] = True
        # if self.first_pack_mask_conf is not None:
        #     mask = self.clac_first_package_mask(
        #         mask,
        #         torch.tensor([y.shape[0]], device=y.device),
        #         torch.tensor([input_length], device=y.device),
        #     )
        if self.pe_type == "split" and self.pos_enc_func is not None:
            pos_emb = torch.zeros((y_emb.shape[0], y_emb.shape[1], self.attn_unit)).to(y_emb)

            posed_input = self.pos_enc_func(y_emb[:1, :input_length])
            if isinstance(posed_input, tuple):
                pos_emb[:1, :input_length] = posed_input[1]
                posed_input = posed_input[0]
            y_emb[:1, :input_length] = posed_input

            posed_output = self.pos_enc_func(y_emb[:1, input_length + 1:])
            if isinstance(posed_output, tuple):
                pos_emb[:1, input_length + 1:] = posed_output[1]
                posed_output = posed_output[0]
            y_emb[:1, input_length + 1:] = posed_output

        if self.pe_type == "split":
            y_emb = self.input_layer(y_emb)
            if isinstance(self.pos_enc_func, (RelPositionalEncoding, LegacyRelPositionalEncoding)):
                y_emb = (y_emb, pos_emb)
        lm_hidden_states, _, cache = self.encoder.forward_one_step(
            y_emb, mask, cache=state
        )
        if self.decoder is None:
            return lm_hidden_states[:, -1], cache

        h = self.decoder(lm_hidden_states[:, -1])[:, :self.text_vocab_size]

        logp = h.log_softmax(dim=-1).squeeze(0)
        # return logp, cache
        return logp, (cache, lm_hidden_states[:, -1])

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
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        h, _, states = self.encoder.forward_one_step(
            self.embed(ys), self._target_mask(ys), cache=batch_state
        )
        h = self.decoder(h[:, -1])
        logp = h.log_softmax(dim=-1)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list
