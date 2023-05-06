from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple

import torch
from typeguard import check_argument_types

from funasr.modules.nets_utils import make_pad_mask
from funasr.modules.attention import MultiHeadedAttention
from funasr.modules.attention import CosineDistanceAttention
from funasr.models.decoder.transformer_decoder import DecoderLayer
from funasr.models.decoder.decoder_layer_sa_asr import SpeakerAttributeAsrDecoderFirstLayer
from funasr.models.decoder.decoder_layer_sa_asr import SpeakerAttributeSpkDecoderFirstLayer
from funasr.modules.dynamic_conv import DynamicConvolution
from funasr.modules.dynamic_conv2d import DynamicConvolution2D
from funasr.modules.embedding import PositionalEncoding
from funasr.modules.layer_norm import LayerNorm
from funasr.modules.lightconv import LightweightConvolution
from funasr.modules.lightconv2d import LightweightConvolution2D
from funasr.modules.mask import subsequent_mask
from funasr.modules.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from funasr.modules.repeat import repeat
from funasr.modules.scorers.scorer_interface import BatchScorerInterface
from funasr.models.decoder.abs_decoder import AbsDecoder

class BaseSAAsrTransformerDecoder(AbsDecoder, BatchScorerInterface):
    
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        spker_embedding_dim: int = 256,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        input_layer: str = "embed",
        use_asr_output_layer: bool = True,
        use_spk_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {input_layer}")

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_asr_output_layer:
            self.asr_output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.asr_output_layer = None

        if use_spk_output_layer:
            self.spk_output_layer = torch.nn.Linear(attention_dim, spker_embedding_dim)
        else:
            self.spk_output_layer = None

        self.cos_distance_att = CosineDistanceAttention()

        self.decoder1 = None
        self.decoder2 = None
        self.decoder3 = None
        self.decoder4 = None

    def forward(
        self,
        asr_hs_pad: torch.Tensor,
        spk_hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        profile: torch.Tensor,
        profile_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        asr_memory = asr_hs_pad
        spk_memory = spk_hs_pad
        memory_mask = (~make_pad_mask(hlens))[:, None, :].to(asr_memory.device)
        # Spk decoder
        x = self.embed(tgt)

        x, tgt_mask, asr_memory, spk_memory, memory_mask, z = self.decoder1(
            x, tgt_mask, asr_memory, spk_memory, memory_mask
        )
        x, tgt_mask, spk_memory, memory_mask = self.decoder2(
            x, tgt_mask, spk_memory, memory_mask
        )
        if self.normalize_before:
            x = self.after_norm(x)
        if self.spk_output_layer is not None:
            x = self.spk_output_layer(x)
        dn, weights = self.cos_distance_att(x, profile, profile_lens)
        # Asr decoder
        x, tgt_mask, asr_memory, memory_mask = self.decoder3(
            z, tgt_mask, asr_memory, memory_mask, dn
        )
        x, tgt_mask, asr_memory, memory_mask = self.decoder4(
            x, tgt_mask, asr_memory, memory_mask
        )

        if self.normalize_before:
            x = self.after_norm(x)
        if self.asr_output_layer is not None:
            x = self.asr_output_layer(x)

        olens = tgt_mask.sum(1)
        return x, weights, olens


    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        asr_memory: torch.Tensor,
        spk_memory: torch.Tensor,
        profile: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        
        x = self.embed(tgt)

        if cache is None:
            cache = [None] * (2 + len(self.decoder2) + len(self.decoder4))
        new_cache = []
        x, tgt_mask, asr_memory, spk_memory, _, z = self.decoder1(
                x, tgt_mask, asr_memory, spk_memory, None, cache=cache[0]
        )
        new_cache.append(x)
        for c, decoder in zip(cache[1: len(self.decoder2) + 1], self.decoder2):
            x, tgt_mask, spk_memory, _ = decoder(
                x, tgt_mask, spk_memory, None, cache=c
            )
            new_cache.append(x)
        if self.normalize_before:
            x = self.after_norm(x)
        else:
            x = x
        if self.spk_output_layer is not None:
            x = self.spk_output_layer(x)
        dn, weights = self.cos_distance_att(x, profile, None)

        x, tgt_mask, asr_memory, _ = self.decoder3(
            z, tgt_mask, asr_memory, None, dn, cache=cache[len(self.decoder2) + 1]
        )
        new_cache.append(x)

        for c, decoder in zip(cache[len(self.decoder2) + 2: ], self.decoder4):
            x, tgt_mask, asr_memory, _ = decoder(
                x, tgt_mask, asr_memory, None, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.asr_output_layer is not None:
            y = torch.log_softmax(self.asr_output_layer(y), dim=-1)

        return y, weights, new_cache

    def score(self, ys, state, asr_enc, spk_enc, profile):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=ys.device).unsqueeze(0)
        logp, weights, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, asr_enc.unsqueeze(0), spk_enc.unsqueeze(0), profile.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), weights.squeeze(), state

class SAAsrTransformerDecoder(BaseSAAsrTransformerDecoder):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        spker_embedding_dim: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        asr_num_blocks: int = 6,
        spk_num_blocks: int = 3,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_asr_output_layer: bool = True,
        use_spk_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        assert check_argument_types()
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            spker_embedding_dim=spker_embedding_dim,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_asr_output_layer=use_asr_output_layer,
            use_spk_output_layer=use_spk_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size

        self.decoder1 = SpeakerAttributeSpkDecoderFirstLayer(
            attention_dim,
            MultiHeadedAttention(
                attention_heads, attention_dim, self_attention_dropout_rate
            ),
            MultiHeadedAttention(
                attention_heads, attention_dim, src_attention_dropout_rate
            ),
            PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
            dropout_rate,
            normalize_before,
            concat_after,
        )
        self.decoder2 = repeat(
            spk_num_blocks - 1,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        
        
        self.decoder3 = SpeakerAttributeAsrDecoderFirstLayer(
            attention_dim,
            spker_embedding_dim,
            MultiHeadedAttention(
                attention_heads, attention_dim, src_attention_dropout_rate
            ),
            PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
            dropout_rate,
            normalize_before,
            concat_after,
        )
        self.decoder4 = repeat(
            asr_num_blocks - 1,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
