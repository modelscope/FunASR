from typing import List
from typing import Tuple
import logging
import torch
import torch.nn as nn
import numpy as np

from funasr.modules.streaming_utils import utils as myutils
from funasr.models.decoder.transformer_decoder import BaseTransformerDecoder

from funasr.modules.attention import MultiHeadedAttentionSANMDecoder, MultiHeadedAttentionCrossAtt
from funasr.modules.embedding import PositionalEncoding
from funasr.modules.layer_norm import LayerNorm
from funasr.modules.positionwise_feed_forward import PositionwiseFeedForwardDecoderSANM
from funasr.modules.repeat import repeat
from funasr.models.decoder.sanm_decoder import DecoderLayerSANM, ParaformerSANMDecoder


class ContextualDecoderLayer(nn.Module):
    def __init__(
        self,
        size,
        self_attn,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an DecoderLayer object."""
        super(ContextualDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        if self_attn is not None:
            self.norm2 = LayerNorm(size)
        if src_attn is not None:
            self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)

    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None,):
        # tgt = self.dropout(tgt)
        if isinstance(tgt, Tuple):
            tgt, _ = tgt
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)

        x = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        if self.training:
            cache = None
        x, cache = self.self_attn(tgt, tgt_mask, cache=cache)
        x = residual + self.dropout(x)
        x_self_attn = x

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = self.src_attn(x, memory, memory_mask)
        x_src_attn = x

        x = residual + self.dropout(x)
        return x, tgt_mask, x_self_attn, x_src_attn


class ContextualBiasDecoder(nn.Module):
    def __init__(
        self,
        size,
        src_attn,
        dropout_rate,
        normalize_before=True,
    ):
        """Construct an DecoderLayer object."""
        super(ContextualBiasDecoder, self).__init__()
        self.size = size
        self.src_attn = src_attn
        if src_attn is not None:
            self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before

    def forward(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):
        x = tgt
        if self.src_attn is not None:
            if self.normalize_before:
                x = self.norm3(x)
            x =  self.dropout(self.src_attn(x, memory, memory_mask))
        return x, tgt_mask, memory, memory_mask, cache


class ContextualParaformerDecoder(ParaformerSANMDecoder):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2006.01713
    """
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        att_layer_num: int = 6,
        kernel_size: int = 21,
        sanm_shfit: int = 0,
    ):
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        if input_layer == 'none':
            self.embed = None
        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                # pos_enc_class(attention_dim, positional_dropout_rate),
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
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = None

        self.att_layer_num = att_layer_num
        self.num_blocks = num_blocks
        if sanm_shfit is None:
            sanm_shfit = (kernel_size - 1) // 2
        self.decoders = repeat(
            att_layer_num - 1,
            lambda lnum: DecoderLayerSANM(
                attention_dim,
                MultiHeadedAttentionSANMDecoder(
                    attention_dim, self_attention_dropout_rate, kernel_size, sanm_shfit=sanm_shfit
                ),
                MultiHeadedAttentionCrossAtt(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForwardDecoderSANM(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.bias_decoder = ContextualBiasDecoder(
            size=attention_dim,
            src_attn=MultiHeadedAttentionCrossAtt(
                attention_heads, attention_dim, src_attention_dropout_rate
            ),
            dropout_rate=dropout_rate,
            normalize_before=True,
        )
        self.bias_output = torch.nn.Conv1d(attention_dim*2, attention_dim, 1, bias=False)
        self.last_decoder = ContextualDecoderLayer(
                attention_dim,
                MultiHeadedAttentionSANMDecoder(
                    attention_dim, self_attention_dropout_rate, kernel_size, sanm_shfit=sanm_shfit
                ),
                MultiHeadedAttentionCrossAtt(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForwardDecoderSANM(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            )
        if num_blocks - att_layer_num <= 0:
            self.decoders2 = None
        else:
            self.decoders2 = repeat(
                num_blocks - att_layer_num,
                lambda lnum: DecoderLayerSANM(
                    attention_dim,
                    MultiHeadedAttentionSANMDecoder(
                        attention_dim, self_attention_dropout_rate, kernel_size, sanm_shfit=0
                    ),
                    None,
                    PositionwiseFeedForwardDecoderSANM(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )

        self.decoders3 = repeat(
            1,
            lambda lnum: DecoderLayerSANM(
                attention_dim,
                None,
                None,
                PositionwiseFeedForwardDecoderSANM(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        contextual_info: torch.Tensor,
        clas_scale: float = 1.0,
        return_hidden: bool = False,
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
        tgt = ys_in_pad
        tgt_mask = myutils.sequence_mask(ys_in_lens, device=tgt.device)[:, :, None]

        memory = hs_pad
        memory_mask = myutils.sequence_mask(hlens, device=memory.device)[:, None, :]

        x = tgt
        x, tgt_mask, memory, memory_mask, _ = self.decoders(
            x, tgt_mask, memory, memory_mask
        )
        _, _, x_self_attn, x_src_attn = self.last_decoder(
            x, tgt_mask, memory, memory_mask
        )

        # contextual paraformer related
        contextual_length = torch.Tensor([contextual_info.shape[1]]).int().repeat(hs_pad.shape[0])
        contextual_mask = myutils.sequence_mask(contextual_length, device=memory.device)[:, None, :]
        cx, tgt_mask, _, _, _ = self.bias_decoder(x_self_attn, tgt_mask, contextual_info, memory_mask=contextual_mask)

        if self.bias_output is not None:
            x = torch.cat([x_src_attn, cx*clas_scale], dim=2)
            x = self.bias_output(x.transpose(1, 2)).transpose(1, 2)  # 2D -> D
            x = x_self_attn + self.dropout(x)

        if self.decoders2 is not None:
            x, tgt_mask, memory, memory_mask, _ = self.decoders2(
                x, tgt_mask, memory, memory_mask
            )

        x, tgt_mask, memory, memory_mask, _ = self.decoders3(
            x, tgt_mask, memory, memory_mask
        )
        if self.normalize_before:
            x = self.after_norm(x)
        olens = tgt_mask.sum(1)
        if self.output_layer is not None and return_hidden is False:
            x = self.output_layer(x)
        return x, olens

    def gen_tf2torch_map_dict(self):

        tensor_name_prefix_torch = self.tf2torch_tensor_name_prefix_torch
        tensor_name_prefix_tf = self.tf2torch_tensor_name_prefix_tf
        map_dict_local = {

            ## decoder
            # ffn
            "{}.decoders.layeridx.norm1.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/decoder_ffn/LayerNorm/gamma".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.decoders.layeridx.norm1.bias".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/decoder_ffn/LayerNorm/beta".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.decoders.layeridx.feed_forward.w_1.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/decoder_ffn/conv1d/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (1024,256),(1,256,1024)
            "{}.decoders.layeridx.feed_forward.w_1.bias".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/decoder_ffn/conv1d/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (1024,),(1024,)
            "{}.decoders.layeridx.feed_forward.norm.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/decoder_ffn/LayerNorm_1/gamma".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (1024,),(1024,)
            "{}.decoders.layeridx.feed_forward.norm.bias".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/decoder_ffn/LayerNorm_1/beta".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (1024,),(1024,)
            "{}.decoders.layeridx.feed_forward.w_2.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/decoder_ffn/conv1d_1/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (256,1024),(1,1024,256)

            # fsmn
            "{}.decoders.layeridx.norm2.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/decoder_memory_block/LayerNorm/gamma".format(
                    tensor_name_prefix_tf),
                    "squeeze": None,
                    "transpose": None,
                },  # (256,),(256,)
            "{}.decoders.layeridx.norm2.bias".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/decoder_memory_block/LayerNorm/beta".format(
                    tensor_name_prefix_tf),
                    "squeeze": None,
                    "transpose": None,
                },  # (256,),(256,)
            "{}.decoders.layeridx.self_attn.fsmn_block.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/decoder_memory_block/depth_conv_w".format(
                    tensor_name_prefix_tf),
                    "squeeze": 0,
                    "transpose": (1, 2, 0),
                },  # (256,1,31),(1,31,256,1)
            # src att
            "{}.decoders.layeridx.norm3.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/multi_head/LayerNorm/gamma".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.decoders.layeridx.norm3.bias".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/multi_head/LayerNorm/beta".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.decoders.layeridx.src_attn.linear_q.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/multi_head/conv1d/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (256,256),(1,256,256)
            "{}.decoders.layeridx.src_attn.linear_q.bias".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/multi_head/conv1d/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.decoders.layeridx.src_attn.linear_k_v.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/multi_head/conv1d_1/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (1024,256),(1,256,1024)
            "{}.decoders.layeridx.src_attn.linear_k_v.bias".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/multi_head/conv1d_1/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (1024,),(1024,)
            "{}.decoders.layeridx.src_attn.linear_out.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/multi_head/conv1d_2/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (256,256),(1,256,256)
            "{}.decoders.layeridx.src_attn.linear_out.bias".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_layeridx/multi_head/conv1d_2/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            # dnn
            "{}.decoders3.layeridx.norm1.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_dnn_layer_layeridx/LayerNorm/gamma".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.decoders3.layeridx.norm1.bias".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_dnn_layer_layeridx/LayerNorm/beta".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.decoders3.layeridx.feed_forward.w_1.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_dnn_layer_layeridx/conv1d/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (1024,256),(1,256,1024)
            "{}.decoders3.layeridx.feed_forward.w_1.bias".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_dnn_layer_layeridx/conv1d/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (1024,),(1024,)
            "{}.decoders3.layeridx.feed_forward.norm.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_dnn_layer_layeridx/LayerNorm_1/gamma".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (1024,),(1024,)
            "{}.decoders3.layeridx.feed_forward.norm.bias".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_dnn_layer_layeridx/LayerNorm_1/beta".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (1024,),(1024,)
            "{}.decoders3.layeridx.feed_forward.w_2.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_dnn_layer_layeridx/conv1d_1/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (256,1024),(1,1024,256)

            # embed_concat_ffn
            "{}.embed_concat_ffn.layeridx.norm1.weight".format(tensor_name_prefix_torch):
                {"name": "{}/cif_concat/LayerNorm/gamma".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.embed_concat_ffn.layeridx.norm1.bias".format(tensor_name_prefix_torch):
                {"name": "{}/cif_concat/LayerNorm/beta".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.embed_concat_ffn.layeridx.feed_forward.w_1.weight".format(tensor_name_prefix_torch):
                {"name": "{}/cif_concat/conv1d/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (1024,256),(1,256,1024)
            "{}.embed_concat_ffn.layeridx.feed_forward.w_1.bias".format(tensor_name_prefix_torch):
                {"name": "{}/cif_concat/conv1d/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (1024,),(1024,)
            "{}.embed_concat_ffn.layeridx.feed_forward.norm.weight".format(tensor_name_prefix_torch):
                {"name": "{}/cif_concat/LayerNorm_1/gamma".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (1024,),(1024,)
            "{}.embed_concat_ffn.layeridx.feed_forward.norm.bias".format(tensor_name_prefix_torch):
                {"name": "{}/cif_concat/LayerNorm_1/beta".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (1024,),(1024,)
            "{}.embed_concat_ffn.layeridx.feed_forward.w_2.weight".format(tensor_name_prefix_torch):
                {"name": "{}/cif_concat/conv1d_1/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (256,1024),(1,1024,256)

            # out norm
            "{}.after_norm.weight".format(tensor_name_prefix_torch):
                {"name": "{}/LayerNorm/gamma".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.after_norm.bias".format(tensor_name_prefix_torch):
                {"name": "{}/LayerNorm/beta".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)

            # in embed
            "{}.embed.0.weight".format(tensor_name_prefix_torch):
                {"name": "{}/w_embs".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (4235,256),(4235,256)

            # out layer
            "{}.output_layer.weight".format(tensor_name_prefix_torch):
                {"name": ["{}/dense/kernel".format(tensor_name_prefix_tf), "{}/w_embs".format(tensor_name_prefix_tf)],
                 "squeeze": [None, None],
                 "transpose": [(1, 0), None],
                 },  # (4235,256),(256,4235)
            "{}.output_layer.bias".format(tensor_name_prefix_torch):
                {"name": ["{}/dense/bias".format(tensor_name_prefix_tf),
                          "seq2seq/2bias" if tensor_name_prefix_tf == "seq2seq/decoder/inputter_1" else "seq2seq/bias"],
                 "squeeze": [None, None],
                 "transpose": [None, None],
                 },  # (4235,),(4235,)

            ## clas decoder
            # src att
            "{}.bias_decoder.norm3.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_15/multi_head_1/LayerNorm/gamma".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.bias_decoder.norm3.bias".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_15/multi_head_1/LayerNorm/beta".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.bias_decoder.src_attn.linear_q.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_15/multi_head_1/conv1d/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (256,256),(1,256,256)
            "{}.bias_decoder.src_attn.linear_q.bias".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_15/multi_head_1/conv1d/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.bias_decoder.src_attn.linear_k_v.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_15/multi_head_1/conv1d_1/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (1024,256),(1,256,1024)
            "{}.bias_decoder.src_attn.linear_k_v.bias".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_15/multi_head_1/conv1d_1/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (1024,),(1024,)
            "{}.bias_decoder.src_attn.linear_out.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_15/multi_head_1/conv1d_2/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (256,256),(1,256,256)
            "{}.bias_decoder.src_attn.linear_out.bias".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_15/multi_head_1/conv1d_2/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            # dnn
            "{}.bias_output.weight".format(tensor_name_prefix_torch):
                {"name": "{}/decoder_fsmn_layer_15/conv1d/kernel".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": (2, 1, 0),
                 },  # (1024,256),(1,256,1024)

        }
        return map_dict_local

    def convert_tf2torch(self,
                         var_dict_tf,
                         var_dict_torch,
                         ):
        map_dict = self.gen_tf2torch_map_dict()
        var_dict_torch_update = dict()
        decoder_layeridx_sets = set()
        for name in sorted(var_dict_torch.keys(), reverse=False):
            names = name.split('.')
            if names[0] == self.tf2torch_tensor_name_prefix_torch:
                if names[1] == "decoders":
                    layeridx = int(names[2])
                    name_q = name.replace(".{}.".format(layeridx), ".layeridx.")
                    layeridx_bias = 0
                    layeridx += layeridx_bias
                    decoder_layeridx_sets.add(layeridx)
                    if name_q in map_dict.keys():
                        name_v = map_dict[name_q]["name"]
                        name_tf = name_v.replace("layeridx", "{}".format(layeridx))
                        data_tf = var_dict_tf[name_tf]
                        if map_dict[name_q]["squeeze"] is not None:
                            data_tf = np.squeeze(data_tf, axis=map_dict[name_q]["squeeze"])
                        if map_dict[name_q]["transpose"] is not None:
                            data_tf = np.transpose(data_tf, map_dict[name_q]["transpose"])
                        data_tf = torch.from_numpy(data_tf).type(torch.float32).to("cpu")
                        assert var_dict_torch[name].size() == data_tf.size(), "{}, {}, {} != {}".format(name, name_tf,
                                                                                                        var_dict_torch[
                                                                                                            name].size(),
                                                                                                        data_tf.size())
                        var_dict_torch_update[name] = data_tf
                        logging.info(
                            "torch tensor: {}, {}, loading from tf tensor: {}, {}".format(name, data_tf.size(), name_v,
                                                                                          var_dict_tf[name_tf].shape))
                elif names[1] == "last_decoder":
                    layeridx = 15
                    name_q = name.replace("last_decoder", "decoders.layeridx")
                    layeridx_bias = 0
                    layeridx += layeridx_bias
                    decoder_layeridx_sets.add(layeridx)
                    if name_q in map_dict.keys():
                        name_v = map_dict[name_q]["name"]
                        name_tf = name_v.replace("layeridx", "{}".format(layeridx))
                        data_tf = var_dict_tf[name_tf]
                        if map_dict[name_q]["squeeze"] is not None:
                            data_tf = np.squeeze(data_tf, axis=map_dict[name_q]["squeeze"])
                        if map_dict[name_q]["transpose"] is not None:
                            data_tf = np.transpose(data_tf, map_dict[name_q]["transpose"])
                        data_tf = torch.from_numpy(data_tf).type(torch.float32).to("cpu")
                        assert var_dict_torch[name].size() == data_tf.size(), "{}, {}, {} != {}".format(name, name_tf,
                                                                                                        var_dict_torch[
                                                                                                            name].size(),
                                                                                                        data_tf.size())
                        var_dict_torch_update[name] = data_tf
                        logging.info(
                            "torch tensor: {}, {}, loading from tf tensor: {}, {}".format(name, data_tf.size(), name_v,
                                                                                          var_dict_tf[name_tf].shape))


                elif names[1] == "decoders2":
                    layeridx = int(names[2])
                    name_q = name.replace(".{}.".format(layeridx), ".layeridx.")
                    name_q = name_q.replace("decoders2", "decoders")
                    layeridx_bias = len(decoder_layeridx_sets)

                    layeridx += layeridx_bias
                    if "decoders." in name:
                        decoder_layeridx_sets.add(layeridx)
                    if name_q in map_dict.keys():
                        name_v = map_dict[name_q]["name"]
                        name_tf = name_v.replace("layeridx", "{}".format(layeridx))
                        data_tf = var_dict_tf[name_tf]
                        if map_dict[name_q]["squeeze"] is not None:
                            data_tf = np.squeeze(data_tf, axis=map_dict[name_q]["squeeze"])
                        if map_dict[name_q]["transpose"] is not None:
                            data_tf = np.transpose(data_tf, map_dict[name_q]["transpose"])
                        data_tf = torch.from_numpy(data_tf).type(torch.float32).to("cpu")
                        assert var_dict_torch[name].size() == data_tf.size(), "{}, {}, {} != {}".format(name, name_tf,
                                                                                                        var_dict_torch[
                                                                                                            name].size(),
                                                                                                        data_tf.size())
                        var_dict_torch_update[name] = data_tf
                        logging.info(
                            "torch tensor: {}, {}, loading from tf tensor: {}, {}".format(name, data_tf.size(), name_v,
                                                                                          var_dict_tf[name_tf].shape))

                elif names[1] == "decoders3":
                    layeridx = int(names[2])
                    name_q = name.replace(".{}.".format(layeridx), ".layeridx.")

                    layeridx_bias = 0
                    layeridx += layeridx_bias
                    if "decoders." in name:
                        decoder_layeridx_sets.add(layeridx)
                    if name_q in map_dict.keys():
                        name_v = map_dict[name_q]["name"]
                        name_tf = name_v.replace("layeridx", "{}".format(layeridx))
                        data_tf = var_dict_tf[name_tf]
                        if map_dict[name_q]["squeeze"] is not None:
                            data_tf = np.squeeze(data_tf, axis=map_dict[name_q]["squeeze"])
                        if map_dict[name_q]["transpose"] is not None:
                            data_tf = np.transpose(data_tf, map_dict[name_q]["transpose"])
                        data_tf = torch.from_numpy(data_tf).type(torch.float32).to("cpu")
                        assert var_dict_torch[name].size() == data_tf.size(), "{}, {}, {} != {}".format(name, name_tf,
                                                                                                        var_dict_torch[
                                                                                                            name].size(),
                                                                                                        data_tf.size())
                        var_dict_torch_update[name] = data_tf
                        logging.info(
                            "torch tensor: {}, {}, loading from tf tensor: {}, {}".format(name, data_tf.size(), name_v,
                                                                                          var_dict_tf[name_tf].shape))
                elif names[1] == "bias_decoder":
                    name_q = name

                    if name_q in map_dict.keys():
                        name_v = map_dict[name_q]["name"]
                        name_tf = name_v
                        data_tf = var_dict_tf[name_tf]
                        if map_dict[name_q]["squeeze"] is not None:
                            data_tf = np.squeeze(data_tf, axis=map_dict[name_q]["squeeze"])
                        if map_dict[name_q]["transpose"] is not None:
                            data_tf = np.transpose(data_tf, map_dict[name_q]["transpose"])
                        data_tf = torch.from_numpy(data_tf).type(torch.float32).to("cpu")
                        assert var_dict_torch[name].size() == data_tf.size(), "{}, {}, {} != {}".format(name, name_tf,
                                                                                                        var_dict_torch[
                                                                                                            name].size(),
                                                                                                        data_tf.size())
                        var_dict_torch_update[name] = data_tf
                        logging.info(
                            "torch tensor: {}, {}, loading from tf tensor: {}, {}".format(name, data_tf.size(), name_v,
                                                                                          var_dict_tf[name_tf].shape))


                elif names[1] == "embed" or names[1] == "output_layer" or names[1] == "bias_output":
                    name_tf = map_dict[name]["name"]
                    if isinstance(name_tf, list):
                        idx_list = 0
                        if name_tf[idx_list] in var_dict_tf.keys():
                            pass
                        else:
                            idx_list = 1
                        data_tf = var_dict_tf[name_tf[idx_list]]
                        if map_dict[name]["squeeze"][idx_list] is not None:
                            data_tf = np.squeeze(data_tf, axis=map_dict[name]["squeeze"][idx_list])
                        if map_dict[name]["transpose"][idx_list] is not None:
                            data_tf = np.transpose(data_tf, map_dict[name]["transpose"][idx_list])
                        data_tf = torch.from_numpy(data_tf).type(torch.float32).to("cpu")
                        assert var_dict_torch[name].size() == data_tf.size(), "{}, {}, {} != {}".format(name, name_tf,
                                                                                                        var_dict_torch[
                                                                                                            name].size(),
                                                                                                        data_tf.size())
                        var_dict_torch_update[name] = data_tf
                        logging.info("torch tensor: {}, {}, loading from tf tensor: {}, {}".format(name, data_tf.size(),
                                                                                                   name_tf[idx_list],
                                                                                                   var_dict_tf[name_tf[
                                                                                                       idx_list]].shape))

                    else:
                        data_tf = var_dict_tf[name_tf]
                        if map_dict[name]["squeeze"] is not None:
                            data_tf = np.squeeze(data_tf, axis=map_dict[name]["squeeze"])
                        if map_dict[name]["transpose"] is not None:
                            data_tf = np.transpose(data_tf, map_dict[name]["transpose"])
                        data_tf = torch.from_numpy(data_tf).type(torch.float32).to("cpu")
                        assert var_dict_torch[name].size() == data_tf.size(), "{}, {}, {} != {}".format(name, name_tf,
                                                                                                        var_dict_torch[
                                                                                                            name].size(),
                                                                                                        data_tf.size())
                        var_dict_torch_update[name] = data_tf
                        logging.info(
                            "torch tensor: {}, {}, loading from tf tensor: {}, {}".format(name, data_tf.size(), name_tf,
                                                                                          var_dict_tf[name_tf].shape))

                elif names[1] == "after_norm":
                    name_tf = map_dict[name]["name"]
                    data_tf = var_dict_tf[name_tf]
                    data_tf = torch.from_numpy(data_tf).type(torch.float32).to("cpu")
                    var_dict_torch_update[name] = data_tf
                    logging.info(
                        "torch tensor: {}, {}, loading from tf tensor: {}, {}".format(name, data_tf.size(), name_tf,
                                                                                      var_dict_tf[name_tf].shape))

                elif names[1] == "embed_concat_ffn":
                    layeridx = int(names[2])
                    name_q = name.replace(".{}.".format(layeridx), ".layeridx.")

                    layeridx_bias = 0
                    layeridx += layeridx_bias
                    if "decoders." in name:
                        decoder_layeridx_sets.add(layeridx)
                    if name_q in map_dict.keys():
                        name_v = map_dict[name_q]["name"]
                        name_tf = name_v.replace("layeridx", "{}".format(layeridx))
                        data_tf = var_dict_tf[name_tf]
                        if map_dict[name_q]["squeeze"] is not None:
                            data_tf = np.squeeze(data_tf, axis=map_dict[name_q]["squeeze"])
                        if map_dict[name_q]["transpose"] is not None:
                            data_tf = np.transpose(data_tf, map_dict[name_q]["transpose"])
                        data_tf = torch.from_numpy(data_tf).type(torch.float32).to("cpu")
                        assert var_dict_torch[name].size() == data_tf.size(), "{}, {}, {} != {}".format(name, name_tf,
                                                                                                        var_dict_torch[
                                                                                                            name].size(),
                                                                                                        data_tf.size())
                        var_dict_torch_update[name] = data_tf
                        logging.info(
                            "torch tensor: {}, {}, loading from tf tensor: {}, {}".format(name, data_tf.size(), name_v,
                                                                                          var_dict_tf[name_tf].shape))

        return var_dict_torch_update
