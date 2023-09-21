from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from funasr.modules.streaming_utils.chunk_utilis import overlap_chunk
import numpy as np
from funasr.torch_utils.device_funcs import to_device
from funasr.modules.nets_utils import make_pad_mask
from funasr.modules.attention import MultiHeadedAttention, MultiHeadedAttentionSANM, MultiHeadedAttentionSANMwithMask
from funasr.modules.embedding import SinusoidalPositionEncoder, StreamSinusoidalPositionEncoder
from funasr.modules.layer_norm import LayerNorm
from funasr.modules.multi_layer_conv import Conv1dLinear
from funasr.modules.multi_layer_conv import MultiLayeredConv1d
from funasr.modules.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from funasr.modules.repeat import repeat
from funasr.modules.subsampling import Conv2dSubsampling
from funasr.modules.subsampling import Conv2dSubsampling2
from funasr.modules.subsampling import Conv2dSubsampling6
from funasr.modules.subsampling import Conv2dSubsampling8
from funasr.modules.subsampling import TooShortUttError
from funasr.modules.subsampling import check_short_utt
from funasr.modules.mask import subsequent_mask, vad_mask

from funasr.models.ctc import CTC
from funasr.models.encoder.abs_encoder import AbsEncoder

class EncoderLayerSANM(nn.Module):
    def __init__(
        self,
        in_size,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayerSANM, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate

    def forward(self, x, mask, cache=None, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x, mask, mask_shfit_chunk=mask_shfit_chunk, mask_att_chunk_encoder=mask_att_chunk_encoder)), dim=-1)
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
            else:
                x = stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.dropout(
                    self.self_attn(x, mask, mask_shfit_chunk=mask_shfit_chunk, mask_att_chunk_encoder=mask_att_chunk_encoder)
                )
            else:
                x = stoch_layer_coeff * self.dropout(
                    self.self_attn(x, mask, mask_shfit_chunk=mask_shfit_chunk, mask_att_chunk_encoder=mask_att_chunk_encoder)
                )
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.in_size == self.size:
            attn, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)
            x = residual + attn
        else:
            x, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)

        return x, cache


class SANMEncoder(AbsEncoder):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    San-m: Memory equipped self-attention for end-to-end speech recognition
    https://arxiv.org/abs/2006.01713

    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=SinusoidalPositionEncoder,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        kernel_size : int = 11,
        sanm_shfit : int = 0,
        lora_list: List[str] = None,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        selfattention_layer_type: str = "sanm",
        tf2torch_tensor_name_prefix_torch: str = "encoder",
        tf2torch_tensor_name_prefix_tf: str = "seq2seq/encoder",
    ):
        super().__init__()
        self._output_size = output_size

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(input_size, output_size, dropout_rate)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                SinusoidalPositionEncoder(),
            )
        elif input_layer is None:
            if input_size == output_size:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        elif input_layer == "pe":
            self.embed = SinusoidalPositionEncoder()
        elif input_layer == "pe_online":
            self.embed = StreamSinusoidalPositionEncoder()
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )

        elif selfattention_layer_type == "sanm":
            encoder_selfattn_layer = MultiHeadedAttentionSANM
            encoder_selfattn_layer_args0 = (
                attention_heads,
                input_size,
                output_size,
                attention_dropout_rate,
                kernel_size,
                sanm_shfit,
                lora_list,
                lora_rank,
                lora_alpha,
                lora_dropout,
            )

            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                output_size,
                attention_dropout_rate,
                kernel_size,
                sanm_shfit,
                lora_list,
                lora_rank,
                lora_alpha,
                lora_dropout,
            )
        self.encoders0 = repeat(
            1,
            lambda lnum: EncoderLayerSANM(
                input_size,
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

        self.encoders = repeat(
            num_blocks-1,
            lambda lnum: EncoderLayerSANM(
                output_size,
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None
        self.dropout = nn.Dropout(dropout_rate)
        self.tf2torch_tensor_name_prefix_torch = tf2torch_tensor_name_prefix_torch
        self.tf2torch_tensor_name_prefix_tf = tf2torch_tensor_name_prefix_tf

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        xs_pad = xs_pad * self.output_size()**0.5
        if self.embed is None:
            xs_pad = xs_pad
        elif (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        # xs_pad = self.dropout(xs_pad)
        encoder_outs = self.encoders0(xs_pad, masks)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]
        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            encoder_outs = self.encoders(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                encoder_outs = encoder_layer(xs_pad, masks)
                xs_pad, masks = encoder_outs[0], encoder_outs[1]

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)
                        xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None

    def _add_overlap_chunk(self, feats: np.ndarray, cache: dict = {}):
        if len(cache) == 0:
            return feats
        cache["feats"] = to_device(cache["feats"], device=feats.device)
        overlap_feats = torch.cat((cache["feats"], feats), dim=1)
        cache["feats"] = overlap_feats[:, -(cache["chunk_size"][0] + cache["chunk_size"][2]):, :]
        return overlap_feats

    def forward_chunk(self,
                      xs_pad: torch.Tensor,
                      ilens: torch.Tensor,
                      cache: dict = None,
                      ctc: CTC = None,
                      ):
        xs_pad *= self.output_size() ** 0.5
        if self.embed is None:
            xs_pad = xs_pad
        else:
            xs_pad = self.embed(xs_pad, cache)
        if cache["tail_chunk"]:
            xs_pad = to_device(cache["feats"], device=xs_pad.device)
        else:
            xs_pad = self._add_overlap_chunk(xs_pad, cache)
        encoder_outs = self.encoders0(xs_pad, None, None, None, None)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]
        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            encoder_outs = self.encoders(xs_pad, None, None, None, None)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                encoder_outs = encoder_layer(xs_pad, None, None, None, None)
                xs_pad, masks = encoder_outs[0], encoder_outs[1]
                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)
                        xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), None, None
        return xs_pad, ilens, None

    def gen_tf2torch_map_dict(self):
        tensor_name_prefix_torch = self.tf2torch_tensor_name_prefix_torch
        tensor_name_prefix_tf = self.tf2torch_tensor_name_prefix_tf
        map_dict_local = {
            ## encoder
            # cicd
            "{}.encoders.layeridx.norm1.weight".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/multi_head/LayerNorm/gamma".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.encoders.layeridx.norm1.bias".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/multi_head/LayerNorm/beta".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.encoders.layeridx.self_attn.linear_q_k_v.weight".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/multi_head/conv1d/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (768,256),(1,256,768)
            "{}.encoders.layeridx.self_attn.linear_q_k_v.bias".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/multi_head/conv1d/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (768,),(768,)
            "{}.encoders.layeridx.self_attn.fsmn_block.weight".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/multi_head/depth_conv_w".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 2, 0),
                 },  # (256,1,31),(1,31,256,1)
            "{}.encoders.layeridx.self_attn.linear_out.weight".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/multi_head/conv1d_1/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (256,256),(1,256,256)
            "{}.encoders.layeridx.self_attn.linear_out.bias".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/multi_head/conv1d_1/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            # ffn
            "{}.encoders.layeridx.norm2.weight".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/ffn/LayerNorm/gamma".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.encoders.layeridx.norm2.bias".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/ffn/LayerNorm/beta".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.encoders.layeridx.feed_forward.w_1.weight".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/ffn/conv1d/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (1024,256),(1,256,1024)
            "{}.encoders.layeridx.feed_forward.w_1.bias".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/ffn/conv1d/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (1024,),(1024,)
            "{}.encoders.layeridx.feed_forward.w_2.weight".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/ffn/conv1d_1/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (256,1024),(1,1024,256)
            "{}.encoders.layeridx.feed_forward.w_2.bias".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/ffn/conv1d_1/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
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
        
        }
    
        return map_dict_local

    def convert_tf2torch(self,
                         var_dict_tf,
                         var_dict_torch,
                         ):
        
        map_dict = self.gen_tf2torch_map_dict()
    
        var_dict_torch_update = dict()
        for name in sorted(var_dict_torch.keys(), reverse=False):
            names = name.split('.')
            if names[0] == self.tf2torch_tensor_name_prefix_torch:
                if names[1] == "encoders0":
                    layeridx = int(names[2])
                    name_q = name.replace(".{}.".format(layeridx), ".layeridx.")
                
                    name_q = name_q.replace("encoders0", "encoders")
                    layeridx_bias = 0
                    layeridx += layeridx_bias
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
                elif names[1] == "encoders":
                    layeridx = int(names[2])
                    name_q = name.replace(".{}.".format(layeridx), ".layeridx.")
                    layeridx_bias = 1
                    layeridx += layeridx_bias
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
            
                elif names[1] == "after_norm":
                    name_tf = map_dict[name]["name"]
                    data_tf = var_dict_tf[name_tf]
                    data_tf = torch.from_numpy(data_tf).type(torch.float32).to("cpu")
                    var_dict_torch_update[name] = data_tf
                    logging.info(
                        "torch tensor: {}, {}, loading from tf tensor: {}, {}".format(name, data_tf.size(), name_tf,
                                                                                      var_dict_tf[name_tf].shape))
    
        return var_dict_torch_update


class SANMEncoderChunkOpt(AbsEncoder):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    SCAMA: Streaming chunk-aware multihead attention for online end-to-end speech recognition
    https://arxiv.org/abs/2006.01713

    """

    def __init__(
            self,
            input_size: int,
            output_size: int = 256,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 6,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.0,
            input_layer: Optional[str] = "conv2d",
            pos_enc_class=SinusoidalPositionEncoder,
            normalize_before: bool = True,
            concat_after: bool = False,
            positionwise_layer_type: str = "linear",
            positionwise_conv_kernel_size: int = 1,
            padding_idx: int = -1,
            interctc_layer_idx: List[int] = [],
            interctc_use_conditioning: bool = False,
            kernel_size: int = 11,
            sanm_shfit: int = 0,
            selfattention_layer_type: str = "sanm",
            chunk_size: Union[int, Sequence[int]] = (16,),
            stride: Union[int, Sequence[int]] = (10,),
            pad_left: Union[int, Sequence[int]] = (0,),
            encoder_att_look_back_factor: Union[int, Sequence[int]] = (1,),
            decoder_att_look_back_factor: Union[int, Sequence[int]] = (1,),
            tf2torch_tensor_name_prefix_torch: str = "encoder",
            tf2torch_tensor_name_prefix_tf: str = "seq2seq/encoder",
    ):
        super().__init__()
        self._output_size = output_size

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(input_size, output_size, dropout_rate)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer is None:
            if input_size == output_size:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        elif input_layer == "pe":
            self.embed = SinusoidalPositionEncoder()
        elif input_layer == "pe_online":
            self.embed = StreamSinusoidalPositionEncoder()
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "sanm":
            encoder_selfattn_layer = MultiHeadedAttentionSANM
            encoder_selfattn_layer_args0 = (
                attention_heads,
                input_size,
                output_size,
                attention_dropout_rate,
                kernel_size,
                sanm_shfit,
            )

            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                output_size,
                attention_dropout_rate,
                kernel_size,
                sanm_shfit,
            )
        self.encoders0 = repeat(
            1,
            lambda lnum: EncoderLayerSANM(
                input_size,
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

        self.encoders = repeat(
            num_blocks - 1,
            lambda lnum: EncoderLayerSANM(
                output_size,
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None
        shfit_fsmn = (kernel_size - 1) // 2
        self.overlap_chunk_cls = overlap_chunk(
            chunk_size=chunk_size,
            stride=stride,
            pad_left=pad_left,
            shfit_fsmn=shfit_fsmn,
            encoder_att_look_back_factor=encoder_att_look_back_factor,
            decoder_att_look_back_factor=decoder_att_look_back_factor,
        )
        self.tf2torch_tensor_name_prefix_torch = tf2torch_tensor_name_prefix_torch
        self.tf2torch_tensor_name_prefix_tf = tf2torch_tensor_name_prefix_tf

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs_pad: torch.Tensor,
            ilens: torch.Tensor,
            prev_states: torch.Tensor = None,
            ctc: CTC = None,
            ind: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        xs_pad *= self.output_size() ** 0.5
        if self.embed is None:
            xs_pad = xs_pad
        elif (
                isinstance(self.embed, Conv2dSubsampling)
                or isinstance(self.embed, Conv2dSubsampling2)
                or isinstance(self.embed, Conv2dSubsampling6)
                or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        mask_shfit_chunk, mask_att_chunk_encoder = None, None
        if self.overlap_chunk_cls is not None:
            ilens = masks.squeeze(1).sum(1)
            chunk_outs = self.overlap_chunk_cls.gen_chunk_mask(ilens, ind)
            xs_pad, ilens = self.overlap_chunk_cls.split_chunk(xs_pad, ilens, chunk_outs=chunk_outs)
            masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
            mask_shfit_chunk = self.overlap_chunk_cls.get_mask_shfit_chunk(chunk_outs, xs_pad.device, xs_pad.size(0),
                                                                           dtype=xs_pad.dtype)
            mask_att_chunk_encoder = self.overlap_chunk_cls.get_mask_att_chunk_encoder(chunk_outs, xs_pad.device,
                                                                                       xs_pad.size(0),
                                                                                       dtype=xs_pad.dtype)

        encoder_outs = self.encoders0(xs_pad, masks, None, mask_shfit_chunk, mask_att_chunk_encoder)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]
        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            encoder_outs = self.encoders(xs_pad, masks, None, mask_shfit_chunk, mask_att_chunk_encoder)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                encoder_outs = encoder_layer(xs_pad, masks, None, mask_shfit_chunk, mask_att_chunk_encoder)
                xs_pad, masks = encoder_outs[0], encoder_outs[1]
                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)
                        xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None

    def _add_overlap_chunk(self, feats: np.ndarray, cache: dict = {}):
        if len(cache) == 0:
            return feats
        cache["feats"] = to_device(cache["feats"], device=feats.device)
        overlap_feats = torch.cat((cache["feats"], feats), dim=1)
        cache["feats"] = overlap_feats[:, -(cache["chunk_size"][0] + cache["chunk_size"][2]):, :]
        return overlap_feats

    def forward_chunk(self,
                      xs_pad: torch.Tensor,
                      ilens: torch.Tensor,
                      cache: dict = None,
                      ):
        xs_pad *= self.output_size() ** 0.5
        if self.embed is None:
            xs_pad = xs_pad
        else:
            xs_pad = self.embed(xs_pad, cache)
        if cache["tail_chunk"]:
            xs_pad = to_device(cache["feats"], device=xs_pad.device)
        else:
            xs_pad = self._add_overlap_chunk(xs_pad, cache)
        if cache["opt"] is None:
            cache_layer_num = len(self.encoders0) + len(self.encoders)
            new_cache = [None] * cache_layer_num
        else:
            new_cache = cache["opt"]

        for layer_idx, encoder_layer in enumerate(self.encoders0):
            encoder_outs = encoder_layer.forward_chunk(xs_pad, new_cache[layer_idx], cache["chunk_size"], cache["encoder_chunk_look_back"])
            xs_pad, new_cache[0] = encoder_outs[0], encoder_outs[1]

        for layer_idx, encoder_layer in enumerate(self.encoders):
            encoder_outs = encoder_layer.forward_chunk(xs_pad, new_cache[layer_idx+len(self.encoders0)], cache["chunk_size"], cache["encoder_chunk_look_back"])
            xs_pad, new_cache[layer_idx+len(self.encoders0)] = encoder_outs[0], encoder_outs[1]

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)
        if cache["encoder_chunk_look_back"] > 0 or cache["encoder_chunk_look_back"] == -1:
            cache["opt"] = new_cache

        return xs_pad, ilens, None

    def gen_tf2torch_map_dict(self):
        tensor_name_prefix_torch = self.tf2torch_tensor_name_prefix_torch
        tensor_name_prefix_tf = self.tf2torch_tensor_name_prefix_tf
        map_dict_local = {
            ## encoder
            # cicd
            "{}.encoders.layeridx.norm1.weight".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/multi_head/LayerNorm/gamma".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.encoders.layeridx.norm1.bias".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/multi_head/LayerNorm/beta".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.encoders.layeridx.self_attn.linear_q_k_v.weight".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/multi_head/conv1d/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (768,256),(1,256,768)
            "{}.encoders.layeridx.self_attn.linear_q_k_v.bias".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/multi_head/conv1d/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (768,),(768,)
            "{}.encoders.layeridx.self_attn.fsmn_block.weight".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/multi_head/depth_conv_w".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 2, 0),
                 },  # (256,1,31),(1,31,256,1)
            "{}.encoders.layeridx.self_attn.linear_out.weight".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/multi_head/conv1d_1/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (256,256),(1,256,256)
            "{}.encoders.layeridx.self_attn.linear_out.bias".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/multi_head/conv1d_1/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            # ffn
            "{}.encoders.layeridx.norm2.weight".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/ffn/LayerNorm/gamma".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.encoders.layeridx.norm2.bias".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/ffn/LayerNorm/beta".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.encoders.layeridx.feed_forward.w_1.weight".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/ffn/conv1d/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (1024,256),(1,256,1024)
            "{}.encoders.layeridx.feed_forward.w_1.bias".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/ffn/conv1d/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (1024,),(1024,)
            "{}.encoders.layeridx.feed_forward.w_2.weight".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/ffn/conv1d_1/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (256,1024),(1,1024,256)
            "{}.encoders.layeridx.feed_forward.w_2.bias".format(tensor_name_prefix_torch):
                {"name": "{}/layer_layeridx/ffn/conv1d_1/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
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
        
        }
    
        return map_dict_local

    def convert_tf2torch(self,
                         var_dict_tf,
                         var_dict_torch,
                         ):
    
        map_dict = self.gen_tf2torch_map_dict()
    
        var_dict_torch_update = dict()
        for name in sorted(var_dict_torch.keys(), reverse=False):
            names = name.split('.')
            if names[0] == self.tf2torch_tensor_name_prefix_torch:
                if names[1] == "encoders0":
                    layeridx = int(names[2])
                    name_q = name.replace(".{}.".format(layeridx), ".layeridx.")
                
                    name_q = name_q.replace("encoders0", "encoders")
                    layeridx_bias = 0
                    layeridx += layeridx_bias
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
                elif names[1] == "encoders":
                    layeridx = int(names[2])
                    name_q = name.replace(".{}.".format(layeridx), ".layeridx.")
                    layeridx_bias = 1
                    layeridx += layeridx_bias
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
            
                elif names[1] == "after_norm":
                    name_tf = map_dict[name]["name"]
                    data_tf = var_dict_tf[name_tf]
                    data_tf = torch.from_numpy(data_tf).type(torch.float32).to("cpu")
                    var_dict_torch_update[name] = data_tf
                    logging.info(
                        "torch tensor: {}, {}, loading from tf tensor: {}, {}".format(name, data_tf.size(), name_tf,
                                                                                      var_dict_tf[name_tf].shape))
    
        return var_dict_torch_update


class SANMVadEncoder(AbsEncoder):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group

    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=SinusoidalPositionEncoder,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        kernel_size : int = 11,
        sanm_shfit : int = 0,
        selfattention_layer_type: str = "sanm",
    ):
        super().__init__()
        self._output_size = output_size

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(input_size, output_size, dropout_rate)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                SinusoidalPositionEncoder(),
            )
        elif input_layer is None:
            if input_size == output_size:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        elif input_layer == "pe":
            self.embed = SinusoidalPositionEncoder()
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )

        elif selfattention_layer_type == "sanm":
            self.encoder_selfattn_layer = MultiHeadedAttentionSANMwithMask
            encoder_selfattn_layer_args0 = (
                attention_heads,
                input_size,
                output_size,
                attention_dropout_rate,
                kernel_size,
                sanm_shfit,
            )

            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                output_size,
                attention_dropout_rate,
                kernel_size,
                sanm_shfit,
            )

        self.encoders0 = repeat(
            1,
            lambda lnum: EncoderLayerSANM(
                input_size,
                output_size,
                self.encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

        self.encoders = repeat(
            num_blocks-1,
            lambda lnum: EncoderLayerSANM(
                output_size,
                output_size,
                self.encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None
        self.dropout = nn.Dropout(dropout_rate)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        vad_indexes: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        sub_masks = subsequent_mask(masks.size(-1), device=xs_pad.device).unsqueeze(0)
        no_future_masks = masks & sub_masks
        xs_pad *= self.output_size()**0.5
        if self.embed is None:
            xs_pad = xs_pad
        elif (isinstance(self.embed, Conv2dSubsampling) or isinstance(self.embed, Conv2dSubsampling2)
              or isinstance(self.embed, Conv2dSubsampling6) or isinstance(self.embed, Conv2dSubsampling8)):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling " +
                    f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        # xs_pad = self.dropout(xs_pad)
        mask_tup0 = [masks, no_future_masks]
        encoder_outs = self.encoders0(xs_pad, mask_tup0)
        xs_pad, _ = encoder_outs[0], encoder_outs[1]
        intermediate_outs = []


        for layer_idx, encoder_layer in enumerate(self.encoders):
                if layer_idx + 1 == len(self.encoders):
                    # This is last layer.
                    coner_mask = torch.ones(masks.size(0),
                                            masks.size(-1),
                                            masks.size(-1),
                                            device=xs_pad.device,
                                            dtype=torch.bool)
                    for word_index, length in enumerate(ilens):
                        coner_mask[word_index, :, :] = vad_mask(masks.size(-1),
                                                                vad_indexes[word_index],
                                                                device=xs_pad.device)
                    layer_mask = masks & coner_mask
                else:
                    layer_mask = no_future_masks
                mask_tup1 = [masks, layer_mask]
                encoder_outs = encoder_layer(xs_pad, mask_tup1)
                xs_pad, layer_mask = encoder_outs[0], encoder_outs[1]

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None
