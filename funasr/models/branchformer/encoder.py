# Copyright 2022 Yifan Peng (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Branchformer encoder definition.

Reference:
    Yifan Peng, Siddharth Dalmia, Ian Lane, and Shinji Watanabe,
    “Branchformer: Parallel MLP-Attention Architectures to Capture
    Local and Global Context for Speech Recognition and Understanding,”
    in Proceedings of ICML, 2022.

"""

import logging
from typing import List, Optional, Tuple, Union

import numpy
import torch
import torch.nn as nn

from funasr.models.branchformer.cgmlp import ConvolutionalGatingMLP
from funasr.models.branchformer.fastformer import FastSelfAttention
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.models.transformer.attention import (  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from funasr.models.transformer.embedding import (  # noqa: H301
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from funasr.models.transformer.layer_norm import LayerNorm
from funasr.models.transformer.utils.repeat import repeat
from funasr.models.transformer.utils.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)

from funasr.register import tables


class BranchformerEncoderLayer(torch.nn.Module):
    """Branchformer encoder layer module.

    Args:
        size (int): model dimension
        attn: standard self-attention or efficient attention, optional
        cgmlp: ConvolutionalGatingMLP, optional
        dropout_rate (float): dropout probability
        merge_method (str): concat, learned_ave, fixed_ave
        cgmlp_weight (float): weight of the cgmlp branch, between 0 and 1,
            used if merge_method is fixed_ave
        attn_branch_drop_rate (float): probability of dropping the attn branch,
            used if merge_method is learned_ave
        stochastic_depth_rate (float): stochastic depth probability
    """

    def __init__(
        self,
        size: int,
        attn: Optional[torch.nn.Module],
        cgmlp: Optional[torch.nn.Module],
        dropout_rate: float,
        merge_method: str,
        cgmlp_weight: float = 0.5,
        attn_branch_drop_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()
        assert (attn is not None) or (cgmlp is not None), "At least one branch should be valid"

        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp
        self.merge_method = merge_method
        self.cgmlp_weight = cgmlp_weight
        self.attn_branch_drop_rate = attn_branch_drop_rate
        self.stochastic_depth_rate = stochastic_depth_rate
        self.use_two_branches = (attn is not None) and (cgmlp is not None)

        if attn is not None:
            self.norm_mha = LayerNorm(size)  # for the MHA module
        if cgmlp is not None:
            self.norm_mlp = LayerNorm(size)  # for the MLP module
        self.norm_final = LayerNorm(size)  # for the final output of the block

        self.dropout = torch.nn.Dropout(dropout_rate)

        if self.use_two_branches:
            if merge_method == "concat":
                self.merge_proj = torch.nn.Linear(size + size, size)

            elif merge_method == "learned_ave":
                # attention-based pooling for two branches
                self.pooling_proj1 = torch.nn.Linear(size, 1)
                self.pooling_proj2 = torch.nn.Linear(size, 1)

                # linear projections for calculating merging weights
                self.weight_proj1 = torch.nn.Linear(size, 1)
                self.weight_proj2 = torch.nn.Linear(size, 1)

                # linear projection after weighted average
                self.merge_proj = torch.nn.Linear(size, size)

            elif merge_method == "fixed_ave":
                assert 0.0 <= cgmlp_weight <= 1.0, "cgmlp weight should be between 0.0 and 1.0"

                # remove the other branch if only one branch is used
                if cgmlp_weight == 0.0:
                    self.use_two_branches = False
                    self.cgmlp = None
                    self.norm_mlp = None
                elif cgmlp_weight == 1.0:
                    self.use_two_branches = False
                    self.attn = None
                    self.norm_mha = None

                # linear projection after weighted average
                self.merge_proj = torch.nn.Linear(size, size)

            else:
                raise ValueError(f"unknown merge method: {merge_method}")

        else:
            self.merge_proj = torch.nn.Identity()

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
        """

        if cache is not None:
            raise NotImplementedError("cache is not None, which is not tested")

        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

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
            if pos_emb is not None:
                return (x, pos_emb), mask
            return x, mask

        # Two branches
        x1 = x
        x2 = x

        # Branch 1: multi-headed attention module
        if self.attn is not None:
            x1 = self.norm_mha(x1)

            if isinstance(self.attn, FastSelfAttention):
                x_att = self.attn(x1, mask)
            else:
                if pos_emb is not None:
                    x_att = self.attn(x1, x1, x1, pos_emb, mask)
                else:
                    x_att = self.attn(x1, x1, x1, mask)

            x1 = self.dropout(x_att)

        # Branch 2: convolutional gating mlp
        if self.cgmlp is not None:
            x2 = self.norm_mlp(x2)

            if pos_emb is not None:
                x2 = (x2, pos_emb)
            x2 = self.cgmlp(x2, mask)
            if isinstance(x2, tuple):
                x2 = x2[0]

            x2 = self.dropout(x2)

        # Merge two branches
        if self.use_two_branches:
            if self.merge_method == "concat":
                x = x + stoch_layer_coeff * self.dropout(
                    self.merge_proj(torch.cat([x1, x2], dim=-1))
                )
            elif self.merge_method == "learned_ave":
                if (
                    self.training
                    and self.attn_branch_drop_rate > 0
                    and torch.rand(1).item() < self.attn_branch_drop_rate
                ):
                    # Drop the attn branch
                    w1, w2 = 0.0, 1.0
                else:
                    # branch1
                    score1 = (
                        self.pooling_proj1(x1).transpose(1, 2) / self.size**0.5
                    )  # (batch, 1, time)
                    if mask is not None:
                        min_value = float(
                            numpy.finfo(torch.tensor(0, dtype=score1.dtype).numpy().dtype).min
                        )
                        score1 = score1.masked_fill(mask.eq(0), min_value)
                        score1 = torch.softmax(score1, dim=-1).masked_fill(mask.eq(0), 0.0)
                    else:
                        score1 = torch.softmax(score1, dim=-1)
                    pooled1 = torch.matmul(score1, x1).squeeze(1)  # (batch, size)
                    weight1 = self.weight_proj1(pooled1)  # (batch, 1)

                    # branch2
                    score2 = (
                        self.pooling_proj2(x2).transpose(1, 2) / self.size**0.5
                    )  # (batch, 1, time)
                    if mask is not None:
                        min_value = float(
                            numpy.finfo(torch.tensor(0, dtype=score2.dtype).numpy().dtype).min
                        )
                        score2 = score2.masked_fill(mask.eq(0), min_value)
                        score2 = torch.softmax(score2, dim=-1).masked_fill(mask.eq(0), 0.0)
                    else:
                        score2 = torch.softmax(score2, dim=-1)
                    pooled2 = torch.matmul(score2, x2).squeeze(1)  # (batch, size)
                    weight2 = self.weight_proj2(pooled2)  # (batch, 1)

                    # normalize weights of two branches
                    merge_weights = torch.softmax(
                        torch.cat([weight1, weight2], dim=-1), dim=-1
                    )  # (batch, 2)
                    merge_weights = merge_weights.unsqueeze(-1).unsqueeze(-1)  # (batch, 2, 1, 1)
                    w1, w2 = merge_weights[:, 0], merge_weights[:, 1]  # (batch, 1, 1)

                x = x + stoch_layer_coeff * self.dropout(self.merge_proj(w1 * x1 + w2 * x2))
            elif self.merge_method == "fixed_ave":
                x = x + stoch_layer_coeff * self.dropout(
                    self.merge_proj((1.0 - self.cgmlp_weight) * x1 + self.cgmlp_weight * x2)
                )
            else:
                raise RuntimeError(f"unknown merge method: {self.merge_method}")
        else:
            if self.attn is None:
                x = x + stoch_layer_coeff * self.dropout(self.merge_proj(x2))
            elif self.cgmlp is None:
                x = x + stoch_layer_coeff * self.dropout(self.merge_proj(x1))
            else:
                # This should not happen
                raise RuntimeError("Both branches are not None, which is unexpected.")

        x = self.norm_final(x)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask


@tables.register("encoder_classes", "BranchformerEncoder")
class BranchformerEncoder(nn.Module):
    """Branchformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        use_attn: bool = True,
        attention_heads: int = 4,
        attention_layer_type: str = "rel_selfattn",
        pos_enc_layer_type: str = "rel_pos",
        rel_pos_type: str = "latest",
        use_cgmlp: bool = True,
        cgmlp_linear_units: int = 2048,
        cgmlp_conv_kernel: int = 31,
        use_linear_after_conv: bool = False,
        gate_activation: str = "identity",
        merge_method: str = "concat",
        cgmlp_weight: Union[float, List[float]] = 0.5,
        attn_branch_drop_rate: Union[float, List[float]] = 0.0,
        num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        zero_triu: bool = False,
        padding_idx: int = -1,
        stochastic_depth_rate: Union[float, List[float]] = 0.0,
    ):
        super().__init__()
        self._output_size = output_size

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if attention_layer_type == "rel_selfattn":
                attention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert attention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert attention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert attention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning("Using legacy_rel_pos and it will be deprecated in the future.")
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer is None:
            if input_size == output_size:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        if attention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif attention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
            logging.warning("Using legacy_rel_selfattn and it will be deprecated in the future.")
        elif attention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        elif attention_layer_type == "fast_selfattn":
            assert pos_enc_layer_type in ["abs_pos", "scaled_abs_pos"]
            encoder_selfattn_layer = FastSelfAttention
            encoder_selfattn_layer_args = (
                output_size,
                attention_heads,
                attention_dropout_rate,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + attention_layer_type)

        cgmlp_layer = ConvolutionalGatingMLP
        cgmlp_layer_args = (
            output_size,
            cgmlp_linear_units,
            cgmlp_conv_kernel,
            dropout_rate,
            use_linear_after_conv,
            gate_activation,
        )

        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks
        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        if isinstance(cgmlp_weight, float):
            cgmlp_weight = [cgmlp_weight] * num_blocks
        if len(cgmlp_weight) != num_blocks:
            raise ValueError(
                f"Length of cgmlp_weight ({len(cgmlp_weight)}) should be equal to "
                f"num_blocks ({num_blocks})"
            )

        if isinstance(attn_branch_drop_rate, float):
            attn_branch_drop_rate = [attn_branch_drop_rate] * num_blocks
        if len(attn_branch_drop_rate) != num_blocks:
            raise ValueError(
                f"Length of attn_branch_drop_rate ({len(attn_branch_drop_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: BranchformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args) if use_attn else None,
                cgmlp_layer(*cgmlp_layer_args) if use_cgmlp else None,
                dropout_rate,
                merge_method,
                cgmlp_weight[lnum],
                attn_branch_drop_rate[lnum],
                stochastic_depth_rate[lnum],
            ),
        )
        self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """

        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if (
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
        elif self.embed is not None:
            xs_pad = self.embed(xs_pad)

        xs_pad, masks = self.encoders(xs_pad, masks)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]

        xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, None
