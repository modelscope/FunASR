# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from funasr.models.data2vec.data_utils import compute_mask_indices
from funasr.models.data2vec.ema_module import EMAModule
from funasr.models.data2vec.grad_multiply import GradMultiply
from funasr.models.data2vec.wav2vec2 import (
    ConvFeatureExtractionModel,
    TransformerEncoder,
)
from funasr.models.transformer.utils.nets_utils import make_pad_mask


def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


class Data2VecEncoder(nn.Module):
    def __init__(
        self,
        # for ConvFeatureExtractionModel
        input_size: int = None,
        extractor_mode: str = None,
        conv_feature_layers: str = "[(512,2,2)] + [(512,2,2)]",
        # for Transformer Encoder
        ## model architecture
        layer_type: str = "transformer",
        layer_norm_first: bool = False,
        encoder_layers: int = 12,
        encoder_embed_dim: int = 768,
        encoder_ffn_embed_dim: int = 3072,
        encoder_attention_heads: int = 12,
        activation_fn: str = "gelu",
        ## dropouts
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        encoder_layerdrop: float = 0.0,
        dropout_input: float = 0.0,
        dropout_features: float = 0.0,
        ## grad settings
        feature_grad_mult: float = 1.0,
        ## masking
        mask_prob: float = 0.65,
        mask_length: int = 10,
        mask_selection: str = "static",
        mask_other: int = 0,
        no_mask_overlap: bool = False,
        mask_min_space: int = 1,
        require_same_masks: bool = True,  # if set as True, collate_fn should be clipping
        mask_dropout: float = 0.0,
        ## channel masking
        mask_channel_length: int = 10,
        mask_channel_prob: float = 0.0,
        mask_channel_before: bool = False,
        mask_channel_selection: str = "static",
        mask_channel_other: int = 0,
        no_mask_channel_overlap: bool = False,
        mask_channel_min_space: int = 1,
        ## positional embeddings
        conv_pos: int = 128,
        conv_pos_groups: int = 16,
        pos_conv_depth: int = 1,
        max_positions: int = 100000,
        # EMA module
        average_top_k_layers: int = 8,
        layer_norm_target_layer: bool = False,
        instance_norm_target_layer: bool = False,
        instance_norm_targets: bool = False,
        layer_norm_targets: bool = False,
        batch_norm_target_layer: bool = False,
        group_norm_target_layer: bool = False,
        ema_decay: float = 0.999,
        ema_end_decay: float = 0.9999,
        ema_anneal_end_step: int = 100000,
        ema_transformer_only: bool = True,
        ema_layers_only: bool = True,
        min_target_var: float = 0.1,
        min_pred_var: float = 0.01,
        # Loss
        loss_beta: float = 0.0,
        loss_scale: float = None,
        # FP16 optimization
        required_seq_len_multiple: int = 2,
    ):
        super().__init__()

        # ConvFeatureExtractionModel
        self.conv_feature_layers = conv_feature_layers
        feature_enc_layers = eval(conv_feature_layers)
        self.extractor_embed = feature_enc_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=extractor_mode,
            in_d=input_size,
        )

        # Transformer Encoder
        ## model architecture
        self.layer_type = layer_type
        self.layer_norm_first = layer_norm_first
        self.encoder_layers = encoder_layers
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.activation_fn = activation_fn
        ## dropout
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.dropout_input = dropout_input
        self.dropout_features = dropout_features
        ## grad settings
        self.feature_grad_mult = feature_grad_mult
        ## masking
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.no_mask_overlap = no_mask_overlap
        self.mask_min_space = mask_min_space
        self.require_same_masks = (
            require_same_masks  # if set as True, collate_fn should be clipping
        )
        self.mask_dropout = mask_dropout
        ## channel masking
        self.mask_channel_length = mask_channel_length
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_before = mask_channel_before
        self.mask_channel_selection = mask_channel_selection
        self.mask_channel_other = mask_channel_other
        self.no_mask_channel_overlap = no_mask_channel_overlap
        self.mask_channel_min_space = mask_channel_min_space
        ## positional embeddings
        self.conv_pos = conv_pos
        self.conv_pos_groups = conv_pos_groups
        self.pos_conv_depth = pos_conv_depth
        self.max_positions = max_positions
        self.mask_emb = nn.Parameter(torch.FloatTensor(self.encoder_embed_dim).uniform_())
        self.encoder = TransformerEncoder(
            dropout=self.dropout,
            encoder_embed_dim=self.encoder_embed_dim,
            required_seq_len_multiple=required_seq_len_multiple,
            pos_conv_depth=self.pos_conv_depth,
            conv_pos=self.conv_pos,
            conv_pos_groups=self.conv_pos_groups,
            # transformer layers
            layer_type=self.layer_type,
            encoder_layers=self.encoder_layers,
            encoder_ffn_embed_dim=self.encoder_ffn_embed_dim,
            encoder_attention_heads=self.encoder_attention_heads,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            activation_fn=self.activation_fn,
            layer_norm_first=self.layer_norm_first,
            encoder_layerdrop=self.encoder_layerdrop,
            max_positions=self.max_positions,
        )
        ## projections and dropouts
        self.post_extract_proj = nn.Linear(self.extractor_embed, self.encoder_embed_dim)
        self.dropout_input = nn.Dropout(self.dropout_input)
        self.dropout_features = nn.Dropout(self.dropout_features)
        self.layer_norm = torch.nn.LayerNorm(self.extractor_embed)
        self.final_proj = nn.Linear(self.encoder_embed_dim, self.encoder_embed_dim)

        # EMA module
        self.average_top_k_layers = average_top_k_layers
        self.layer_norm_target_layer = layer_norm_target_layer
        self.instance_norm_target_layer = instance_norm_target_layer
        self.instance_norm_targets = instance_norm_targets
        self.layer_norm_targets = layer_norm_targets
        self.batch_norm_target_layer = batch_norm_target_layer
        self.group_norm_target_layer = group_norm_target_layer
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_end_step = ema_anneal_end_step
        self.ema_transformer_only = ema_transformer_only
        self.ema_layers_only = ema_layers_only
        self.min_target_var = min_target_var
        self.min_pred_var = min_pred_var
        self.ema = None

        # Loss
        self.loss_beta = loss_beta
        self.loss_scale = loss_scale

        # FP16 optimization
        self.required_seq_len_multiple = required_seq_len_multiple

        self.num_updates = 0

        logging.info("Data2VecEncoder settings: {}".format(self.__dict__))

    def make_ema_teacher(self):
        skip_keys = set()
        if self.ema_layers_only:
            self.ema_transformer_only = True
            for k, _ in self.encoder.pos_conv.named_parameters():
                skip_keys.add(f"pos_conv.{k}")

        self.ema = EMAModule(
            self.encoder if self.ema_transformer_only else self,
            ema_decay=self.ema_decay,
            ema_fp32=True,
            skip_keys=skip_keys,
        )

    def set_num_updates(self, num_updates):
        if self.ema is None and self.final_proj is not None:
            logging.info("Making EMA Teacher")
            self.make_ema_teacher()
        elif self.training and self.ema is not None:
            if self.ema_decay != self.ema_end_decay:
                if num_updates >= self.ema_anneal_end_step:
                    decay = self.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.ema_decay,
                        self.ema_end_decay,
                        num_updates,
                        self.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.encoder if self.ema_transformer_only else self)

        self.num_updates = num_updates

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
    ):
        B, T, C = x.shape

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices).to(x.device).unsqueeze(1).expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=1,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=self.require_same_masks,
                    mask_dropout=self.mask_dropout,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
                )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size).to(torch.float32) / stride + 1)

        conv_cfg_list = eval(self.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def forward(
        self,
        xs_pad,
        ilens=None,
        mask=False,
        features_only=True,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
    ):
        # create padding_mask by ilens
        if ilens is not None:
            padding_mask = make_pad_mask(lengths=ilens).to(xs_pad.device)
        else:
            padding_mask = None

        features = xs_pad

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(features)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(features)

        features = features.transpose(1, 2)

        features = self.layer_norm(features)

        orig_padding_mask = padding_mask

        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )
            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        pre_encoder_features = None
        if self.ema_transformer_only:
            pre_encoder_features = features.clone()

        features = self.dropout_input(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
        else:
            x = features
            mask_indices = None

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=layer,
        )

        if features_only:
            encoder_out_lens = (1 - padding_mask.long()).sum(1)
            return x, encoder_out_lens, None

        result = {
            "losses": {},
            "padding_mask": padding_mask,
            "x": x,
        }

        with torch.no_grad():
            self.ema.model.eval()

            if self.ema_transformer_only:
                y, layer_results = self.ema.model.extract_features(
                    pre_encoder_features,
                    padding_mask=padding_mask,
                    min_layer=self.encoder_layers - self.average_top_k_layers,
                )
                y = {
                    "x": y,
                    "padding_mask": padding_mask,
                    "layer_results": layer_results,
                }
            else:
                y = self.ema.model.extract_features(
                    source=xs_pad,
                    padding_mask=orig_padding_mask,
                    mask=False,
                )

            target_layer_results = [l[2] for l in y["layer_results"]]

            permuted = False
            if self.instance_norm_target_layer or self.batch_norm_target_layer:
                target_layer_results = [
                    tl.permute(1, 2, 0) for tl in target_layer_results  # TBC -> BCT
                ]
                permuted = True

            if self.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(tl.float(), running_mean=None, running_var=None, training=True)
                    for tl in target_layer_results
                ]

            if self.instance_norm_target_layer:
                target_layer_results = [F.instance_norm(tl.float()) for tl in target_layer_results]

            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]

            if self.group_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-2:]) for tl in target_layer_results
                ]

            if self.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:]) for tl in target_layer_results
                ]

            y = sum(target_layer_results) / len(target_layer_results)

            if self.layer_norm_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])

            if self.instance_norm_targets:
                y = F.instance_norm(y.float().transpose(1, 2)).transpose(1, 2)

            if not permuted:
                y = y.transpose(0, 1)

            y = y[mask_indices]

        x = x[mask_indices]
        x = self.final_proj(x)

        sz = x.size(-1)

        if self.loss_beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(
                x.float(), y.float(), reduction="none", beta=self.loss_beta
            ).sum(dim=-1)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(sz)

        result["losses"]["regression"] = loss.sum() * scale

        if "sample_size" not in result:
            result["sample_size"] = loss.numel()

        with torch.no_grad():
            result["target_var"] = self.compute_var(y)
            result["pred_var"] = self.compute_var(x.float())

        if self.num_updates > 5000 and result["target_var"] < self.min_target_var:
            logging.error(
                f"target var is {result['target_var'].item()} < {self.min_target_var}, exiting"
            )
            raise Exception(
                f"target var is {result['target_var'].item()} < {self.min_target_var}, exiting"
            )
        if self.num_updates > 5000 and result["pred_var"] < self.min_pred_var:
            logging.error(f"pred var is {result['pred_var'].item()} < {self.min_pred_var}, exiting")
            raise Exception(
                f"pred var is {result['pred_var'].item()} < {self.min_pred_var}, exiting"
            )

        if self.ema is not None:
            result["ema_decay"] = self.ema.get_decay() * 1000

        return result

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y**2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs**2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    def extract_features(self, xs_pad, ilens, mask=False, layer=None):
        res = self.forward(
            xs_pad,
            ilens,
            mask=mask,
            features_only=True,
            layer=layer,
        )
        return res

    def remove_pretraining_modules(self, last_layer=None):
        self.final_proj = None
        self.ema = None
        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )

    def output_size(self) -> int:
        return self.encoder_embed_dim
