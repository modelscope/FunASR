#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import types
import torch
import torch.nn as nn
from funasr.register import tables


def export_rebuild_model(model, **kwargs):
    model.device = kwargs.get("device")
    is_onnx = kwargs.get("type", "onnx") == "onnx"
    encoder_class = tables.encoder_classes.get(kwargs["encoder"] + "Export")
    model.encoder = encoder_class(model.encoder, onnx=is_onnx)

    predictor_class = tables.predictor_classes.get(kwargs["predictor"] + "Export")
    model.predictor = predictor_class(model.predictor, onnx=is_onnx)

    decoder_class = tables.decoder_classes.get(kwargs["decoder"] + "Export")
    model.decoder = decoder_class(model.decoder, onnx=is_onnx)

    from funasr.utils.torch_function import sequence_mask

    model.make_pad_mask = sequence_mask(kwargs["max_seq_len"], flip=False)

    model.forward = types.MethodType(export_forward, model)
    model.export_dummy_inputs = types.MethodType(export_dummy_inputs, model)
    model.export_input_names = types.MethodType(export_input_names, model)
    model.export_output_names = types.MethodType(export_output_names, model)
    model.export_dynamic_axes = types.MethodType(export_dynamic_axes, model)
    model.export_name = types.MethodType(export_name, model)

    model.export_name = "model"
    return model


def export_forward(
    self,
    speech: torch.Tensor,
    speech_lengths: torch.Tensor,
):
    # a. To device
    batch = {"speech": speech, "speech_lengths": speech_lengths}
    # batch = to_device(batch, device=self.device)

    enc, enc_len = self.encoder(**batch)
    mask = self.make_pad_mask(enc_len)[:, None, :]
    pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(enc, mask)
    pre_token_length = pre_token_length.floor().type(torch.int32)

    decoder_out, _ = self.decoder(enc, enc_len, pre_acoustic_embeds, pre_token_length)
    decoder_out = torch.log_softmax(decoder_out, dim=-1)
    # sample_ids = decoder_out.argmax(dim=-1)

    return decoder_out, pre_token_length


def export_dummy_inputs(self):
    speech = torch.randn(2, 30, 560)
    speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
    return (speech, speech_lengths)


def export_input_names(self):
    return ["speech", "speech_lengths"]


def export_output_names(self):
    return ["logits", "token_num"]


def export_dynamic_axes(self):
    return {
        "speech": {0: "batch_size", 1: "feats_length"},
        "speech_lengths": {
            0: "batch_size",
        },
        "logits": {0: "batch_size", 1: "logits_length"},
    }


def export_name(
    self,
):
    return "model.onnx"


class EncoderLayerSANMExport(nn.Module):
    def __init__(
        self,
        model,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = model.self_attn
        self.feed_forward = model.feed_forward
        self.norm1 = model.norm1
        self.norm2 = model.norm2
        self.in_size = model.in_size
        self.size = model.size

    def forward(self, x, mask):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        if self.in_size == self.size:
            x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + residual

        return x, mask


import torch


@tables.register("encoder_classes", "SenseVoiceEncoderExport")
class SenseVoiceEncoderExport(nn.Module):
    def __init__(
        self,
        model,
        max_seq_len=512,
        feats_dim=560,
        model_name="encoder",
        onnx: bool = True,
    ):
        super().__init__()
        self.embed = model.embed

        self.model = model
        self.feats_dim = feats_dim
        self._output_size = model._output_size

        from funasr.utils.torch_function import sequence_mask

        self.make_pad_mask = sequence_mask(max_seq_len, flip=False)

        from funasr.models.sanm.attention import MultiHeadedAttentionSANMExport

        if hasattr(model, "encoders0"):
            for i, d in enumerate(self.model.encoders0):
                if isinstance(d.self_attn, MultiHeadedAttentionSANM):
                    d.self_attn = MultiHeadedAttentionSANMExport(d.self_attn)
                self.model.encoders0[i] = EncoderLayerSANMExport(d)

        for i, d in enumerate(self.model.encoders):
            if isinstance(d.self_attn, MultiHeadedAttentionSANM):
                d.self_attn = MultiHeadedAttentionSANMExport(d.self_attn)
            self.model.encoders[i] = EncoderLayerSANMExport(d)

        for i, d in enumerate(self.model.encoders):
            if isinstance(d.self_attn, MultiHeadedAttentionSANM):
                d.self_attn = MultiHeadedAttentionSANMExport(d.self_attn)
            self.model.encoders[i] = EncoderLayerSANMExport(d)

        self.model_name = model_name
        self.num_heads = model.encoders[0].self_attn.h
        self.hidden_size = model.encoders[0].self_attn.linear_out.out_features

    def prepare_mask(self, mask):
        mask_3d_btd = mask[:, :, None]
        if len(mask.shape) == 2:
            mask_4d_bhlt = 1 - mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask_4d_bhlt = 1 - mask[:, None, :]
        mask_4d_bhlt = mask_4d_bhlt * -10000.0

        return mask_3d_btd, mask_4d_bhlt

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor, online: bool = False):
        if not online:
            speech = speech * self._output_size**0.5
        mask = self.make_pad_mask(speech_lengths)
        mask = self.prepare_mask(mask)

        xs_pad = self.embed(speech)

        encoder_outs = self.model.encoders0(xs_pad, mask)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]

        encoder_outs = self.model.encoders(xs_pad, mask)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.model.after_norm(xs_pad)

        return xs_pad, speech_lengths

    def get_output_size(self):
        return self.model.encoders[0].size

    def get_dummy_inputs(self):
        feats = torch.randn(1, 100, self.feats_dim)
        return feats

    def get_input_names(self):
        return ["feats"]

    def get_output_names(self):
        return ["encoder_out", "encoder_out_lens", "predictor_weight"]

    def get_dynamic_axes(self):
        return {
            "feats": {1: "feats_length"},
            "encoder_out": {1: "enc_out_length"},
            "predictor_weight": {1: "pre_out_length"},
        }
