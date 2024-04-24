#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import types
import torch
from funasr.register import tables


def export_rebuild_model(model, **kwargs):

    is_onnx = kwargs.get("type", "onnx") == "onnx"
    encoder_class = tables.encoder_classes.get(kwargs["encoder"] + "Export")
    model.encoder = encoder_class(model.encoder, onnx=is_onnx)

    model.forward = types.MethodType(export_forward, model)
    model.export_dummy_inputs = types.MethodType(export_dummy_inputs, model)
    model.export_input_names = types.MethodType(export_input_names, model)
    model.export_output_names = types.MethodType(export_output_names, model)
    model.export_dynamic_axes = types.MethodType(export_dynamic_axes, model)
    model.export_name = types.MethodType(export_name, model)

    return model


def export_forward(
    self,
    inputs: torch.Tensor,
    text_lengths: torch.Tensor,
    vad_indexes: torch.Tensor,
    sub_masks: torch.Tensor,
):
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


def export_dummy_inputs(self):
    length = 120
    text_indexes = torch.randint(0, self.embed.num_embeddings, (1, length)).type(torch.int32)
    text_lengths = torch.tensor([length], dtype=torch.int32)
    vad_mask = torch.ones(length, length, dtype=torch.float32)[None, None, :, :]
    sub_masks = torch.ones(length, length, dtype=torch.float32)
    sub_masks = torch.tril(sub_masks).type(torch.float32)
    return (text_indexes, text_lengths, vad_mask, sub_masks[None, None, :, :])


def export_input_names(self):
    return ["inputs", "text_lengths", "vad_masks", "sub_masks"]


def export_output_names(self):
    return ["logits"]


def export_dynamic_axes(self):
    return {
        "inputs": {1: "feats_length"},
        "vad_masks": {2: "feats_length1", 3: "feats_length2"},
        "sub_masks": {2: "feats_length1", 3: "feats_length2"},
        "logits": {1: "logits_length"},
    }


def export_name(self):
    return "model.onnx"
