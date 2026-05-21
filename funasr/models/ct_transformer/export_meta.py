#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import types
import torch
from funasr.register import tables


def export_rebuild_model(model, **kwargs):

    """Export rebuild model.
    
        Args:
            model: Model instance or model name.
            **kwargs: Additional keyword arguments.
        """
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


def export_forward(self, inputs: torch.Tensor, text_lengths: torch.Tensor):
    """Compute loss value from buffer sequences.

    Args:
        input (torch.Tensor): Input ids. (batch, len)
        hidden (torch.Tensor): Target ids. (batch, len)

    """
    x = self.embed(inputs)
    h, _ = self.encoder(x, text_lengths)
    y = self.decoder(h)
    return y


def export_dummy_inputs(self):
    """Export dummy inputs."""
    length = 120
    text_indexes = torch.randint(0, self.embed.num_embeddings, (2, length)).type(torch.int32)
    text_lengths = torch.tensor([length - 20, length], dtype=torch.int32)
    return (text_indexes, text_lengths)


def export_input_names(self):
    """Export input names."""
    return ["inputs", "text_lengths"]


def export_output_names(self):
    """Export output names."""
    return ["logits"]


def export_dynamic_axes(self):
    """Export dynamic axes."""
    return {
        "inputs": {0: "batch_size", 1: "feats_length"},
        "text_lengths": {
            0: "batch_size",
        },
        "logits": {0: "batch_size", 1: "logits_length"},
    }


def export_name(self):
    """Export name."""
    return "model.onnx"
