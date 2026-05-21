#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import types
import copy
import torch
from funasr.register import tables


def export_rebuild_model(model, **kwargs):
    # self.device = kwargs.get("device")
    """Export rebuild model.
    
        Args:
            model: Model instance or model name.
            **kwargs: Additional keyword arguments.
        """
    is_onnx = kwargs.get("type", "onnx") == "onnx"
    encoder_class = tables.encoder_classes.get(kwargs["encoder"] + "Export")

    if hasattr(model, "ctc"):
        model.encoder = encoder_class(
            model.encoder,
            onnx=is_onnx,
            feats_dim=kwargs.get("input_size", 560),
            ctc_linear=model.ctc.ctc_lo
        )
    else:
        assert False, print(model)
        model.encoder = encoder_class(model.encoder, onnx=is_onnx, feats_dim=kwargs.get("input_size", 560))

    # from funasr.utils.torch_function import sequence_mask
    # model.make_pad_mask = sequence_mask(max_seq_len=None, flip=False)

    encoder_model = copy.copy(model)

    # encoder
    encoder_model.forward = types.MethodType(export_encoder_forward, encoder_model)
    encoder_model.export_dummy_inputs = types.MethodType(export_encoder_dummy_inputs, encoder_model)
    encoder_model.export_input_names = types.MethodType(export_encoder_input_names, encoder_model)
    encoder_model.export_output_names = types.MethodType(export_encoder_output_names, encoder_model)
    encoder_model.export_dynamic_axes = types.MethodType(export_encoder_dynamic_axes, encoder_model)
    encoder_model.export_name = types.MethodType(export_encoder_name, encoder_model)

    return encoder_model


def export_encoder_forward(
    self,
    speech: torch.Tensor,
    speech_lengths: torch.Tensor,
):
    # a. To device
    """Export encoder forward.
    
        Args:
            speech: Speech audio tensor, shape (batch, time).
            speech_lengths: Length of each speech sample.
        """
    batch = {
        "speech": speech,
        "speech_lengths": speech_lengths,
        "online": True
    }
    # batch = to_device(batch, device=self.device)

    encoder_out, encoder_out_len = self.encoder(**batch)
    # mask = self.make_pad_mask(encoder_out_len)[:, None, :]
    # alphas, _ = self.predictor.forward_cnn(enc, mask)

    # return encoder_out, encoder_out_len, alphas
    return encoder_out, encoder_out_len


def export_encoder_dummy_inputs(self):
    """Export encoder dummy inputs."""
    speech = torch.randn(2, 30, 280)
    speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
    return (speech, speech_lengths)


def export_encoder_input_names(self):
    """Export encoder input names."""
    return ["speech", "speech_lengths"]


def export_encoder_output_names(self):
    # return ["encoder_out", "encoder_out_len", "alphas"]
    """Export encoder output names."""
    return ["encoder_out", "encoder_out_len"]


def export_encoder_dynamic_axes(self):
    """Export encoder dynamic axes."""
    return {
        "speech": {
            0: "batch_size", 1: "feats_length"
        },
        "speech_lengths": {
            0: "batch_size",
        },
        "encoder_out": {
            0: "batch_size", 1: "feats_length"
        },
        "encoder_out_len": {
            0: "batch_size",
        },
    }


def export_encoder_name(self):
    """Export encoder name."""
    return "encoder.onnx"
