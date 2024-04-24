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


def export_forward(self, feats: torch.Tensor, *args, **kwargs):

    scores, out_caches = self.encoder(feats, *args)

    return scores, out_caches


def export_dummy_inputs(self, data_in=None, frame=30):
    if data_in is None:
        speech = torch.randn(1, frame, self.encoder_conf.get("input_dim"))
    else:
        speech = None  # Undo

    cache_frames = self.encoder_conf.get("lorder") + self.encoder_conf.get("rorder") - 1
    in_cache0 = torch.randn(1, self.encoder_conf.get("proj_dim"), cache_frames, 1)
    in_cache1 = torch.randn(1, self.encoder_conf.get("proj_dim"), cache_frames, 1)
    in_cache2 = torch.randn(1, self.encoder_conf.get("proj_dim"), cache_frames, 1)
    in_cache3 = torch.randn(1, self.encoder_conf.get("proj_dim"), cache_frames, 1)

    return (speech, in_cache0, in_cache1, in_cache2, in_cache3)


def export_input_names(self):
    return ["speech", "in_cache0", "in_cache1", "in_cache2", "in_cache3"]


def export_output_names(self):
    return ["logits", "out_cache0", "out_cache1", "out_cache2", "out_cache3"]


def export_dynamic_axes(self):
    return {
        "speech": {1: "feats_length"},
    }


def export_name(
    self,
):
    return "model.onnx"
