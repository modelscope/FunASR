#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import types
import torch
import torch.nn.functional as F


def export_rebuild_model(model, **kwargs):
    model.device = kwargs.get("device")

    # store original forward since self.extract_features is calling it
    model._original_forward = model.forward

    model.forward = types.MethodType(export_forward, model)
    model.export_dummy_inputs = types.MethodType(export_dummy_inputs, model)
    model.export_input_names = types.MethodType(export_input_names, model)
    model.export_output_names = types.MethodType(export_output_names, model)
    model.export_dynamic_axes = types.MethodType(export_dynamic_axes, model)
    model.export_name = types.MethodType(export_name, model)

    return model


def export_forward(self, x: torch.Tensor):
    with torch.no_grad():
        if self.cfg.normalize:
            mean = torch.mean(x, dim=1, keepdim=True)
            var = torch.var(x, dim=1, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + 1e-5)
            x = x.view(x.shape[0], -1)

        # Call the original forward directly just like extract_features
        # Cannot directly use self.extract_features since it is being replaced by export_forward
        res = self._original_forward(
            source=x, padding_mask=None, mask=False, features_only=True, remove_extra_tokens=True
        )

        x = res["x"]

        return x


def export_dummy_inputs(self):
    return (torch.randn(1, 16000),)


def export_input_names(self):
    return ["input"]


def export_output_names(self):
    return ["output"]


def export_dynamic_axes(self):
    return {
        "input": {
            0: "batch_size",
            1: "sequence_length",
        },
        "output": {0: "batch_size", 1: "sequence_length"},
    }


def export_name(self):
    return "emotion2vec"
