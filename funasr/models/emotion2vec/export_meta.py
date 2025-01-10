#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import types
import torch


def export_rebuild_model(model, **kwargs):
    """Creates a wrapper model for ONNX export"""

    class WrapperModule(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.model.eval()

        def forward(self, x):
            with torch.no_grad():
                feats = self.model.extract_features(x, padding_mask=None)
                return feats["x"]  # Return only the features tensor

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

        @property
        def export_name(self):
            return "emotion2vec"
        
        def export_dummy_inputs(self):
            return torch.randn(1, 16000)

    wrapped_model = WrapperModule(model)
    return wrapped_model

# def export_dummy_inputs(self):
#     return torch.randn(1, 16000)

# def export_input_names(self):
#     return ["input"]


# def export_output_names(self):
#     return ["output"]


# def export_dynamic_axes(self):
#     return {
#         "input": {
#             0: "batch_size",
#             1: "sequence_length",
#         },
#         "output": {0: "batch_size", 1: "sequence_length"},
#     }
