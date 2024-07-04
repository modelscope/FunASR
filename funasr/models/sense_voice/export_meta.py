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
    # encoder_class = tables.encoder_classes.get(kwargs["encoder"] + "Export")
    # model.encoder = encoder_class(model.encoder, onnx=is_onnx)

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
    language: torch.Tensor,
    textnorm: torch.Tensor,
    **kwargs,
):
    speech = speech.to(device=kwargs["device"])
    speech_lengths = speech_lengths.to(device=kwargs["device"])

    language_query = self.embed(language).to(speech.device)

    textnorm_query = self.embed(textnorm).to(speech.device)
    speech = torch.cat((textnorm_query, speech), dim=1)
    speech_lengths += 1

    event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
        speech.size(0), 1, 1
    )
    input_query = torch.cat((language_query, event_emo_query), dim=1)
    speech = torch.cat((input_query, speech), dim=1)
    speech_lengths += 3

    # Encoder
    encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]

    # c. Passed the encoder result and the beam search
    ctc_logits = self.ctc.log_softmax(encoder_out)

    return ctc_logits, encoder_out_lens


def export_dummy_inputs(self):
    speech = torch.randn(2, 30, 560)
    speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
    language = torch.tensor([0, 0], dtype=torch.int32)
    textnorm = torch.tensor([15, 15], dtype=torch.int32)
    return (speech, speech_lengths, language, textnorm)


def export_input_names(self):
    return ["speech", "speech_lengths", "language", "textnorm"]


def export_output_names(self):
    return ["ctc_logits", "encoder_out_lens"]


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
