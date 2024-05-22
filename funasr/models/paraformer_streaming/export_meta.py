#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import types
import torch
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

    return model


def export_rebuild_model(model, **kwargs):
    # self.device = kwargs.get("device")
    is_onnx = kwargs.get("type", "onnx") == "onnx"
    encoder_class = tables.encoder_classes.get(kwargs["encoder"] + "Export")
    model.encoder = encoder_class(model.encoder, onnx=is_onnx)

    predictor_class = tables.predictor_classes.get(kwargs["predictor"] + "Export")
    model.predictor = predictor_class(model.predictor, onnx=is_onnx)

    if kwargs["decoder"] == "ParaformerSANMDecoder":
        kwargs["decoder"] = "ParaformerSANMDecoderOnline"
    decoder_class = tables.decoder_classes.get(kwargs["decoder"] + "Export")
    model.decoder = decoder_class(model.decoder, onnx=is_onnx)

    from funasr.utils.torch_function import sequence_mask

    model.make_pad_mask = sequence_mask(max_seq_len=None, flip=False)

    import copy
    import types

    encoder_model = copy.copy(model)
    decoder_model = copy.copy(model)

    # encoder
    encoder_model.forward = types.MethodType(export_encoder_forward, encoder_model)
    encoder_model.export_dummy_inputs = types.MethodType(export_encoder_dummy_inputs, encoder_model)
    encoder_model.export_input_names = types.MethodType(export_encoder_input_names, encoder_model)
    encoder_model.export_output_names = types.MethodType(export_encoder_output_names, encoder_model)
    encoder_model.export_dynamic_axes = types.MethodType(export_encoder_dynamic_axes, encoder_model)
    encoder_model.export_name = types.MethodType(export_encoder_name, encoder_model)

    # decoder
    decoder_model.forward = types.MethodType(export_decoder_forward, decoder_model)
    decoder_model.export_dummy_inputs = types.MethodType(export_decoder_dummy_inputs, decoder_model)
    decoder_model.export_input_names = types.MethodType(export_decoder_input_names, decoder_model)
    decoder_model.export_output_names = types.MethodType(export_decoder_output_names, decoder_model)
    decoder_model.export_dynamic_axes = types.MethodType(export_decoder_dynamic_axes, decoder_model)
    decoder_model.export_name = types.MethodType(export_decoder_name, decoder_model)

    return encoder_model, decoder_model


def export_encoder_forward(
    self,
    speech: torch.Tensor,
    speech_lengths: torch.Tensor,
):
    # a. To device
    batch = {"speech": speech, "speech_lengths": speech_lengths, "online": True}
    # batch = to_device(batch, device=self.device)

    enc, enc_len = self.encoder(**batch)
    mask = self.make_pad_mask(enc_len)[:, None, :]
    alphas, _ = self.predictor.forward_cnn(enc, mask)

    return enc, enc_len, alphas


def export_encoder_dummy_inputs(self):
    speech = torch.randn(2, 30, 560)
    speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
    return (speech, speech_lengths)


def export_encoder_input_names(self):
    return ["speech", "speech_lengths"]


def export_encoder_output_names(self):
    return ["enc", "enc_len", "alphas"]


def export_encoder_dynamic_axes(self):
    return {
        "speech": {0: "batch_size", 1: "feats_length"},
        "speech_lengths": {
            0: "batch_size",
        },
        "enc": {0: "batch_size", 1: "feats_length"},
        "enc_len": {
            0: "batch_size",
        },
        "alphas": {0: "batch_size", 1: "feats_length"},
    }


def export_encoder_name(self):
    return "model.onnx"


def export_decoder_forward(
    self,
    enc: torch.Tensor,
    enc_len: torch.Tensor,
    acoustic_embeds: torch.Tensor,
    acoustic_embeds_len: torch.Tensor,
    *args,
):
    decoder_out, out_caches = self.decoder(
        enc, enc_len, acoustic_embeds, acoustic_embeds_len, *args
    )
    sample_ids = decoder_out.argmax(dim=-1)

    return decoder_out, sample_ids, out_caches


def export_decoder_dummy_inputs(self):
    dummy_inputs = self.decoder.get_dummy_inputs(enc_size=self.encoder._output_size)
    return dummy_inputs


def export_decoder_input_names(self):
    return self.decoder.get_input_names()


def export_decoder_output_names(self):
    return self.decoder.get_output_names()


def export_decoder_dynamic_axes(self):
    return self.decoder.get_dynamic_axes()


def export_decoder_name(self):
    return "decoder.onnx"
