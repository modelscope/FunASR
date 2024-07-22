#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch
import types

from funasr.register import tables
from funasr.models.seaco_paraformer.export_meta import ContextualEmbedderExport


class ContextualEmbedderExport2(ContextualEmbedderExport):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.embedding = model.bias_embed
        model.bias_encoder.batch_first = False
        self.bias_encoder = model.bias_encoder
    
    def export_dummy_inputs(self):
        hotword = torch.tensor(
            [
                [10, 11, 12, 13, 14, 10, 11, 12, 13, 14],
                [100, 101, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [10, 11, 12, 13, 14, 10, 11, 12, 13, 14],
                [100, 101, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.int32,
        )
        # hotword_length = torch.tensor([10, 2, 1], dtype=torch.int32)
        return (hotword)


def export_rebuild_model(model, **kwargs):
    is_onnx = kwargs.get("type", "onnx") == "onnx"

    encoder_class = tables.encoder_classes.get(kwargs["encoder"] + "Export")
    model.encoder = encoder_class(model.encoder, onnx=is_onnx)

    predictor_class = tables.predictor_classes.get(kwargs["predictor"] + "Export")
    model.predictor = predictor_class(model.predictor, onnx=is_onnx)

    # little difference with bias encoder with seaco paraformer
    embedder_class = ContextualEmbedderExport2
    embedder_model = embedder_class(model, onnx=is_onnx)

    if kwargs["decoder"] == "ParaformerSANMDecoder":
        kwargs["decoder"] = "ParaformerSANMDecoderOnline"
    decoder_class = tables.decoder_classes.get(kwargs["decoder"] + "Export")
    model.decoder = decoder_class(model.decoder, onnx=is_onnx)

    from funasr.utils.torch_function import sequence_mask

    model.make_pad_mask = sequence_mask(kwargs["max_seq_len"], flip=False)
    model.feats_dim = 560

    import copy

    backbone_model = copy.copy(model)

    # backbone
    backbone_model.forward = types.MethodType(export_backbone_forward, backbone_model)
    backbone_model.export_dummy_inputs = types.MethodType(
        export_backbone_dummy_inputs, backbone_model
    )
    backbone_model.export_input_names = types.MethodType(
        export_backbone_input_names, backbone_model
    )
    backbone_model.export_output_names = types.MethodType(
        export_backbone_output_names, backbone_model
    )
    backbone_model.export_dynamic_axes = types.MethodType(
        export_backbone_dynamic_axes, backbone_model
    )
    
    embedder_model.export_name = "model_eb"
    backbone_model.export_name = "model"

    return backbone_model, embedder_model


def export_backbone_forward(
    self,
    speech: torch.Tensor,
    speech_lengths: torch.Tensor,
    bias_embed: torch.Tensor,
):
    batch = {"speech": speech, "speech_lengths": speech_lengths}

    enc, enc_len = self.encoder(**batch)
    mask = self.make_pad_mask(enc_len)[:, None, :]
    pre_acoustic_embeds, pre_token_length, _, _ = self.predictor(enc, mask)
    pre_token_length = pre_token_length.floor().type(torch.int32)

    decoder_out, _ = self.decoder(enc, enc_len, pre_acoustic_embeds, pre_token_length, bias_embed)
    decoder_out = torch.log_softmax(decoder_out, dim=-1)

    return decoder_out, pre_token_length


def export_backbone_dummy_inputs(self):
    speech = torch.randn(2, 30, self.feats_dim)
    speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
    bias_embed = torch.randn(2, 1, 512)
    return (speech, speech_lengths, bias_embed)


def export_backbone_input_names(self):
    return ["speech", "speech_lengths", "bias_embed"]


def export_backbone_output_names(self):
    return ["logits", "token_num"]


def export_backbone_dynamic_axes(self):
    return {
        "speech": {0: "batch_size", 1: "feats_length"},
        "speech_lengths": {
            0: "batch_size",
        },
        "bias_embed": {0: "batch_size", 1: "num_hotwords"},
        "logits": {0: "batch_size", 1: "logits_length"},
    }


def export_backbone_name(self):
    return "model.onnx"
