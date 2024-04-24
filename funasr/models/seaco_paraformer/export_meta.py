#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch

from funasr.register import tables


class ContextualEmbedderExport(torch.nn.Module):
    def __init__(
        self,
        model,
        max_seq_len=512,
        feats_dim=560,
        **kwargs,
    ):
        super().__init__()
        self.embedding = model.decoder.embed  # model.bias_embed
        model.bias_encoder.batch_first = False
        self.bias_encoder = model.bias_encoder

    def forward(self, hotword):
        hotword = self.embedding(hotword).transpose(0, 1)  # batch second
        hw_embed, (_, _) = self.bias_encoder(hotword)
        return hw_embed

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
        return hotword

    def export_input_names(self):
        return ["hotword"]

    def export_output_names(self):
        return ["hw_embed"]

    def export_dynamic_axes(self):
        return {
            "hotword": {
                0: "num_hotwords",
            },
            "hw_embed": {
                0: "num_hotwords",
            },
        }

    def export_name(self):
        return "model_eb.onnx"


def export_rebuild_model(model, **kwargs):
    model.device = kwargs.get("device")
    is_onnx = kwargs.get("type", "onnx") == "onnx"
    encoder_class = tables.encoder_classes.get(kwargs["encoder"] + "Export")
    model.encoder = encoder_class(model.encoder, onnx=is_onnx)

    predictor_class = tables.predictor_classes.get(kwargs["predictor"] + "Export")
    model.predictor = predictor_class(model.predictor, onnx=is_onnx)

    # before decoder convert into export class
    embedder_class = ContextualEmbedderExport
    embedder_model = embedder_class(model, onnx=is_onnx)

    decoder_class = tables.decoder_classes.get(kwargs["decoder"] + "Export")
    model.decoder = decoder_class(model.decoder, onnx=is_onnx)

    seaco_decoder_class = tables.decoder_classes.get(kwargs["seaco_decoder"] + "Export")
    model.seaco_decoder = seaco_decoder_class(model.seaco_decoder, onnx=is_onnx)

    from funasr.utils.torch_function import sequence_mask

    model.make_pad_mask = sequence_mask(kwargs["max_seq_len"], flip=False)

    from funasr.utils.torch_function import sequence_mask

    model.make_pad_mask = sequence_mask(kwargs["max_seq_len"], flip=False)
    model.feats_dim = 560
    model.NOBIAS = 8377

    import copy
    import types

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
    backbone_model.export_name = types.MethodType(export_backbone_name, backbone_model)

    return backbone_model, embedder_model


def export_backbone_forward(
    self,
    speech: torch.Tensor,
    speech_lengths: torch.Tensor,
    bias_embed: torch.Tensor,
    # lmbd: float,
):
    # a. To device
    batch = {"speech": speech, "speech_lengths": speech_lengths}

    enc, enc_len = self.encoder(**batch)
    mask = self.make_pad_mask(enc_len)[:, None, :]
    pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(enc, mask)
    pre_token_length = pre_token_length.floor().type(torch.int32)

    decoder_out, decoder_hidden, _ = self.decoder(
        enc, enc_len, pre_acoustic_embeds, pre_token_length, return_hidden=True, return_both=True
    )
    decoder_out = torch.log_softmax(decoder_out, dim=-1)
    # seaco forward
    B, N, D = bias_embed.shape
    _contextual_length = torch.ones(B) * N

    # ASF
    hotword_scores = self.seaco_decoder.forward_asf6(
        bias_embed, _contextual_length, decoder_hidden, pre_token_length
    )
    hotword_scores = hotword_scores[0].sum(0).sum(0)
    # _ = self.decoder2(bias_embed, _contextual_length, decoder_hidden, pre_token_length)
    # hotword_scores = self.decoder2.model.decoders[-1].attn_mat[0][0].sum(0).sum(0)
    dec_filter = torch.sort(hotword_scores, descending=True)[1][:51]
    contextual_info = bias_embed[:, dec_filter]
    num_hot_word = contextual_info.shape[1]
    _contextual_length = torch.Tensor([num_hot_word]).int().repeat(B).to(enc.device)

    # again
    cif_attended, _ = self.seaco_decoder(
        contextual_info, _contextual_length, pre_acoustic_embeds, pre_token_length
    )
    dec_attended, _ = self.seaco_decoder(
        contextual_info, _contextual_length, decoder_hidden, pre_token_length
    )
    merged = cif_attended + dec_attended
    dha_output = self.hotword_output_layer(merged)
    dha_pred = torch.log_softmax(dha_output, dim=-1)
    # merging logits
    dha_ids = dha_pred.max(-1)[-1]
    dha_mask = (dha_ids == self.NOBIAS).int().unsqueeze(-1)
    decoder_out = decoder_out * dha_mask + dha_pred * (1 - dha_mask)
    return decoder_out, pre_token_length, alphas


def export_backbone_dummy_inputs(self):
    speech = torch.randn(2, 30, self.feats_dim)
    speech_lengths = torch.tensor([15, 30], dtype=torch.int32)
    bias_embed = torch.randn(2, 1, 512)
    return (speech, speech_lengths, bias_embed)


def export_backbone_input_names(self):
    return ["speech", "speech_lengths", "bias_embed"]


def export_backbone_output_names(self):
    return ["logits", "token_num", "alphas"]


def export_backbone_dynamic_axes(self):
    return {
        "speech": {0: "batch_size", 1: "feats_length"},
        "speech_lengths": {
            0: "batch_size",
        },
        "bias_embed": {0: "batch_size", 1: "num_hotwords"},
        "logits": {0: "batch_size", 1: "logits_length"},
        "pre_acoustic_embeds": {1: "feats_length1"},
    }


def export_backbone_name(self):
    return "model.onnx"
