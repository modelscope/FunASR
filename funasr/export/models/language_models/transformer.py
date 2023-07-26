import os

import torch
import torch.nn as nn
from funasr.modules.vgg2l import import VGG2L
from funasr.modules.attention import MultiHeadedAttention
from funasr.modules.subsampling import (
    Conv2dSubsampling, Conv2dSubsampling6, Conv2dSubsampling8)

from funasr.export.models.modules.encoder_layer import EncoderLayerConformer as OnnxEncoderLayer
from funasr.export.models.language_models.embed import Embedding
from funasr.export.models.modules.multihead_att import OnnxMultiHeadedAttention

from funasr.export.utils.torch_function import MakePadMask

class TransformerLM(nn.Module, AbsExportModel):
    def __init__(self, model, max_seq_len=512, **kwargs):
        super().__init__()
        self.embed = Embedding(model.embed, max_seq_len)
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        # replace multihead attention module into customized module.
        for i, d in enumerate(self.encoder.encoders):
            # d is EncoderLayer
            if isinstance(d.self_attn, MultiHeadedAttention):
                d.self_attn = OnnxMultiHeadedAttention(d.self_attn)
            self.encoder.encoders[i] = OnnxEncoderLayer(d)

        self.model_name = "transformer_lm"
        self.num_heads = self.encoder.encoders[0].self_attn.h
        self.hidden_size = self.encoder.encoders[0].self_attn.linear_out.out_features

    def prepare_mask(self, mask):
        if len(mask.shape) == 2:
            mask = mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask = mask[:, None, :]
        mask = 1 - mask
        return mask * -10000.0

    def forward(self, y, cache):
        feats_length = torch.ones(y.shape).sum(dim=-1).type(torch.long)
        mask = self.make_pad_mask(feats_length)  # (B, T)
        mask = (y != 0) * mask

        xs = self.embed(y)
        # forward_one_step of Encoder
        if isinstance(
            self.encoder.embed,
            (Conv2dSubsampling, Conv2dSubsampling6, Conv2dSubsampling8, VGG2L),
        ):
            xs, mask = self.encoder.embed(xs, mask)
        else:
            xs = self.encoder.embed(xs)

        new_cache = []
        mask = self.prepare_mask(mask)
        for c, e in zip(cache, self.encoder.encoders):
            xs, mask = e(xs, mask, c)
            new_cache.append(xs)

        if self.encoder.normalize_before:
            xs = self.encoder.after_norm(xs)

        h = self.decoder(xs[:, -1])
        return h, new_cache

    def get_dummy_inputs(self):
        tgt = torch.LongTensor([1]).unsqueeze(0)
        cache = [
            torch.zeros((1, 1, self.encoder.encoders[0].size))
            for _ in range(len(self.encoder.encoders))
        ]
        return (tgt, cache)

    def is_optimizable(self):
        return True

    def get_input_names(self):
        return ["tgt"] + ["cache_%d" % i for i in range(len(self.encoder.encoders))]

    def get_output_names(self):
        return ["y"] + ["out_cache_%d" % i for i in range(len(self.encoder.encoders))]

    def get_dynamic_axes(self):
        ret = {"tgt": {0: "tgt_batch", 1: "tgt_length"}}
        ret.update(
            {
                "cache_%d" % d: {0: "cache_%d_batch" % d, 1: "cache_%d_length" % d}
                for d in range(len(self.encoder.encoders))
            }
        )
        ret.update(
            {
                "out_cache_%d"
                % d: {0: "out_cache_%d_batch" % d, 1: "out_cache_%d_length" % d}
                for d in range(len(self.encoder.encoders))
            }
        )
        return ret

    def get_model_config(self, path):
        return {
            "use_lm": True,
            "model_path": os.path.join(path, f"{self.model_name}.onnx"),
            "lm_type": "TransformerLM",
            "odim": self.encoder.encoders[0].size,
            "nlayers": len(self.encoder.encoders),
        }
