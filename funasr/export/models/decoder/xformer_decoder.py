import os

import torch
import torch.nn as nn

from funasr.modules.attention import MultiHeadedAttention

from funasr.export.models.modules.decoder_layer import DecoderLayer as OnnxDecoderLayer
from funasr.export.models.language_models.embed import Embedding
from funasr.export.models.modules.multihead_att import \
    OnnxMultiHeadedAttention

from funasr.export.utils.torch_function import MakePadMask, subsequent_mask

class XformerDecoder(nn.Module):
    def __init__(self, model, max_seq_len=512, **kwargs):
        super().__init__()
        self.embed = Embedding(model.embed, max_seq_len)
        self.model = model
        self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        if isinstance(self.model.decoders[0].self_attn, MultiHeadedAttention):
            self.num_heads = self.model.decoders[0].self_attn.h
            self.hidden_size = self.model.decoders[0].self_attn.linear_out.out_features

        # replace multihead attention module into customized module.
        for i, d in enumerate(self.model.decoders):
            # d is DecoderLayer
            if isinstance(d.self_attn, MultiHeadedAttention):
                d.self_attn = OnnxMultiHeadedAttention(d.self_attn)
            if isinstance(d.src_attn, MultiHeadedAttention):
                d.src_attn = OnnxMultiHeadedAttention(d.src_attn)
            self.model.decoders[i] = OnnxDecoderLayer(d)

        self.model_name = "xformer_decoder"

    def prepare_mask(self, mask):
        if len(mask.shape) == 2:
            mask = mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask = mask[:, None, :]
        mask = 1 - mask
        return mask * -10000.0

    def forward(self, tgt, memory, cache):
        mask = subsequent_mask(tgt.size(-1)).unsqueeze(0)  # (B, T)

        x = self.embed(tgt)
        mask = self.prepare_mask(mask)
        new_cache = []
        for c, decoder in zip(cache, self.model.decoders):
            x, mask = decoder(x, mask, memory, None, c)
            new_cache.append(x)
            x = x[:, 1:, :]

        if self.model.normalize_before:
            y = self.model.after_norm(x[:, -1])
        else:
            y = x[:, -1]

        if self.model.output_layer is not None:
            y = torch.log_softmax(self.model.output_layer(y), dim=-1)
        return y, new_cache

    def get_dummy_inputs(self, enc_size):
        tgt = torch.LongTensor([0]).unsqueeze(0)
        enc_out = torch.randn(1, 100, enc_size)
        cache = [
            torch.zeros((1, 1, self.model.decoders[0].size))
            for _ in range(len(self.model.decoders))
        ]
        return (tgt, enc_out, cache)

    def is_optimizable(self):
        return True

    def get_input_names(self):
        return ["tgt", "memory"] + [
            "cache_%d" % i for i in range(len(self.model.decoders))
        ]

    def get_output_names(self):
        return ["y"] + ["out_cache_%d" % i for i in range(len(self.model.decoders))]

    def get_dynamic_axes(self):
        ret = {
            "tgt": {0: "tgt_batch", 1: "tgt_length"},
            "memory": {0: "memory_batch", 1: "memory_length"},
        }
        ret.update(
            {
                "cache_%d" % d: {0: "cache_%d_batch" % d, 1: "cache_%d_length" % d}
                for d in range(len(self.model.decoders))
            }
        )
        return ret

    def get_model_config(self, path):
        return {
            "dec_type": "XformerDecoder",
            "model_path": os.path.join(path, f"{self.model_name}.onnx"),
            "n_layers": len(self.model.decoders),
            "odim": self.model.decoders[0].size,
        }
