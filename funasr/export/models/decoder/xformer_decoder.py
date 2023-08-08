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
    def __init__(self,
                 model,
                 max_seq_len = 512,
                 model_name = 'decoder',
                 onnx: bool = True,):
        super().__init__()
        self.embed = Embedding(model.embed, max_seq_len)
        self.model = model
        if onnx:
            self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        else:
            self.make_pad_mask = subsequent_mask(max_seq_len, flip=False)

        if isinstance(self.model.decoders[0].self_attn, MultiHeadedAttention):
            self.num_heads = self.model.decoders[0].self_attn.h
            self.hidden_size = self.model.decoders[0].self_attn.linear_out.out_features

        # replace multi-head attention module into customized module.
        for i, d in enumerate(self.model.decoders):
            # d is DecoderLayer
            if isinstance(d.self_attn, MultiHeadedAttention):
                d.self_attn = OnnxMultiHeadedAttention(d.self_attn)
            if isinstance(d.src_attn, MultiHeadedAttention):
                d.src_attn = OnnxMultiHeadedAttention(d.src_attn)
            self.model.decoders[i] = OnnxDecoderLayer(d)

        self.model_name = model_name

    def prepare_mask(self, mask):
        mask_3d_btd = mask[:, :, None]
        if len(mask.shape) == 2:
            mask_4d_bhlt = 1 - mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask_4d_bhlt = 1 - mask[:, None, :]

        mask_4d_bhlt = mask_4d_bhlt * -10000.0
        return mask_3d_btd, mask_4d_bhlt

    def forward(self,
                tgt,
                memory,
                cache):

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
        memory = torch.randn(1, 100, enc_size)
        cache_num = len(self.model.decoders)
        cache = [
            torch.zeros((1, 1, self.model.decoders[0].size))
            for _ in range(cache_num)
        ]
        return (tgt, memory, cache)

    def is_optimizable(self):
        return True

    def get_input_names(self):
        cache_num = len(self.model.decoders)
        return ["tgt", "memory"] + [
            "cache_%d" % i for i in range(cache_num)
        ]

    def get_output_names(self):
        cache_num = len(self.model.decoders)
        return ["y"] + ["out_cache_%d" % i for i in range(cache_num)]

    def get_dynamic_axes(self):
        ret = {
            "tgt": {0: "tgt_batch", 1: "tgt_length"},
            "memory": {0: "memory_batch", 1: "memory_length"},
        }
        cache_num = len(self.model.decoders)
        ret.update(
            {
                "cache_%d" % d: {0: "cache_%d_batch" % d, 2: "cache_%d_length" % d}
                for d in range(cache_num)
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
