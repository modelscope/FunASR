import os
from funasr.export import models

import torch
import torch.nn as nn


from funasr.export.utils.torch_function import MakePadMask
from funasr.export.utils.torch_function import sequence_mask

from funasr.modules.attention import MultiHeadedAttentionSANMDecoder
from funasr.export.models.modules.multihead_att import MultiHeadedAttentionSANMDecoder as MultiHeadedAttentionSANMDecoder_export
from funasr.modules.attention import MultiHeadedAttentionCrossAtt, MultiHeadedAttention
from funasr.export.models.modules.multihead_att import MultiHeadedAttentionCrossAtt as MultiHeadedAttentionCrossAtt_export
from funasr.export.models.modules.multihead_att import OnnxMultiHeadedAttention
from funasr.modules.positionwise_feed_forward import PositionwiseFeedForwardDecoderSANM
from funasr.export.models.modules.feedforward import PositionwiseFeedForwardDecoderSANM as PositionwiseFeedForwardDecoderSANM_export
from funasr.export.models.modules.decoder_layer import DecoderLayer as DecoderLayer_export


class ParaformerDecoderSAN(nn.Module):
    def __init__(self, model,
                 max_seq_len=512,
                 model_name='decoder',
                 onnx: bool = True,):
        super().__init__()
        # self.embed = model.embed #Embedding(model.embed, max_seq_len)
        self.model = model
        if onnx:
            self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        else:
            self.make_pad_mask = sequence_mask(max_seq_len, flip=False)

        for i, d in enumerate(self.model.decoders):
            if isinstance(d.feed_forward, PositionwiseFeedForwardDecoderSANM):
                d.feed_forward = PositionwiseFeedForwardDecoderSANM_export(d.feed_forward)
            if isinstance(d.self_attn, MultiHeadedAttentionSANMDecoder):
                d.self_attn = MultiHeadedAttentionSANMDecoder_export(d.self_attn)
            # if isinstance(d.src_attn, MultiHeadedAttentionCrossAtt):
            #     d.src_attn = MultiHeadedAttentionCrossAtt_export(d.src_attn)
            if isinstance(d.src_attn, MultiHeadedAttention):
                d.src_attn = OnnxMultiHeadedAttention(d.src_attn)
            self.model.decoders[i] = DecoderLayer_export(d)
        
        self.output_layer = model.output_layer
        self.after_norm = model.after_norm
        self.model_name = model_name
        

    def prepare_mask(self, mask):
        mask_3d_btd = mask[:, :, None]
        if len(mask.shape) == 2:
            mask_4d_bhlt = 1 - mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask_4d_bhlt = 1 - mask[:, None, :]
        mask_4d_bhlt = mask_4d_bhlt * -10000.0
    
        return mask_3d_btd, mask_4d_bhlt

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ):

        tgt = ys_in_pad
        tgt_mask = self.make_pad_mask(ys_in_lens)
        tgt_mask, _ = self.prepare_mask(tgt_mask)
        # tgt_mask = myutils.sequence_mask(ys_in_lens, device=tgt.device)[:, :, None]

        memory = hs_pad
        memory_mask = self.make_pad_mask(hlens)
        _, memory_mask = self.prepare_mask(memory_mask)
        # memory_mask = myutils.sequence_mask(hlens, device=memory.device)[:, None, :]

        x = tgt
        x, tgt_mask, memory, memory_mask = self.model.decoders(
            x, tgt_mask, memory, memory_mask
        )
        x = self.after_norm(x)
        x = self.output_layer(x)

        return x, ys_in_lens


    def get_dummy_inputs(self, enc_size):
        tgt = torch.LongTensor([0]).unsqueeze(0)
        memory = torch.randn(1, 100, enc_size)
        pre_acoustic_embeds = torch.randn(1, 1, enc_size)
        cache_num = len(self.model.decoders) + len(self.model.decoders2)
        cache = [
            torch.zeros((1, self.model.decoders[0].size, self.model.decoders[0].self_attn.kernel_size))
            for _ in range(cache_num)
        ]
        return (tgt, memory, pre_acoustic_embeds, cache)

    def is_optimizable(self):
        return True

    def get_input_names(self):
        cache_num = len(self.model.decoders) + len(self.model.decoders2)
        return ['tgt', 'memory', 'pre_acoustic_embeds'] \
               + ['cache_%d' % i for i in range(cache_num)]

    def get_output_names(self):
        cache_num = len(self.model.decoders) + len(self.model.decoders2)
        return ['y'] \
               + ['out_cache_%d' % i for i in range(cache_num)]

    def get_dynamic_axes(self):
        ret = {
            'tgt': {
                0: 'tgt_batch',
                1: 'tgt_length'
            },
            'memory': {
                0: 'memory_batch',
                1: 'memory_length'
            },
            'pre_acoustic_embeds': {
                0: 'acoustic_embeds_batch',
                1: 'acoustic_embeds_length',
            }
        }
        cache_num = len(self.model.decoders) + len(self.model.decoders2)
        ret.update({
            'cache_%d' % d: {
                0: 'cache_%d_batch' % d,
                2: 'cache_%d_length' % d
            }
            for d in range(cache_num)
        })
        return ret

    def get_model_config(self, path):
        return {
            "dec_type": "XformerDecoder",
            "model_path": os.path.join(path, f'{self.model_name}.onnx'),
            "n_layers": len(self.model.decoders) + len(self.model.decoders2),
            "odim": self.model.decoders[0].size
        }