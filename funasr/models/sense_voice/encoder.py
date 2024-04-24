import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from funasr.models.transformer.utils.nets_utils import make_pad_mask


def sense_voice_encode_forward(
    self,
    x: torch.Tensor,
    ilens: torch.Tensor = None,
    **kwargs,
):
    use_padmask = self.use_padmask
    x = F.gelu(self.conv1(x))
    x = F.gelu(self.conv2(x))
    x = x.permute(0, 2, 1)

    n_frames = x.size(1)
    max_pos = self.positional_embedding.size(0)
    max_pos = n_frames if n_frames < max_pos else max_pos
    x = (x[:, :max_pos, :] + self.positional_embedding[None, :max_pos, :]).to(x.dtype)

    if ilens is not None:
        if self.downsample_rate == 4:
            olens = (
                1
                + (ilens - self.conv1.kernel_size[0] + 2 * self.conv1.padding[0])
                // self.conv1.stride[0]
            )
        else:
            olens = ilens
        olens = (
            1
            + (olens - self.conv2.kernel_size[0] + 2 * self.conv2.padding[0])
            // self.conv2.stride[0]
        )
        olens = torch.clamp(olens, max=max_pos)
    else:
        olens = None

    if use_padmask and olens is not None:
        padding_mask = (~make_pad_mask(olens)[:, None, :]).to(x.device)
    else:
        padding_mask = None

    for layer, block in enumerate(self.blocks):
        x = block(x, mask=padding_mask, is_pad_mask=True)

    x = self.ln_post(x)

    if ilens is None:
        return x
    else:
        return x, olens
