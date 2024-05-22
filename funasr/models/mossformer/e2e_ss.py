import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from funasr.models.base_model import FunASRModel
from funasr.models.encoder.mossformer_encoder import MossFormerEncoder, MossFormer_MaskNet
from funasr.models.decoder.mossformer_decoder import MossFormerDecoder


class MossFormer(FunASRModel):
    """The MossFormer model for separating input mixed speech into different speaker's speech.

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    num_blocks : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        Skip connection around intra.
    use_global_pos_enc : bool
        Global positional encodings.
    max_length : int
        Maximum sequence length.
    kernel_size: int
        Encoder and decoder kernel size
    """

    def __init__(
        self,
        in_channels=512,
        out_channels=512,
        num_blocks=24,
        kernel_size=16,
        norm="ln",
        num_spks=2,
        skip_around_intra=True,
        use_global_pos_enc=True,
        max_length=20000,
    ):
        super(MossFormer, self).__init__()
        self.num_spks = num_spks
        # Encoding
        self.enc = MossFormerEncoder(
            kernel_size=kernel_size, out_channels=in_channels, in_channels=1
        )

        ##Compute Mask
        self.mask_net = MossFormer_MaskNet(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            norm=norm,
            num_spks=num_spks,
            skip_around_intra=skip_around_intra,
            use_global_pos_enc=use_global_pos_enc,
            max_length=max_length,
        )
        self.dec = MossFormerDecoder(
            in_channels=out_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            bias=False,
        )

    def forward(self, input):
        x = self.enc(input)
        mask = self.mask_net(x)
        x = torch.stack([x] * self.num_spks)
        sep_x = x * mask

        # Decoding
        est_source = torch.cat(
            [self.dec(sep_x[i]).unsqueeze(-1) for i in range(self.num_spks)],
            dim=-1,
        )
        T_origin = input.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        out = []
        for spk in range(self.num_spks):
            out.append(est_source[:, :, spk])
        return out
