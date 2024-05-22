#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch

from funasr.register import tables
from funasr.models.transformer.utils.nets_utils import get_activation


@tables.register("joint_network_classes", "joint_network")
class JointNetwork(torch.nn.Module):
    """Transducer joint network module.

    Args:
        output_size: Output size.
        encoder_size: Encoder output size.
        decoder_size: Decoder output size..
        joint_space_size: Joint space size.
        joint_act_type: Type of activation for joint network.
        **activation_parameters: Parameters for the activation function.

    """

    def __init__(
        self,
        output_size: int,
        encoder_size: int,
        decoder_size: int,
        joint_space_size: int = 256,
        joint_activation_type: str = "tanh",
    ) -> None:
        """Construct a JointNetwork object."""
        super().__init__()

        self.lin_enc = torch.nn.Linear(encoder_size, joint_space_size)
        self.lin_dec = torch.nn.Linear(decoder_size, joint_space_size, bias=False)

        self.lin_out = torch.nn.Linear(joint_space_size, output_size)

        self.joint_activation = get_activation(joint_activation_type)

    def forward(
        self,
        enc_out: torch.Tensor,
        dec_out: torch.Tensor,
        project_input: bool = True,
    ) -> torch.Tensor:
        """Joint computation of encoder and decoder hidden state sequences.

        Args:
            enc_out: Expanded encoder output state sequences (B, T, 1, D_enc)
            dec_out: Expanded decoder output state sequences (B, 1, U, D_dec)

        Returns:
            joint_out: Joint output state sequences. (B, T, U, D_out)

        """
        if project_input:
            joint_out = self.joint_activation(self.lin_enc(enc_out) + self.lin_dec(dec_out))
        else:
            joint_out = self.joint_activation(enc_out + dec_out)
        return self.lin_out(joint_out)
