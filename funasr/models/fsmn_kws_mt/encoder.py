from typing import Tuple, Dict
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from funasr.models.fsmn_kws.encoder import (toKaldiMatrix, LinearTransform, AffineTransform, RectifiedLinear, FSMNBlock, FsmnStack, BasicBlock)


from funasr.register import tables


'''
FSMN net for keyword spotting
input_dim:              input dimension
linear_dim:             fsmn input dimensionll
proj_dim:               fsmn projection dimension
lorder:                 fsmn left order
rorder:                 fsmn right order
num_syn:                output dimension
fsmn_layers:            no. of sequential fsmn layers
'''

@tables.register("encoder_classes", "FSMNMT")
class FSMNMT(nn.Module):
    def __init__(
            self,
            input_dim: int,
            input_affine_dim: int,
            fsmn_layers: int,
            linear_dim: int,
            proj_dim: int,
            lorder: int,
            rorder: int,
            lstride: int,
            rstride: int,
            output_affine_dim: int,
            output_dim: int,
            output_dim2: int,
            use_softmax: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.input_affine_dim = input_affine_dim
        self.fsmn_layers = fsmn_layers
        self.linear_dim = linear_dim
        self.proj_dim = proj_dim
        self.output_affine_dim = output_affine_dim
        self.output_dim = output_dim
        self.output_dim2 = output_dim2

        self.in_linear1 = AffineTransform(input_dim, input_affine_dim)
        self.in_linear2 = AffineTransform(input_affine_dim, linear_dim)
        self.relu = RectifiedLinear(linear_dim, linear_dim)
        self.fsmn = FsmnStack(*[BasicBlock(linear_dim, proj_dim, lorder, rorder, lstride, rstride, i) for i in
                                range(fsmn_layers)])
        self.out_linear1 = AffineTransform(linear_dim, output_affine_dim)
        self.out_linear1_2 = AffineTransform(linear_dim, output_affine_dim)
        self.out_linear2 = AffineTransform(output_affine_dim, output_dim)
        self.out_linear2_2 = AffineTransform(output_affine_dim, output_dim2)

        self.use_softmax = use_softmax
        if self.use_softmax:
            self.softmax = nn.Softmax(dim=-1)

    def output_size(self) -> int:
        return self.output_dim

    def output_size2(self) -> int:
        return self.output_dim2

    def forward(
            self,
            input: torch.Tensor,
            cache: Dict[str, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            input (torch.Tensor): Input tensor (B, T, D)
            cache: when cache is not None, the forward is in streaming. The type of cache is a dict, egs,
            {'cache_layer_1': torch.Tensor(B, T1, D)}, T1 is equal to self.lorder. It is {} for the 1st frame
        """

        x1 = self.in_linear1(input)
        x2 = self.in_linear2(x1)
        x3 = self.relu(x2)
        x4 = self.fsmn(x3, cache)  # self.cache will update automatically in self.fsmn
        x5 = self.out_linear1(x4)
        x6 = self.out_linear2(x5)

        x5_2 = self.out_linear1_2(x4)
        x6_2 = self.out_linear2_2(x5_2)

        if self.use_softmax:
            x7 = self.softmax(x6)
            x7_2 = self.softmax(x6_2)
            return x7, x7_2

        return x6, x6_2


@tables.register("encoder_classes", "FSMNMTConvert")
class FSMNMTConvert(nn.Module):
    def __init__(
            self,
            input_dim: int,
            input_affine_dim: int,
            fsmn_layers: int,
            linear_dim: int,
            proj_dim: int,
            lorder: int,
            rorder: int,
            lstride: int,
            rstride: int,
            output_affine_dim: int,
            output_dim: int,
            output_dim2: int,
            use_softmax: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.input_affine_dim = input_affine_dim
        self.fsmn_layers = fsmn_layers
        self.linear_dim = linear_dim
        self.proj_dim = proj_dim
        self.output_affine_dim = output_affine_dim
        self.output_dim = output_dim
        self.output_dim2 = output_dim2

        self.in_linear1 = AffineTransform(input_dim, input_affine_dim)
        self.in_linear2 = AffineTransform(input_affine_dim, linear_dim)
        self.relu = RectifiedLinear(linear_dim, linear_dim)
        self.fsmn = FsmnStack(*[BasicBlock(linear_dim, proj_dim, lorder, rorder, lstride, rstride, i) for i in
                                range(fsmn_layers)])
        self.out_linear1 = AffineTransform(linear_dim, output_affine_dim)
        self.out_linear1_2 = AffineTransform(linear_dim, output_affine_dim)
        self.out_linear2 = AffineTransform(output_affine_dim, output_dim)
        self.out_linear2_2 = AffineTransform(output_affine_dim, output_dim2)

        self.use_softmax = use_softmax
        if self.use_softmax:
            self.softmax = nn.Softmax(dim=-1)

    def output_size(self) -> int:
        return self.output_dim

    def output_size2(self) -> int:
        return self.output_dim2

    def to_kaldi_net(self):
        re_str = ''
        re_str += '<Nnet>\n'
        re_str += self.in_linear1.to_kaldi_net()
        re_str += self.in_linear2.to_kaldi_net()
        re_str += self.relu.to_kaldi_net()

        for fsmn in self.fsmn:
            re_str += fsmn.to_kaldi_net()

        re_str += self.out_linear1.to_kaldi_net()
        re_str += self.out_linear2.to_kaldi_net()
        re_str += '<Softmax> %d %d\n' % (self.output_dim, self.output_dim)
        re_str += '</Nnet>\n'

        return re_str

    def to_kaldi_net2(self):
        re_str = ''
        re_str += '<Nnet>\n'
        re_str += self.in_linear1.to_kaldi_net()
        re_str += self.in_linear2.to_kaldi_net()
        re_str += self.relu.to_kaldi_net()

        for fsmn in self.fsmn:
            re_str += fsmn.to_kaldi_net()

        re_str += self.out_linear1_2.to_kaldi_net()
        re_str += self.out_linear2_2.to_kaldi_net()
        re_str += '<Softmax> %d %d\n' % (self.output_dim2, self.output_dim2)
        re_str += '</Nnet>\n'

        return re_str

    def to_pytorch_net(self, kaldi_file):
        with open(kaldi_file, 'r', encoding='utf8') as fread:
            fread = open(kaldi_file, 'r')
            nnet_start_line = fread.readline()
            assert nnet_start_line.strip() == '<Nnet>'

            self.in_linear1.to_pytorch_net(fread)
            self.in_linear2.to_pytorch_net(fread)
            self.relu.to_pytorch_net(fread)

            for fsmn in self.fsmn:
                fsmn.to_pytorch_net(fread)

            self.out_linear1.to_pytorch_net(fread)
            self.out_linear2.to_pytorch_net(fread)

            softmax_line = fread.readline()
            softmax_split = softmax_line.strip().split()
            assert softmax_split[0].strip() == '<Softmax>'
            assert int(softmax_split[1]) == self.output_dim
            assert int(softmax_split[2]) == self.output_dim

            nnet_end_line = fread.readline()
            assert nnet_end_line.strip() == '</Nnet>'
        fread.close()
