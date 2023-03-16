from typing import Tuple, Dict
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearTransform(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, input):
        output = self.linear(input)

        return output


class AffineTransform(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(AffineTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        output = self.linear(input)

        return output


class RectifiedLinear(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(RectifiedLinear, self).__init__()
        self.dim = input_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        out = self.relu(input)
        return out


class FSMNBlock(nn.Module):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            lorder=None,
            rorder=None,
            lstride=1,
            rstride=1,
    ):
        super(FSMNBlock, self).__init__()

        self.dim = input_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride

        self.conv_left = nn.Conv2d(
            self.dim, self.dim, [lorder, 1], dilation=[lstride, 1], groups=self.dim, bias=False)

        if self.rorder > 0:
            self.conv_right = nn.Conv2d(
                self.dim, self.dim, [rorder, 1], dilation=[rstride, 1], groups=self.dim, bias=False)
        else:
            self.conv_right = None

    def forward(self, input: torch.Tensor, cache: torch.Tensor):
        x = torch.unsqueeze(input, 1)
        x_per = x.permute(0, 3, 2, 1)  # B D T C
        
        cache = cache.to(x_per.device)
        y_left = torch.cat((cache, x_per), dim=2)
        cache = y_left[:, :, -(self.lorder - 1) * self.lstride:, :]
        y_left = self.conv_left(y_left)
        out = x_per + y_left

        if self.conv_right is not None:
            # maybe need to check
            y_right = F.pad(x_per, [0, 0, 0, self.rorder * self.rstride])
            y_right = y_right[:, :, self.rstride:, :]
            y_right = self.conv_right(y_right)
            out += y_right

        out_per = out.permute(0, 3, 2, 1)
        output = out_per.squeeze(1)

        return output, cache


class BasicBlock(nn.Sequential):
    def __init__(self,
                 linear_dim: int,
                 proj_dim: int,
                 lorder: int,
                 rorder: int,
                 lstride: int,
                 rstride: int,
                 stack_layer: int
                 ):
        super(BasicBlock, self).__init__()
        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride
        self.stack_layer = stack_layer
        self.linear = LinearTransform(linear_dim, proj_dim)
        self.fsmn_block = FSMNBlock(proj_dim, proj_dim, lorder, rorder, lstride, rstride)
        self.affine = AffineTransform(proj_dim, linear_dim)
        self.relu = RectifiedLinear(linear_dim, linear_dim)

    def forward(self, input: torch.Tensor, in_cache: Dict[str, torch.Tensor]):
        x1 = self.linear(input)  # B T D
        cache_layer_name = 'cache_layer_{}'.format(self.stack_layer)
        if cache_layer_name not in in_cache:
            in_cache[cache_layer_name] = torch.zeros(x1.shape[0], x1.shape[-1], (self.lorder - 1) * self.lstride, 1)
        x2, in_cache[cache_layer_name] = self.fsmn_block(x1, in_cache[cache_layer_name])
        x3 = self.affine(x2)
        x4 = self.relu(x3)
        return x4


class FsmnStack(nn.Sequential):
    def __init__(self, *args):
        super(FsmnStack, self).__init__(*args)

    def forward(self, input: torch.Tensor, in_cache: Dict[str, torch.Tensor]):
        x = input
        for module in self._modules.values():
            x = module(x, in_cache)
        return x


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


class FSMN(nn.Module):
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
            output_dim: int
    ):
        super(FSMN, self).__init__()

        self.input_dim = input_dim
        self.input_affine_dim = input_affine_dim
        self.fsmn_layers = fsmn_layers
        self.linear_dim = linear_dim
        self.proj_dim = proj_dim
        self.output_affine_dim = output_affine_dim
        self.output_dim = output_dim

        self.in_linear1 = AffineTransform(input_dim, input_affine_dim)
        self.in_linear2 = AffineTransform(input_affine_dim, linear_dim)
        self.relu = RectifiedLinear(linear_dim, linear_dim)
        self.fsmn = FsmnStack(*[BasicBlock(linear_dim, proj_dim, lorder, rorder, lstride, rstride, i) for i in
                                range(fsmn_layers)])
        self.out_linear1 = AffineTransform(linear_dim, output_affine_dim)
        self.out_linear2 = AffineTransform(output_affine_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def fuse_modules(self):
        pass

    def forward(
            self,
            input: torch.Tensor,
            in_cache: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            input (torch.Tensor): Input tensor (B, T, D)
            in_cache: when in_cache is not None, the forward is in streaming. The type of in_cache is a dict, egs,
            {'cache_layer_1': torch.Tensor(B, T1, D)}, T1 is equal to self.lorder. It is {} for the 1st frame
        """

        x1 = self.in_linear1(input)
        x2 = self.in_linear2(x1)
        x3 = self.relu(x2)
        x4 = self.fsmn(x3, in_cache)  # self.in_cache will update automatically in self.fsmn
        x5 = self.out_linear1(x4)
        x6 = self.out_linear2(x5)
        x7 = self.softmax(x6)

        return x7


'''
one deep fsmn layer
dimproj:                projection dimension, input and output dimension of memory blocks
dimlinear:              dimension of mapping layer
lorder:                 left order
rorder:                 right order
lstride:                left stride
rstride:                right stride
'''


class DFSMN(nn.Module):

    def __init__(self, dimproj=64, dimlinear=128, lorder=20, rorder=1, lstride=1, rstride=1):
        super(DFSMN, self).__init__()

        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride

        self.expand = AffineTransform(dimproj, dimlinear)
        self.shrink = LinearTransform(dimlinear, dimproj)

        self.conv_left = nn.Conv2d(
            dimproj, dimproj, [lorder, 1], dilation=[lstride, 1], groups=dimproj, bias=False)

        if rorder > 0:
            self.conv_right = nn.Conv2d(
                dimproj, dimproj, [rorder, 1], dilation=[rstride, 1], groups=dimproj, bias=False)
        else:
            self.conv_right = None

    def forward(self, input):
        f1 = F.relu(self.expand(input))
        p1 = self.shrink(f1)

        x = torch.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)

        y_left = F.pad(x_per, [0, 0, (self.lorder - 1) * self.lstride, 0])

        if self.conv_right is not None:
            y_right = F.pad(x_per, [0, 0, 0, (self.rorder) * self.rstride])
            y_right = y_right[:, :, self.rstride:, :]
            out = x_per + self.conv_left(y_left) + self.conv_right(y_right)
        else:
            out = x_per + self.conv_left(y_left)

        out1 = out.permute(0, 3, 2, 1)
        output = input + out1.squeeze(1)

        return output


'''
build stacked dfsmn layers
'''


def buildDFSMNRepeats(linear_dim=128, proj_dim=64, lorder=20, rorder=1, fsmn_layers=6):
    repeats = [
        nn.Sequential(
            DFSMN(proj_dim, linear_dim, lorder, rorder, 1, 1))
        for i in range(fsmn_layers)
    ]

    return nn.Sequential(*repeats)


if __name__ == '__main__':
    fsmn = FSMN(400, 140, 4, 250, 128, 10, 2, 1, 1, 140, 2599)
    print(fsmn)

    num_params = sum(p.numel() for p in fsmn.parameters())
    print('the number of model params: {}'.format(num_params))
    x = torch.zeros(128, 200, 400)  # batch-size * time * dim
    y, _ = fsmn(x)  # batch-size * time * dim
    print('input shape: {}'.format(x.shape))
    print('output shape: {}'.format(y.shape))

    print(fsmn.to_kaldi_net())
