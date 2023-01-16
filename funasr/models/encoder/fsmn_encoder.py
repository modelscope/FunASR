import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class LinearTransform(nn.Module):

    def __init__(self, input_dim, output_dim, quantize=0):
        super(LinearTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.quantize = quantize
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input):
        if self.quantize:
            output = self.quant(input)
        else:
            output = input
        output = self.linear(output)
        if self.quantize:
            output = self.dequant(output)

        return output


class AffineTransform(nn.Module):

    def __init__(self, input_dim, output_dim, quantize=0):
        super(AffineTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.quantize = quantize
        self.linear = nn.Linear(input_dim, output_dim)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input):
        if self.quantize:
            output = self.quant(input)
        else:
            output = input
        output = self.linear(output)
        if self.quantize:
            output = self.dequant(output)

        return output


class FSMNBlock(nn.Module):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            lorder=None,
            rorder=None,
            lstride=1,
            rstride=1,
            quantize=0
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
        self.quantize = quantize
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input):
        x = torch.unsqueeze(input, 1)
        x_per = x.permute(0, 3, 2, 1)

        y_left = F.pad(x_per, [0, 0, (self.lorder - 1) * self.lstride, 0])
        if self.quantize:
            y_left = self.quant(y_left)
        y_left = self.conv_left(y_left)
        if self.quantize:
            y_left = self.dequant(y_left)
        out = x_per + y_left

        if self.conv_right is not None:
            y_right = F.pad(x_per, [0, 0, 0, self.rorder * self.rstride])
            y_right = y_right[:, :, self.rstride:, :]
            if self.quantize:
                y_right = self.quant(y_right)
            y_right = self.conv_right(y_right)
            if self.quantize:
                y_right = self.dequant(y_right)
            out += y_right

        out_per = out.permute(0, 3, 2, 1)
        output = out_per.squeeze(1)

        return output


class RectifiedLinear(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(RectifiedLinear, self).__init__()
        self.dim = input_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        out = self.relu(input)
        # out = self.dropout(out)
        return out


def _build_repeats(
        fsmn_layers: int,
        linear_dim: int,
        proj_dim: int,
        lorder: int,
        rorder: int,
        lstride=1,
        rstride=1,
):
    repeats = [
        nn.Sequential(
            LinearTransform(linear_dim, proj_dim),
            FSMNBlock(proj_dim, proj_dim, lorder, rorder, 1, 1),
            AffineTransform(proj_dim, linear_dim),
            RectifiedLinear(linear_dim, linear_dim))
        for i in range(fsmn_layers)
    ]

    return nn.Sequential(*repeats)


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
            output_dim: int,
    ):
        super(FSMN, self).__init__()

        self.input_dim = input_dim
        self.input_affine_dim = input_affine_dim
        self.fsmn_layers = fsmn_layers
        self.linear_dim = linear_dim
        self.proj_dim = proj_dim
        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride
        self.output_affine_dim = output_affine_dim
        self.output_dim = output_dim

        self.in_linear1 = AffineTransform(input_dim, input_affine_dim)
        self.in_linear2 = AffineTransform(input_affine_dim, linear_dim)
        self.relu = RectifiedLinear(linear_dim, linear_dim)

        self.fsmn = _build_repeats(fsmn_layers,
                                   linear_dim,
                                   proj_dim,
                                   lorder, rorder,
                                   lstride, rstride)

        self.out_linear1 = AffineTransform(linear_dim, output_affine_dim)
        self.out_linear2 = AffineTransform(output_affine_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def fuse_modules(self):
        pass

    def forward(
            self,
            input: torch.Tensor,
            in_cache: torch.Tensor = torch.zeros(0, 0, 0, dtype=torch.float)
    ) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Input tensor (B, T, D)
            in_cache(torhc.Tensor): (B, D, C), C is the accumulated cache size
        """

        x1 = self.in_linear1(input)
        x2 = self.in_linear2(x1)
        x3 = self.relu(x2)
        x4 = self.fsmn(x3)
        x5 = self.out_linear1(x4)
        x6 = self.out_linear2(x5)
        x7 = self.softmax(x6)

        return x7
        # return x6, in_cache


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
