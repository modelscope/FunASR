from typing import Tuple, Dict
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from funasr.register import tables


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
            self.dim, self.dim, [lorder, 1], dilation=[lstride, 1], groups=self.dim, bias=False
        )

        if self.rorder > 0:
            self.conv_right = nn.Conv2d(
                self.dim, self.dim, [rorder, 1], dilation=[rstride, 1], groups=self.dim, bias=False
            )
        else:
            self.conv_right = None

    def forward(self, input: torch.Tensor, cache: torch.Tensor = None):
        x = torch.unsqueeze(input, 1)
        x_per = x.permute(0, 3, 2, 1)  # B D T C

        if cache is not None:
            cache = cache.to(x_per.device)
            y_left = torch.cat((cache, x_per), dim=2)
            cache = y_left[:, :, -(self.lorder - 1) * self.lstride :, :]
        else:
            y_left = F.pad(x_per, [0, 0, (self.lorder - 1) * self.lstride, 0])

        y_left = self.conv_left(y_left)
        out = x_per + y_left

        if self.conv_right is not None:
            # maybe need to check
            y_right = F.pad(x_per, [0, 0, 0, self.rorder * self.rstride])
            y_right = y_right[:, :, self.rstride :, :]
            y_right = self.conv_right(y_right)
            out += y_right

        out_per = out.permute(0, 3, 2, 1)
        output = out_per.squeeze(1)

        return output, cache


class BasicBlock(nn.Module):
    def __init__(
        self,
        linear_dim: int,
        proj_dim: int,
        lorder: int,
        rorder: int,
        lstride: int,
        rstride: int,
        stack_layer: int,
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

    def forward(self, input: torch.Tensor, cache: Dict[str, torch.Tensor] = None):
        x1 = self.linear(input)  # B T D

        if cache is not None:
            cache_layer_name = 'cache_layer_{}'.format(self.stack_layer)
            if cache_layer_name not in cache:
                cache[cache_layer_name] = torch.zeros(
                    x1.shape[0], x1.shape[-1], (self.lorder - 1) * self.lstride, 1
                )
            x2, cache[cache_layer_name] = self.fsmn_block(x1, cache[cache_layer_name])
        else:
            x2, _ = self.fsmn_block(x1, None)
        x3 = self.affine(x2)
        x4 = self.relu(x3)
        return x4


class BasicBlock_export(nn.Module):
    def __init__(
        self,
        model,
    ):
        super(BasicBlock_export, self).__init__()
        self.linear = model.linear
        self.fsmn_block = model.fsmn_block
        self.affine = model.affine
        self.relu = model.relu

    def forward(self, input: torch.Tensor, in_cache: torch.Tensor):
        x = self.linear(input)  # B T D
        # cache_layer_name = 'cache_layer_{}'.format(self.stack_layer)
        # if cache_layer_name not in in_cache:
        #     in_cache[cache_layer_name] = torch.zeros(x1.shape[0], x1.shape[-1], (self.lorder - 1) * self.lstride, 1)
        x, out_cache = self.fsmn_block(x, in_cache)
        x = self.affine(x)
        x = self.relu(x)
        return x, out_cache


class FsmnStack(nn.Sequential):
    def __init__(self, *args):
        super(FsmnStack, self).__init__(*args)

    def forward(self, input: torch.Tensor, cache: Dict[str, torch.Tensor]):
        x = input
        for module in self._modules.values():
            x = module(x, cache)
        return x


"""
FSMN net for keyword spotting
input_dim:              input dimension
linear_dim:             fsmn input dimensionll
proj_dim:               fsmn projection dimension
lorder:                 fsmn left order
rorder:                 fsmn right order
num_syn:                output dimension
fsmn_layers:            no. of sequential fsmn layers
"""


@tables.register("encoder_classes", "FSMN")
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

        self.in_linear1 = AffineTransform(input_dim, input_affine_dim)
        self.in_linear2 = AffineTransform(input_affine_dim, linear_dim)
        self.relu = RectifiedLinear(linear_dim, linear_dim)
        self.fsmn = FsmnStack(
            *[
                BasicBlock(linear_dim, proj_dim, lorder, rorder, lstride, rstride, i)
                for i in range(fsmn_layers)
            ]
        )
        self.out_linear1 = AffineTransform(linear_dim, output_affine_dim)
        self.out_linear2 = AffineTransform(output_affine_dim, output_dim)

        self.use_softmax = use_softmax
        if self.use_softmax:
            self.softmax = nn.Softmax(dim=-1)

    def fuse_modules(self):
        pass

    def output_size(self) -> int:
        return self.output_dim

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

        if self.use_softmax:
            x7 = self.softmax(x6)
            return x7

        return x6


@tables.register("encoder_classes", "FSMNExport")
class FSMNExport(nn.Module):
    def __init__(
        self,
        model,
        **kwargs,
    ):
        super().__init__()

        # self.input_dim = input_dim
        # self.input_affine_dim = input_affine_dim
        # self.fsmn_layers = fsmn_layers
        # self.linear_dim = linear_dim
        # self.proj_dim = proj_dim
        # self.output_affine_dim = output_affine_dim
        # self.output_dim = output_dim
        #
        # self.in_linear1 = AffineTransform(input_dim, input_affine_dim)
        # self.in_linear2 = AffineTransform(input_affine_dim, linear_dim)
        # self.relu = RectifiedLinear(linear_dim, linear_dim)
        # self.fsmn = FsmnStack(*[BasicBlock(linear_dim, proj_dim, lorder, rorder, lstride, rstride, i) for i in
        #                         range(fsmn_layers)])
        # self.out_linear1 = AffineTransform(linear_dim, output_affine_dim)
        # self.out_linear2 = AffineTransform(output_affine_dim, output_dim)
        # self.softmax = nn.Softmax(dim=-1)

        self.in_linear1 = model.in_linear1
        self.in_linear2 = model.in_linear2
        self.relu = model.relu
        # self.fsmn = model.fsmn
        self.out_linear1 = model.out_linear1
        self.out_linear2 = model.out_linear2
        self.softmax = model.softmax
        self.fsmn = model.fsmn
        for i, d in enumerate(model.fsmn):
            if isinstance(d, BasicBlock):
                self.fsmn[i] = BasicBlock_export(d)

    def fuse_modules(self):
        pass

    def forward(
        self,
        input: torch.Tensor,
        *args,
    ):
        """
        Args:
            input (torch.Tensor): Input tensor (B, T, D)
            in_cache: when in_cache is not None, the forward is in streaming. The type of in_cache is a dict, egs,
            {'cache_layer_1': torch.Tensor(B, T1, D)}, T1 is equal to self.lorder. It is {} for the 1st frame
        """

        x = self.in_linear1(input)
        x = self.in_linear2(x)
        x = self.relu(x)
        # x4 = self.fsmn(x3, in_cache)  # self.in_cache will update automatically in self.fsmn
        out_caches = list()
        for i, d in enumerate(self.fsmn):
            in_cache = args[i]
            x, out_cache = d(x, in_cache)
            out_caches.append(out_cache)
        x = self.out_linear1(x)
        x = self.out_linear2(x)
        x = self.softmax(x)

        return x, out_caches
