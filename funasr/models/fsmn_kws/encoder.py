from typing import Tuple, Dict
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from funasr.register import tables


def toKaldiMatrix(np_mat):
    np.set_printoptions(threshold=np.inf, linewidth=np.nan)
    out_str = str(np_mat)
    out_str = out_str.replace('[', '')
    out_str = out_str.replace(']', '')
    return '[ %s ]\n' % out_str


class LinearTransform(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, input):
        output = self.linear(input)

        return output

    def to_kaldi_net(self):
        re_str = ''
        re_str += '<LinearTransform> %d %d\n' % (self.output_dim,
                                                 self.input_dim)
        re_str += '<LearnRateCoef> 1\n'

        linear_weights = self.state_dict()['linear.weight']
        x = linear_weights.squeeze().numpy()
        re_str += toKaldiMatrix(x)

        return re_str

    def to_pytorch_net(self, fread):
        linear_line = fread.readline()
        linear_split = linear_line.strip().split()
        assert len(linear_split) == 3
        assert linear_split[0] == '<LinearTransform>'
        self.output_dim = int(linear_split[1])
        self.input_dim = int(linear_split[2])

        learn_rate_line = fread.readline()
        assert learn_rate_line.find('LearnRateCoef') != -1

        self.linear.reset_parameters()

        linear_weights = self.state_dict()['linear.weight']
        #print(linear_weights.shape)
        new_weights = torch.zeros((self.output_dim, self.input_dim),
                                  dtype=torch.float32)
        for i in range(self.output_dim):
            line = fread.readline()
            splits = line.strip().strip('\[\]').strip().split()
            assert len(splits) == self.input_dim
            cols = torch.tensor([float(item) for item in splits],
                                dtype=torch.float32)
            new_weights[i, :] = cols

        self.linear.weight.data = new_weights


class AffineTransform(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(AffineTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        output = self.linear(input)

        return output

    def to_kaldi_net(self):
        re_str = ''
        re_str += '<AffineTransform> %d %d\n' % (self.output_dim,
                                                 self.input_dim)
        re_str += '<LearnRateCoef> 1 <BiasLearnRateCoef> 1 <MaxNorm> 0\n'

        linear_weights = self.state_dict()['linear.weight']
        x = linear_weights.squeeze().numpy()
        re_str += toKaldiMatrix(x)

        linear_bias = self.state_dict()['linear.bias']
        x = linear_bias.squeeze().numpy()
        re_str += toKaldiMatrix(x)

        return re_str

    def to_pytorch_net(self, fread):
        affine_line = fread.readline()
        affine_split = affine_line.strip().split()
        assert len(affine_split) == 3
        assert affine_split[0] == '<AffineTransform>'
        self.output_dim = int(affine_split[1])
        self.input_dim = int(affine_split[2])
        print('AffineTransform output/input dim: %d %d' %
              (self.output_dim, self.input_dim))

        learn_rate_line = fread.readline()
        assert learn_rate_line.find('LearnRateCoef') != -1

        #linear_weights = self.state_dict()['linear.weight']
        #print(linear_weights.shape)
        self.linear.reset_parameters()

        new_weights = torch.zeros((self.output_dim, self.input_dim),
                                  dtype=torch.float32)
        for i in range(self.output_dim):
            line = fread.readline()
            splits = line.strip().strip('\[\]').strip().split()
            assert len(splits) == self.input_dim
            cols = torch.tensor([float(item) for item in splits],
                                dtype=torch.float32)
            new_weights[i, :] = cols

        self.linear.weight.data = new_weights

        linear_bias = self.state_dict()['linear.bias']
        #print(linear_bias.shape)
        bias_line = fread.readline()
        splits = bias_line.strip().strip('\[\]').strip().split()
        assert len(splits) == self.output_dim
        new_bias = torch.tensor([float(item) for item in splits],
                                dtype=torch.float32)

        self.linear.bias.data = new_bias


class RectifiedLinear(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(RectifiedLinear, self).__init__()
        self.dim = input_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        out = self.relu(input)
        return out

    def to_kaldi_net(self):
        re_str = ''
        re_str += '<RectifiedLinear> %d %d\n' % (self.dim, self.dim)
        return re_str

    def to_pytorch_net(self, fread):
        line = fread.readline()
        splits = line.strip().split()
        assert len(splits) == 3
        assert splits[0] == '<RectifiedLinear>'
        assert int(splits[1]) == int(splits[2])
        assert int(splits[1]) == self.dim
        self.dim = int(splits[1])


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

    def to_kaldi_net(self):
        re_str = ''
        re_str += '<Fsmn> %d %d\n' % (self.dim, self.dim)
        re_str += '<LearnRateCoef> %d <LOrder> %d <ROrder> %d <LStride> %d <RStride> %d <MaxNorm> 0\n' % (
            1, self.lorder, self.rorder, self.lstride, self.rstride)

        #print(self.conv_left.weight,self.conv_right.weight)
        lfiters = self.state_dict()['conv_left.weight']
        x = np.flipud(lfiters.squeeze().numpy().T)
        re_str += toKaldiMatrix(x)

        if self.conv_right is not None:
            rfiters = self.state_dict()['conv_right.weight']
            x = (rfiters.squeeze().numpy().T)
            re_str += toKaldiMatrix(x)

        return re_str

    def to_pytorch_net(self, fread):
        fsmn_line = fread.readline()
        fsmn_split = fsmn_line.strip().split()
        assert len(fsmn_split) == 3
        assert fsmn_split[0] == '<Fsmn>'
        self.dim = int(fsmn_split[1])

        params_line = fread.readline()
        params_split = params_line.strip().strip('\[\]').strip().split()
        assert len(params_split) == 12
        assert params_split[0] == '<LearnRateCoef>'
        assert params_split[2] == '<LOrder>'
        self.lorder = int(params_split[3])
        assert params_split[4] == '<ROrder>'
        self.rorder = int(params_split[5])
        assert params_split[6] == '<LStride>'
        self.lstride = int(params_split[7])
        assert params_split[8] == '<RStride>'
        self.rstride = int(params_split[9])
        assert params_split[10] == '<MaxNorm>'

        #lfilters = self.state_dict()['conv_left.weight']
        #print(lfilters.shape)
        print('read conv_left weight')
        new_lfilters = torch.zeros((self.lorder, 1, self.dim, 1),
                                   dtype=torch.float32)
        for i in range(self.lorder):
            print('read conv_left weight -- %d' % i)
            line = fread.readline()
            splits = line.strip().strip('\[\]').strip().split()
            assert len(splits) == self.dim
            cols = torch.tensor([float(item) for item in splits],
                                dtype=torch.float32)
            new_lfilters[self.lorder - 1 - i, 0, :, 0] = cols

        new_lfilters = torch.transpose(new_lfilters, 0, 2)
        #print(new_lfilters.shape)

        self.conv_left.reset_parameters()
        self.conv_left.weight.data = new_lfilters
        #print(self.conv_left.weight.shape)

        if self.rorder > 0:
            #rfilters = self.state_dict()['conv_right.weight']
            #print(rfilters.shape)
            print('read conv_right weight')
            new_rfilters = torch.zeros((self.rorder, 1, self.dim, 1),
                                       dtype=torch.float32)
            line = fread.readline()
            for i in range(self.rorder):
                print('read conv_right weight -- %d' % i)
                line = fread.readline()
                splits = line.strip().strip('\[\]').strip().split()
                assert len(splits) == self.dim
                cols = torch.tensor([float(item) for item in splits],
                                    dtype=torch.float32)
                new_rfilters[i, 0, :, 0] = cols

            new_rfilters = torch.transpose(new_rfilters, 0, 2)
            #print(new_rfilters.shape)
            self.conv_right.reset_parameters()
            self.conv_right.weight.data = new_rfilters
            #print(self.conv_right.weight.shape)

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

    def to_kaldi_net(self):
        re_str = ''
        re_str += self.linear.to_kaldi_net()
        re_str += self.fsmn_block.to_kaldi_net()
        re_str += self.affine.to_kaldi_net()
        re_str += self.relu.to_kaldi_net()

        return re_str

    def to_pytorch_net(self, fread):
        self.linear.to_pytorch_net(fread)
        self.fsmn_block.to_pytorch_net(fread)
        self.affine.to_pytorch_net(fread)
        self.relu.to_pytorch_net(fread)


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

    def to_kaldi_net(self):
        re_str = ''
        for module in self._modules.values():
            re_str += module.to_kaldi_net()

        return re_str

    def to_pytorch_net(self, fread):
        for module in self._modules.values():
            module.to_pytorch_net(fread)


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


@tables.register("encoder_classes", "FSMNConvert")
class FSMNConvert(nn.Module):
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
