import torch
from torch.nn import functional as F
from funasr.models.encoder.abs_encoder import AbsEncoder
from typing import Tuple


class BasicLayer(torch.nn.Module):

    def __init__(self, in_filters: int, filters: int, stride: int, bn_momentum: float = 0.5):

        super().__init__()
        self.stride = stride
        self.in_filters = in_filters
        self.filters = filters

        self.bn1 = torch.nn.BatchNorm2d(in_filters, eps=1e-3, momentum=bn_momentum, affine=True)
        self.relu1 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(in_filters, filters, 3, stride, bias=False)

        self.bn2 = torch.nn.BatchNorm2d(filters, eps=1e-3, momentum=bn_momentum, affine=True)
        self.relu2 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(filters, filters, 3, 1, bias=False)

        if in_filters != filters or stride > 1:
            self.conv_sc = torch.nn.Conv2d(in_filters, filters, 1, stride, bias=False)
            self.bn_sc = torch.nn.BatchNorm2d(filters, eps=1e-3, momentum=bn_momentum, affine=True)

    def proper_padding(self, x, stride):
        # align padding mode to tf.layers.conv2d with padding_mod="same"
        if stride == 1:
            return F.pad(x, (1, 1, 1, 1), "constant", 0)
        elif stride == 2:
            h, w = x.size(2), x.size(3)
            # (left, right, top, bottom)
            return F.pad(x, (w % 2, 1, h % 2, 1), "constant", 0)

    def forward(self, xs_pad, ilens):
        identity = xs_pad
        if self.in_filters != self.filters or self.stride > 1:
            identity = self.conv_sc(identity)
            identity = self.bn_sc(identity)

        xs_pad = self.relu1(self.bn1(xs_pad))
        xs_pad = self.proper_padding(xs_pad, self.stride)
        xs_pad = self.conv1(xs_pad)

        xs_pad = self.relu2(self.bn2(xs_pad))
        xs_pad = self.proper_padding(xs_pad, 1)
        xs_pad = self.conv2(xs_pad)

        if self.stride == 2:
            ilens = (ilens + 1) // self.stride

        return xs_pad + identity, ilens


class BasicBlock(torch.nn.Module):
    def __init__(self, in_filters, filters, num_layer, stride, bn_momentum=0.5):
        super().__init__()
        self.num_layer = num_layer

        for i in range(num_layer):
            layer = BasicLayer(in_filters if i == 0 else filters, filters,
                               stride if i == 0 else 1, bn_momentum)
            self.add_module("layer_{}".format(i), layer)

    def forward(self, xs_pad, ilens):

        for i in range(self.num_layer):
            xs_pad, ilens = self._modules["layer_{}".format(i)](xs_pad, ilens)

        return xs_pad, ilens


class ResNet34(AbsEncoder):
    def __init__(
            self,
            input_size,
            use_head_conv=True,
            batchnorm_momentum=0.5,
            use_head_maxpool=False,
            num_nodes_pooling_layer=256,
            layers_in_block=(3, 4, 6, 3),
            filters_in_block=(32, 64, 128, 256),
    ):
        super(ResNet34, self).__init__()

        self.use_head_conv = use_head_conv
        self.use_head_maxpool = use_head_maxpool
        self.num_nodes_pooling_layer = num_nodes_pooling_layer
        self.layers_in_block = layers_in_block
        self.filters_in_block = filters_in_block
        self.input_size = input_size

        pre_filters = filters_in_block[0]
        if use_head_conv:
            self.pre_conv = torch.nn.Conv2d(1, pre_filters, 3, 1, 1, bias=False, padding_mode="zeros")
            self.pre_conv_bn = torch.nn.BatchNorm2d(pre_filters, eps=1e-3, momentum=batchnorm_momentum)

        if use_head_maxpool:
            self.head_maxpool = torch.nn.MaxPool2d(3, 1, padding=1)

        for i in range(len(layers_in_block)):
            if i == 0:
                in_filters = pre_filters if self.use_head_conv else 1
            else:
                in_filters = filters_in_block[i-1]

            block = BasicBlock(in_filters,
                               filters=filters_in_block[i],
                               num_layer=layers_in_block[i],
                               stride=1 if i == 0 else 2,
                               bn_momentum=batchnorm_momentum)
            self.add_module("block_{}".format(i), block)

        self.resnet0_dense = torch.nn.Conv2d(filters_in_block[-1], num_nodes_pooling_layer, 1)
        self.resnet0_bn = torch.nn.BatchNorm2d(num_nodes_pooling_layer, eps=1e-3, momentum=batchnorm_momentum)

    def output_size(self) -> int:
        return self.num_nodes_pooling_layer

    def forward(self, xs_pad: torch.Tensor, ilens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = xs_pad
        assert features.size(-1) == self.input_size, \
            "Dimension of features {} doesn't match the input_size {}.".format(features.size(-1), self.input_size)
        features = torch.unsqueeze(features, dim=1)
        if self.use_head_conv:
            features = self.pre_conv(features)
            features = self.pre_conv_bn(features)
            features = F.relu(features)

        if self.use_head_maxpool:
            features = self.head_maxpool(features)

        resnet_outs, resnet_out_lens = features, ilens
        for i in range(len(self.layers_in_block)):
            block = self._modules["block_{}".format(i)]
            resnet_outs, resnet_out_lens = block(resnet_outs, resnet_out_lens)

        features = self.resnet0_dense(resnet_outs)
        features = F.relu(features)
        features = self.resnet0_bn(features)

        return features, ilens // 8
