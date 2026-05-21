# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        """Forward pass for training.
        
            Args:
                ctx: TODO.
                x: TODO.
                scale: TODO.
            """
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        """Backward.
        
            Args:
                ctx: TODO.
                grad: TODO.
            """
        return grad * ctx.scale, None
