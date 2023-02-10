from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from typeguard import check_argument_types
import numpy as np
from funasr.modules.nets_utils import make_pad_mask
from funasr.modules.layer_norm import LayerNorm
from funasr.models.encoder.abs_encoder import AbsEncoder
import math
from funasr.modules.repeat import repeat
from funasr.modules.multi_layer_conv import FsmnFeedForward


class FsmnBlock(torch.nn.Module):
    def __init__(
            self,
            n_feat,
            dropout_rate,
            kernel_size,
            fsmn_shift=0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fsmn_block = nn.Conv1d(n_feat, n_feat, kernel_size, stride=1,
                                    padding=0, groups=n_feat, bias=False)
        # padding
        left_padding = (kernel_size - 1) // 2
        if fsmn_shift > 0:
            left_padding = left_padding + fsmn_shift
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward(self, inputs, mask, mask_shfit_chunk=None):
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk

        inputs = inputs * mask
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x = x + inputs
        x = self.dropout(x)
        return x * mask


class EncoderLayer(torch.nn.Module):
    def __init__(
            self,
            in_size,
            size,
            feed_forward,
            fsmn_block,
            dropout_rate=0.0
    ):
        super().__init__()
        self.in_size = in_size
        self.size = size
        self.ffn = feed_forward
        self.memory = fsmn_block
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
            self,
            xs_pad: torch.Tensor,
            mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # xs_pad in Batch, Time, Dim

        context = self.ffn(xs_pad)[0]
        memory = self.memory(context, mask)

        memory = self.dropout(memory)
        if self.in_size == self.size:
            return memory + xs_pad, mask

        return memory, mask


class FsmnEncoder(AbsEncoder):
    """Encoder using Fsmn
      """

    def __init__(self,
                 in_units,
                 filter_size,
                 fsmn_num_layers,
                 dnn_num_layers,
                 num_memory_units=512,
                 ffn_inner_dim=2048,
                 dropout_rate=0.0,
                 shift=0,
                 position_encoder=None,
                 sample_rate=1,
                 out_units=None,
                 tf2torch_tensor_name_prefix_torch="post_net",
                 tf2torch_tensor_name_prefix_tf="EAND/post_net"
                 ):
        """Initializes the parameters of the encoder.

        Args:
          filter_size: the total order of memory block
          fsmn_num_layers: The number of fsmn layers.
          dnn_num_layers: The number of dnn layers
          num_units: The number of memory units.
          ffn_inner_dim: The number of units of the inner linear transformation
            in the feed forward layer.
          dropout_rate: The probability to drop units from the outputs.
          shift: left padding, to control delay
          position_encoder: The :class:`opennmt.layers.position.PositionEncoder` to
            apply on inputs or ``None``.
        """
        super(FsmnEncoder, self).__init__()
        self.in_units = in_units
        self.filter_size = filter_size
        self.fsmn_num_layers = fsmn_num_layers
        self.dnn_num_layers = dnn_num_layers
        self.num_memory_units = num_memory_units
        self.ffn_inner_dim = ffn_inner_dim
        self.dropout_rate = dropout_rate
        self.shift = shift
        if not isinstance(shift, list):
            self.shift = [shift for _ in range(self.fsmn_num_layers)]
        self.sample_rate = sample_rate
        if not isinstance(sample_rate, list):
            self.sample_rate = [sample_rate for _ in range(self.fsmn_num_layers)]
        self.position_encoder = position_encoder
        self.dropout = nn.Dropout(dropout_rate)
        self.out_units = out_units
        self.tf2torch_tensor_name_prefix_torch = tf2torch_tensor_name_prefix_torch
        self.tf2torch_tensor_name_prefix_tf = tf2torch_tensor_name_prefix_tf

        self.fsmn_layers = repeat(
            self.fsmn_num_layers,
            lambda lnum: EncoderLayer(
                in_units if lnum == 0 else num_memory_units,
                num_memory_units,
                FsmnFeedForward(
                    in_units if lnum == 0 else num_memory_units,
                    ffn_inner_dim,
                    num_memory_units,
                    1,
                    dropout_rate
                ),
                FsmnBlock(
                    num_memory_units,
                    dropout_rate,
                    filter_size,
                    self.shift[lnum]
                )
            ),
        )

        self.dnn_layers = repeat(
            dnn_num_layers,
            lambda lnum: FsmnFeedForward(
                num_memory_units,
                ffn_inner_dim,
                num_memory_units,
                1,
                dropout_rate,
            )
        )
        if out_units is not None:
            self.conv1d = nn.Conv1d(num_memory_units, out_units, 1, 1)

    def output_size(self) -> int:
        return self.num_memory_units

    def forward(
            self,
            xs_pad: torch.Tensor,
            ilens: torch.Tensor,
            prev_states: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        inputs = xs_pad
        if self.position_encoder is not None:
            inputs = self.position_encoder(inputs)

        inputs = self.dropout(inputs)
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        inputs = self.fsmn_layers(inputs, masks)[0]
        inputs = self.dnn_layers(inputs)[0]

        if self.out_units is not None:
            inputs = self.conv1d(inputs.transpose(1, 2)).transpose(1, 2)

        return inputs, ilens, None

    def gen_tf2torch_map_dict(self):
        tensor_name_prefix_torch = self.tf2torch_tensor_name_prefix_torch
        tensor_name_prefix_tf = self.tf2torch_tensor_name_prefix_tf
        map_dict_local = {
            # torch: conv1d.weight in "out_channel in_channel kernel_size"
            # tf   : conv1d.weight in "kernel_size in_channel out_channel"
            # torch: linear.weight in "out_channel in_channel"
            # tf   :  dense.weight in "in_channel out_channel"
            # for fsmn_layers
            "{}.fsmn_layers.layeridx.ffn.norm.bias".format(tensor_name_prefix_torch):
                {"name": "{}/fsmn_layer_layeridx/ffn/LayerNorm/beta".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },
            "{}.fsmn_layers.layeridx.ffn.norm.weight".format(tensor_name_prefix_torch):
                {"name": "{}/fsmn_layer_layeridx/ffn/LayerNorm/gamma".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },
            "{}.fsmn_layers.layeridx.ffn.w_1.bias".format(tensor_name_prefix_torch):
                {"name": "{}/fsmn_layer_layeridx/ffn/conv1d/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },
            "{}.fsmn_layers.layeridx.ffn.w_1.weight".format(tensor_name_prefix_torch):
                {"name": "{}/fsmn_layer_layeridx/ffn/conv1d/kernel".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": (2, 1, 0),
                 },
            "{}.fsmn_layers.layeridx.ffn.w_2.weight".format(tensor_name_prefix_torch):
                {"name": "{}/fsmn_layer_layeridx/ffn/conv1d_1/kernel".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": (2, 1, 0),
                 },
            "{}.fsmn_layers.layeridx.memory.fsmn_block.weight".format(tensor_name_prefix_torch):
                {"name": "{}/fsmn_layer_layeridx/memory/depth_conv_w".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 2, 0),
                 },  # (1, 31, 512, 1) -> (31, 512, 1) -> (512, 1, 31)

            # for dnn_layers
            "{}.dnn_layers.layeridx.norm.bias".format(tensor_name_prefix_torch):
                {"name": "{}/dnn_layer_layeridx/LayerNorm/beta".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },
            "{}.dnn_layers.layeridx.norm.weight".format(tensor_name_prefix_torch):
                {"name": "{}/dnn_layer_layeridx/LayerNorm/gamma".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },
            "{}.dnn_layers.layeridx.w_1.bias".format(tensor_name_prefix_torch):
                {"name": "{}/dnn_layer_layeridx/conv1d/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },
            "{}.dnn_layers.layeridx.w_1.weight".format(tensor_name_prefix_torch):
                {"name": "{}/dnn_layer_layeridx/conv1d/kernel".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": (2, 1, 0),
                 },
            "{}.dnn_layers.layeridx.w_2.weight".format(tensor_name_prefix_torch):
                {"name": "{}/dnn_layer_layeridx/conv1d_1/kernel".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": (2, 1, 0),
                 },

        }
        if self.out_units is not None:
            # add output layer
            map_dict_local.update({
                "{}.conv1d.weight".format(tensor_name_prefix_torch):
                    {"name": "{}/conv1d/kernel".format(tensor_name_prefix_tf),
                     "squeeze": None,
                     "transpose": (2, 1, 0),
                     },
                "{}.conv1d.bias".format(tensor_name_prefix_torch):
                    {"name": "{}/conv1d/bias".format(tensor_name_prefix_tf),
                     "squeeze": None,
                     "transpose": None,
                     },
            })

        return map_dict_local

    def convert_tf2torch(self,
                         var_dict_tf,
                         var_dict_torch,
                         ):

        map_dict = self.gen_tf2torch_map_dict()

        var_dict_torch_update = dict()
        for name in sorted(var_dict_torch.keys(), reverse=False):
            if name.startswith(self.tf2torch_tensor_name_prefix_torch):
                # process special (first and last) layers
                if name in map_dict:
                    name_tf = map_dict[name]["name"]
                    data_tf = var_dict_tf[name_tf]
                    if map_dict[name]["squeeze"] is not None:
                        data_tf = np.squeeze(data_tf, axis=map_dict[name]["squeeze"])
                    if map_dict[name]["transpose"] is not None:
                        data_tf = np.transpose(data_tf, map_dict[name]["transpose"])
                    data_tf = torch.from_numpy(data_tf).type(torch.float32).to("cpu")
                    assert var_dict_torch[name].size() == data_tf.size(), \
                        "{}, {}, {} != {}".format(name, name_tf,
                                                  var_dict_torch[name].size(), data_tf.size())
                    var_dict_torch_update[name] = data_tf
                    logging.info("torch tensor: {}, {}, loading from tf tensor: {}, {}".format(
                        name, data_tf.size(), name_tf, var_dict_tf[name_tf].shape
                    ))
                # process general layers
                else:
                    # self.tf2torch_tensor_name_prefix_torch may include ".", solve this case
                    names = name.replace(self.tf2torch_tensor_name_prefix_torch, "todo").split('.')
                    layeridx = int(names[2])
                    name_q = name.replace(".{}.".format(layeridx), ".layeridx.")
                    if name_q in map_dict.keys():
                        name_v = map_dict[name_q]["name"]
                        name_tf = name_v.replace("layeridx", "{}".format(layeridx))
                        data_tf = var_dict_tf[name_tf]
                        if map_dict[name_q]["squeeze"] is not None:
                            data_tf = np.squeeze(data_tf, axis=map_dict[name_q]["squeeze"])
                        if map_dict[name_q]["transpose"] is not None:
                            data_tf = np.transpose(data_tf, map_dict[name_q]["transpose"])
                        data_tf = torch.from_numpy(data_tf).type(torch.float32).to("cpu")
                        assert var_dict_torch[name].size() == data_tf.size(), \
                            "{}, {}, {} != {}".format(name, name_tf,
                                                      var_dict_torch[name].size(), data_tf.size())
                        var_dict_torch_update[name] = data_tf
                        logging.info("torch tensor: {}, {}, loading from tf tensor: {}, {}".format(
                            name, data_tf.size(), name_tf, var_dict_tf[name_tf].shape
                        ))
                    else:
                        logging.warning("{} is missed from tf checkpoint".format(name))

        return var_dict_torch_update
