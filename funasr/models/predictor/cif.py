import torch
from torch import nn
from torch import Tensor
import logging
import numpy as np
from funasr.torch_utils.device_funcs import to_device
from funasr.modules.nets_utils import make_pad_mask
from funasr.modules.streaming_utils.utils import sequence_mask
from typing import Optional, Tuple

class CifPredictor(nn.Module):
    def __init__(self, idim, l_order, r_order, threshold=1.0, dropout=0.1, smooth_factor=1.0, noise_threshold=0, tail_threshold=0.45):
        super(CifPredictor, self).__init__()

        self.pad = nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = nn.Conv1d(idim, idim, l_order + r_order + 1, groups=idim)
        self.cif_output = nn.Linear(idim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.threshold = threshold
        self.smooth_factor = smooth_factor
        self.noise_threshold = noise_threshold
        self.tail_threshold = tail_threshold

    def forward(self, hidden, target_label=None, mask=None, ignore_id=-1, mask_chunk_predictor=None,
                target_label_length=None):
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        memory = self.cif_conv1d(queries)
        output = memory + context
        output = self.dropout(output)
        output = output.transpose(1, 2)
        output = torch.relu(output)
        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(alphas * self.smooth_factor - self.noise_threshold)
        if mask is not None:
            mask = mask.transpose(-1, -2).float()
            alphas = alphas * mask
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor
        alphas = alphas.squeeze(-1)
        mask = mask.squeeze(-1)
        if target_label_length is not None:
            target_length = target_label_length
        elif target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None
        token_num = alphas.sum(-1)
        if target_length is not None:
            alphas *= (target_length / token_num)[:, None].repeat(1, alphas.size(1))
        elif self.tail_threshold > 0.0:
            hidden, alphas, token_num = self.tail_process_fn(hidden, alphas, token_num, mask=mask)
            
        acoustic_embeds, cif_peak = cif(hidden, alphas, self.threshold)
        
        if target_length is None and self.tail_threshold > 0.0:
            token_num_int = torch.max(token_num).type(torch.int32).item()
            acoustic_embeds = acoustic_embeds[:, :token_num_int, :]
            
        return acoustic_embeds, token_num, alphas, cif_peak

    def tail_process_fn(self, hidden, alphas, token_num=None, mask=None):
        b, t, d = hidden.size()
        tail_threshold = self.tail_threshold
        if mask is not None:
            zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
            ones_t = torch.ones_like(zeros_t)
            mask_1 = torch.cat([mask, zeros_t], dim=1)
            mask_2 = torch.cat([ones_t, mask], dim=1)
            mask = mask_2 - mask_1
            tail_threshold = mask * tail_threshold
            alphas = torch.cat([alphas, zeros_t], dim=1)
            alphas = torch.add(alphas, tail_threshold)
        else:
            tail_threshold = torch.tensor([tail_threshold], dtype=alphas.dtype).to(alphas.device)
            tail_threshold = torch.reshape(tail_threshold, (1, 1))
            alphas = torch.cat([alphas, tail_threshold], dim=1)
        zeros = torch.zeros((b, 1, d), dtype=hidden.dtype).to(hidden.device)
        hidden = torch.cat([hidden, zeros], dim=1)
        token_num = alphas.sum(dim=-1)
        token_num_floor = torch.floor(token_num)

        return hidden, alphas, token_num_floor


    def gen_frame_alignments(self,
                             alphas: torch.Tensor = None,
                             encoder_sequence_length: torch.Tensor = None):
        batch_size, maximum_length = alphas.size()
        int_type = torch.int32

        is_training = self.training
        if is_training:
            token_num = torch.round(torch.sum(alphas, dim=1)).type(int_type)
        else:
            token_num = torch.floor(torch.sum(alphas, dim=1)).type(int_type)

        max_token_num = torch.max(token_num).item()

        alphas_cumsum = torch.cumsum(alphas, dim=1)
        alphas_cumsum = torch.floor(alphas_cumsum).type(int_type)
        alphas_cumsum = alphas_cumsum[:, None, :].repeat(1, max_token_num, 1)

        index = torch.ones([batch_size, max_token_num], dtype=int_type)
        index = torch.cumsum(index, dim=1)
        index = index[:, :, None].repeat(1, 1, maximum_length).to(alphas_cumsum.device)

        index_div = torch.floor(torch.true_divide(alphas_cumsum, index)).type(int_type)
        index_div_bool_zeros = index_div.eq(0)
        index_div_bool_zeros_count = torch.sum(index_div_bool_zeros, dim=-1) + 1
        index_div_bool_zeros_count = torch.clamp(index_div_bool_zeros_count, 0, encoder_sequence_length.max())
        token_num_mask = (~make_pad_mask(token_num, maxlen=max_token_num)).to(token_num.device)
        index_div_bool_zeros_count *= token_num_mask

        index_div_bool_zeros_count_tile = index_div_bool_zeros_count[:, :, None].repeat(1, 1, maximum_length)
        ones = torch.ones_like(index_div_bool_zeros_count_tile)
        zeros = torch.zeros_like(index_div_bool_zeros_count_tile)
        ones = torch.cumsum(ones, dim=2)
        cond = index_div_bool_zeros_count_tile == ones
        index_div_bool_zeros_count_tile = torch.where(cond, zeros, ones)

        index_div_bool_zeros_count_tile_bool = index_div_bool_zeros_count_tile.type(torch.bool)
        index_div_bool_zeros_count_tile = 1 - index_div_bool_zeros_count_tile_bool.type(int_type)
        index_div_bool_zeros_count_tile_out = torch.sum(index_div_bool_zeros_count_tile, dim=1)
        index_div_bool_zeros_count_tile_out = index_div_bool_zeros_count_tile_out.type(int_type)
        predictor_mask = (~make_pad_mask(encoder_sequence_length, maxlen=encoder_sequence_length.max())).type(
            int_type).to(encoder_sequence_length.device)
        index_div_bool_zeros_count_tile_out = index_div_bool_zeros_count_tile_out * predictor_mask

        predictor_alignments = index_div_bool_zeros_count_tile_out
        predictor_alignments_length = predictor_alignments.sum(-1).type(encoder_sequence_length.dtype)
        return predictor_alignments.detach(), predictor_alignments_length.detach()


class CifPredictorV2(nn.Module):
    def __init__(self,
                 idim,
                 l_order,
                 r_order,
                 threshold=1.0,
                 dropout=0.1,
                 smooth_factor=1.0,
                 noise_threshold=0,
                 tail_threshold=0.0,
                 tf2torch_tensor_name_prefix_torch="predictor",
                 tf2torch_tensor_name_prefix_tf="seq2seq/cif",
                 tail_mask=True,
                 ):
        super(CifPredictorV2, self).__init__()

        self.pad = nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = nn.Conv1d(idim, idim, l_order + r_order + 1)
        self.cif_output = nn.Linear(idim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.threshold = threshold
        self.smooth_factor = smooth_factor
        self.noise_threshold = noise_threshold
        self.tail_threshold = tail_threshold
        self.tf2torch_tensor_name_prefix_torch = tf2torch_tensor_name_prefix_torch
        self.tf2torch_tensor_name_prefix_tf = tf2torch_tensor_name_prefix_tf
        self.tail_mask = tail_mask

    def forward(self, hidden, target_label=None, mask=None, ignore_id=-1, mask_chunk_predictor=None,
                target_label_length=None):
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        output = torch.relu(self.cif_conv1d(queries))
        output = output.transpose(1, 2)

        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(alphas * self.smooth_factor - self.noise_threshold)
        if mask is not None:
            mask = mask.transpose(-1, -2).float()
            alphas = alphas * mask
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor
        alphas = alphas.squeeze(-1)
        mask = mask.squeeze(-1)
        if target_label_length is not None:
            target_length = target_label_length
        elif target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None
        token_num = alphas.sum(-1)
        if target_length is not None:
            alphas *= (target_length / token_num)[:, None].repeat(1, alphas.size(1))
        elif self.tail_threshold > 0.0:
            if self.tail_mask:
                hidden, alphas, token_num = self.tail_process_fn(hidden, alphas, token_num, mask=mask)
            else:
                hidden, alphas, token_num = self.tail_process_fn(hidden, alphas, token_num, mask=None)

        acoustic_embeds, cif_peak = cif(hidden, alphas, self.threshold)
        if target_length is None and self.tail_threshold > 0.0:
            token_num_int = torch.max(token_num).type(torch.int32).item()
            acoustic_embeds = acoustic_embeds[:, :token_num_int, :]

        return acoustic_embeds, token_num, alphas, cif_peak

    def forward_chunk(self, hidden, cache=None):
        batch_size, len_time, hidden_size = hidden.shape
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        output = torch.relu(self.cif_conv1d(queries))
        output = output.transpose(1, 2)
        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(alphas * self.smooth_factor - self.noise_threshold)

        alphas = alphas.squeeze(-1)

        token_length = []
        list_fires = []
        list_frames = []
        cache_alphas = []
        cache_hiddens = []

        if cache is not None and "chunk_size" in cache:
            alphas[:, :cache["chunk_size"][0]] = 0.0
            if "is_final" in cache and not cache["is_final"]:
                alphas[:, sum(cache["chunk_size"][:2]):] = 0.0
        if cache is not None and "cif_alphas" in cache and "cif_hidden" in cache:
            cache["cif_hidden"] = to_device(cache["cif_hidden"], device=hidden.device)
            cache["cif_alphas"] = to_device(cache["cif_alphas"], device=alphas.device)
            hidden = torch.cat((cache["cif_hidden"], hidden), dim=1)
            alphas = torch.cat((cache["cif_alphas"], alphas), dim=1)
        if cache is not None and "is_final" in cache and cache["is_final"]:
            tail_hidden = torch.zeros((batch_size, 1, hidden_size), device=hidden.device)
            tail_alphas = torch.tensor([[self.tail_threshold]], device=alphas.device)
            tail_alphas = torch.tile(tail_alphas, (batch_size, 1))
            hidden = torch.cat((hidden, tail_hidden), dim=1)
            alphas = torch.cat((alphas, tail_alphas), dim=1)

        len_time = alphas.shape[1]
        for b in range(batch_size):
            integrate = 0.0
            frames = torch.zeros((hidden_size), device=hidden.device)
            list_frame = []
            list_fire = []
            for t in range(len_time):
                alpha = alphas[b][t]
                if alpha + integrate < self.threshold:
                    integrate += alpha
                    list_fire.append(integrate)
                    frames += alpha * hidden[b][t]
                else:
                    frames += (self.threshold - integrate) * hidden[b][t]
                    list_frame.append(frames)
                    integrate += alpha
                    list_fire.append(integrate)
                    integrate -= self.threshold
                    frames = integrate * hidden[b][t]

            cache_alphas.append(integrate)
            if integrate > 0.0:
                cache_hiddens.append(frames / integrate)
            else:
                cache_hiddens.append(frames)

            token_length.append(torch.tensor(len(list_frame), device=alphas.device))
            list_fires.append(list_fire)
            list_frames.append(list_frame)

        cache["cif_alphas"] = torch.stack(cache_alphas, axis=0)
        cache["cif_alphas"] = torch.unsqueeze(cache["cif_alphas"], axis=0)
        cache["cif_hidden"] = torch.stack(cache_hiddens, axis=0)
        cache["cif_hidden"] = torch.unsqueeze(cache["cif_hidden"], axis=0)

        max_token_len = max(token_length)
        if max_token_len == 0:
             return hidden, torch.stack(token_length, 0)
        list_ls = []
        for b in range(batch_size):
            pad_frames = torch.zeros((max_token_len - token_length[b], hidden_size), device=alphas.device)
            if token_length[b] == 0:
                list_ls.append(pad_frames)
            else:
                list_frames[b] = torch.stack(list_frames[b])
                list_ls.append(torch.cat((list_frames[b], pad_frames), dim=0))

        cache["cif_alphas"] = torch.stack(cache_alphas, axis=0)
        cache["cif_alphas"] = torch.unsqueeze(cache["cif_alphas"], axis=0)
        cache["cif_hidden"] = torch.stack(cache_hiddens, axis=0)
        cache["cif_hidden"] = torch.unsqueeze(cache["cif_hidden"], axis=0)
        return torch.stack(list_ls, 0), torch.stack(token_length, 0)


    def tail_process_fn(self, hidden, alphas, token_num=None, mask=None):
        b, t, d = hidden.size()
        tail_threshold = self.tail_threshold
        if mask is not None:
            zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
            ones_t = torch.ones_like(zeros_t)
            mask_1 = torch.cat([mask, zeros_t], dim=1)
            mask_2 = torch.cat([ones_t, mask], dim=1)
            mask = mask_2 - mask_1
            tail_threshold = mask * tail_threshold
            alphas = torch.cat([alphas, zeros_t], dim=1)
            alphas = torch.add(alphas, tail_threshold)
        else:
            tail_threshold = torch.tensor([tail_threshold], dtype=alphas.dtype).to(alphas.device)
            tail_threshold = torch.reshape(tail_threshold, (1, 1))
            if b > 1:
                alphas = torch.cat([alphas, tail_threshold.repeat(b, 1)], dim=1)
            else:
                alphas = torch.cat([alphas, tail_threshold], dim=1)
        zeros = torch.zeros((b, 1, d), dtype=hidden.dtype).to(hidden.device)
        hidden = torch.cat([hidden, zeros], dim=1)
        token_num = alphas.sum(dim=-1)
        token_num_floor = torch.floor(token_num)

        return hidden, alphas, token_num_floor

    def gen_frame_alignments(self,
                             alphas: torch.Tensor = None,
                             encoder_sequence_length: torch.Tensor = None):
        batch_size, maximum_length = alphas.size()
        int_type = torch.int32

        is_training = self.training
        if is_training:
            token_num = torch.round(torch.sum(alphas, dim=1)).type(int_type)
        else:
            token_num = torch.floor(torch.sum(alphas, dim=1)).type(int_type)

        max_token_num = torch.max(token_num).item()

        alphas_cumsum = torch.cumsum(alphas, dim=1)
        alphas_cumsum = torch.floor(alphas_cumsum).type(int_type)
        alphas_cumsum = alphas_cumsum[:, None, :].repeat(1, max_token_num, 1)

        index = torch.ones([batch_size, max_token_num], dtype=int_type)
        index = torch.cumsum(index, dim=1)
        index = index[:, :, None].repeat(1, 1, maximum_length).to(alphas_cumsum.device)

        index_div = torch.floor(torch.true_divide(alphas_cumsum, index)).type(int_type)
        index_div_bool_zeros = index_div.eq(0)
        index_div_bool_zeros_count = torch.sum(index_div_bool_zeros, dim=-1) + 1
        index_div_bool_zeros_count = torch.clamp(index_div_bool_zeros_count, 0, encoder_sequence_length.max())
        token_num_mask = (~make_pad_mask(token_num, maxlen=max_token_num)).to(token_num.device)
        index_div_bool_zeros_count *= token_num_mask

        index_div_bool_zeros_count_tile = index_div_bool_zeros_count[:, :, None].repeat(1, 1, maximum_length)
        ones = torch.ones_like(index_div_bool_zeros_count_tile)
        zeros = torch.zeros_like(index_div_bool_zeros_count_tile)
        ones = torch.cumsum(ones, dim=2)
        cond = index_div_bool_zeros_count_tile == ones
        index_div_bool_zeros_count_tile = torch.where(cond, zeros, ones)

        index_div_bool_zeros_count_tile_bool = index_div_bool_zeros_count_tile.type(torch.bool)
        index_div_bool_zeros_count_tile = 1 - index_div_bool_zeros_count_tile_bool.type(int_type)
        index_div_bool_zeros_count_tile_out = torch.sum(index_div_bool_zeros_count_tile, dim=1)
        index_div_bool_zeros_count_tile_out = index_div_bool_zeros_count_tile_out.type(int_type)
        predictor_mask = (~make_pad_mask(encoder_sequence_length, maxlen=encoder_sequence_length.max())).type(
            int_type).to(encoder_sequence_length.device)
        index_div_bool_zeros_count_tile_out = index_div_bool_zeros_count_tile_out * predictor_mask

        predictor_alignments = index_div_bool_zeros_count_tile_out
        predictor_alignments_length = predictor_alignments.sum(-1).type(encoder_sequence_length.dtype)
        return predictor_alignments.detach(), predictor_alignments_length.detach()

    def gen_tf2torch_map_dict(self):
    
        tensor_name_prefix_torch = self.tf2torch_tensor_name_prefix_torch
        tensor_name_prefix_tf = self.tf2torch_tensor_name_prefix_tf
        map_dict_local = {
            ## predictor
            "{}.cif_conv1d.weight".format(tensor_name_prefix_torch):
                {"name": "{}/conv1d/kernel".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": (2, 1, 0),
                 },  # (256,256,3),(3,256,256)
            "{}.cif_conv1d.bias".format(tensor_name_prefix_torch):
                {"name": "{}/conv1d/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
            "{}.cif_output.weight".format(tensor_name_prefix_torch):
                {"name": "{}/conv1d_1/kernel".format(tensor_name_prefix_tf),
                 "squeeze": 0,
                 "transpose": (1, 0),
                 },  # (1,256),(1,256,1)
            "{}.cif_output.bias".format(tensor_name_prefix_torch):
                {"name": "{}/conv1d_1/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (1,),(1,)
        }
        return map_dict_local

    def convert_tf2torch(self,
                         var_dict_tf,
                         var_dict_torch,
                         ):
        map_dict = self.gen_tf2torch_map_dict()
        var_dict_torch_update = dict()
        for name in sorted(var_dict_torch.keys(), reverse=False):
            names = name.split('.')
            if names[0] == self.tf2torch_tensor_name_prefix_torch:
                name_tf = map_dict[name]["name"]
                data_tf = var_dict_tf[name_tf]
                if map_dict[name]["squeeze"] is not None:
                    data_tf = np.squeeze(data_tf, axis=map_dict[name]["squeeze"])
                if map_dict[name]["transpose"] is not None:
                    data_tf = np.transpose(data_tf, map_dict[name]["transpose"])
                data_tf = torch.from_numpy(data_tf).type(torch.float32).to("cpu")
                assert var_dict_torch[name].size() == data_tf.size(), "{}, {}, {} != {}".format(name, name_tf,
                                                                                                var_dict_torch[
                                                                                                    name].size(),
                                                                                                data_tf.size())
                var_dict_torch_update[name] = data_tf
                logging.info(
                    "torch tensor: {}, {}, loading from tf tensor: {}, {}".format(name, data_tf.size(), name_tf,
                                                                                  var_dict_tf[name_tf].shape))
    
        return var_dict_torch_update


class mae_loss(nn.Module):

    def __init__(self, normalize_length=False):
        super(mae_loss, self).__init__()
        self.normalize_length = normalize_length
        self.criterion = torch.nn.L1Loss(reduction='sum')

    def forward(self, token_length, pre_token_length):
        loss_token_normalizer = token_length.size(0)
        if self.normalize_length:
            loss_token_normalizer = token_length.sum().type(torch.float32)
        loss = self.criterion(token_length, pre_token_length)
        loss = loss / loss_token_normalizer
        return loss


def cif(hidden, alphas, threshold):
    batch_size, len_time, hidden_size = hidden.size()

    # loop varss
    integrate = torch.zeros([batch_size], device=hidden.device)
    frame = torch.zeros([batch_size, hidden_size], device=hidden.device)
    # intermediate vars along time
    list_fires = []
    list_frames = []

    for t in range(len_time):
        alpha = alphas[:, t]
        distribution_completion = torch.ones([batch_size], device=hidden.device) - integrate

        integrate += alpha
        list_fires.append(integrate)

        fire_place = integrate >= threshold
        integrate = torch.where(fire_place,
                                integrate - torch.ones([batch_size], device=hidden.device),
                                integrate)
        cur = torch.where(fire_place,
                          distribution_completion,
                          alpha)
        remainds = alpha - cur

        frame += cur[:, None] * hidden[:, t, :]
        list_frames.append(frame)
        frame = torch.where(fire_place[:, None].repeat(1, hidden_size),
                            remainds[:, None] * hidden[:, t, :],
                            frame)

    fires = torch.stack(list_fires, 1)
    frames = torch.stack(list_frames, 1)
    list_ls = []
    len_labels = torch.round(alphas.sum(-1)).int()
    max_label_len = len_labels.max()
    for b in range(batch_size):
        fire = fires[b, :]
        l = torch.index_select(frames[b, :, :], 0, torch.nonzero(fire >= threshold).squeeze())
        pad_l = torch.zeros([max_label_len - l.size(0), hidden_size], device=hidden.device)
        list_ls.append(torch.cat([l, pad_l], 0))
    return torch.stack(list_ls, 0), fires


def cif_wo_hidden(alphas, threshold):
    batch_size, len_time = alphas.size()

    # loop varss
    integrate = torch.zeros([batch_size], device=alphas.device)
    # intermediate vars along time
    list_fires = []

    for t in range(len_time):
        alpha = alphas[:, t]

        integrate += alpha
        list_fires.append(integrate)

        fire_place = integrate >= threshold
        integrate = torch.where(fire_place,
                                integrate - torch.ones([batch_size], device=alphas.device)*threshold,
                                integrate)

    fires = torch.stack(list_fires, 1)
    return fires


class CifPredictorV3(nn.Module):
    def __init__(self,
                 idim,
                 l_order,
                 r_order,
                 threshold=1.0,
                 dropout=0.1,
                 smooth_factor=1.0,
                 noise_threshold=0,
                 tail_threshold=0.0,
                 tf2torch_tensor_name_prefix_torch="predictor",
                 tf2torch_tensor_name_prefix_tf="seq2seq/cif",
                 smooth_factor2=1.0,
                 noise_threshold2=0,
                 upsample_times=5,
                 upsample_type="cnn",
                 use_cif1_cnn=True,
                 tail_mask=True,
                 ):
        super(CifPredictorV3, self).__init__()

        self.pad = nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = nn.Conv1d(idim, idim, l_order + r_order + 1)
        self.cif_output = nn.Linear(idim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.threshold = threshold
        self.smooth_factor = smooth_factor
        self.noise_threshold = noise_threshold
        self.tail_threshold = tail_threshold
        self.tf2torch_tensor_name_prefix_torch = tf2torch_tensor_name_prefix_torch
        self.tf2torch_tensor_name_prefix_tf = tf2torch_tensor_name_prefix_tf

        self.upsample_times = upsample_times
        self.upsample_type = upsample_type
        self.use_cif1_cnn = use_cif1_cnn
        if self.upsample_type == 'cnn':
            self.upsample_cnn = nn.ConvTranspose1d(idim, idim, self.upsample_times, self.upsample_times)
            self.cif_output2 = nn.Linear(idim, 1)
        elif self.upsample_type == 'cnn_blstm':
            self.upsample_cnn = nn.ConvTranspose1d(idim, idim, self.upsample_times, self.upsample_times)
            self.blstm = nn.LSTM(idim, idim, 1, bias=True, batch_first=True, dropout=0.0, bidirectional=True)
            self.cif_output2 = nn.Linear(idim*2, 1)
        elif self.upsample_type == 'cnn_attn':
            self.upsample_cnn = nn.ConvTranspose1d(idim, idim, self.upsample_times, self.upsample_times)
            from funasr.models.encoder.transformer_encoder import EncoderLayer as TransformerEncoderLayer
            from funasr.modules.attention import MultiHeadedAttention
            from funasr.modules.positionwise_feed_forward import PositionwiseFeedForward
            positionwise_layer_args = (
                idim,
                idim*2,
                0.1,
            )
            self.self_attn = TransformerEncoderLayer(
                idim,
                MultiHeadedAttention(
                    4, idim, 0.1
                ),
                PositionwiseFeedForward(*positionwise_layer_args),
                0.1,
                True, #normalize_before,
                False, #concat_after,
            )
            self.cif_output2 = nn.Linear(idim, 1)
        self.smooth_factor2 = smooth_factor2
        self.noise_threshold2 = noise_threshold2

    def forward(self, hidden, target_label=None, mask=None, ignore_id=-1, mask_chunk_predictor=None,
                target_label_length=None):
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        output = torch.relu(self.cif_conv1d(queries))

        # alphas2 is an extra head for timestamp prediction
        if not self.use_cif1_cnn:
            _output = context
        else:
            _output = output
        if self.upsample_type == 'cnn':
            output2 = self.upsample_cnn(_output)
            output2 = output2.transpose(1,2)
        elif self.upsample_type == 'cnn_blstm':
            output2 = self.upsample_cnn(_output)
            output2 = output2.transpose(1,2)
            output2, (_, _) = self.blstm(output2)
        elif self.upsample_type == 'cnn_attn':
            output2 = self.upsample_cnn(_output)
            output2 = output2.transpose(1,2)
            output2, _ = self.self_attn(output2, mask)
        # import pdb; pdb.set_trace()
        alphas2 = torch.sigmoid(self.cif_output2(output2))
        alphas2 = torch.nn.functional.relu(alphas2 * self.smooth_factor2 - self.noise_threshold2)
        # repeat the mask in T demension to match the upsampled length
        if mask is not None:
            mask2 = mask.repeat(1, self.upsample_times, 1).transpose(-1, -2).reshape(alphas2.shape[0], -1)
            mask2 = mask2.unsqueeze(-1)
            alphas2 = alphas2 * mask2
        alphas2 = alphas2.squeeze(-1)
        token_num2 = alphas2.sum(-1)

        output = output.transpose(1, 2)

        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(alphas * self.smooth_factor - self.noise_threshold)
        if mask is not None:
            mask = mask.transpose(-1, -2).float()
            alphas = alphas * mask
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor
        alphas = alphas.squeeze(-1)
        mask = mask.squeeze(-1)
        if target_label_length is not None:
            target_length = target_label_length
        elif target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None
        token_num = alphas.sum(-1)

        if target_length is not None:
            alphas *= (target_length / token_num)[:, None].repeat(1, alphas.size(1))
        elif self.tail_threshold > 0.0:
            hidden, alphas, token_num = self.tail_process_fn(hidden, alphas, token_num, mask=mask)

        acoustic_embeds, cif_peak = cif(hidden, alphas, self.threshold)
        if target_length is None and self.tail_threshold > 0.0:
            token_num_int = torch.max(token_num).type(torch.int32).item()
            acoustic_embeds = acoustic_embeds[:, :token_num_int, :]
        return acoustic_embeds, token_num, alphas, cif_peak, token_num2

    def get_upsample_timestamp(self, hidden, mask=None, token_num=None):
        h = hidden
        b = hidden.shape[0]
        context = h.transpose(1, 2)
        queries = self.pad(context)
        output = torch.relu(self.cif_conv1d(queries))

        # alphas2 is an extra head for timestamp prediction
        if not self.use_cif1_cnn:
            _output = context
        else:
            _output = output
        if self.upsample_type == 'cnn':
            output2 = self.upsample_cnn(_output)
            output2 = output2.transpose(1,2)
        elif self.upsample_type == 'cnn_blstm':
            output2 = self.upsample_cnn(_output)
            output2 = output2.transpose(1,2)
            output2, (_, _) = self.blstm(output2)
        elif self.upsample_type == 'cnn_attn':
            output2 = self.upsample_cnn(_output)
            output2 = output2.transpose(1,2)
            output2, _ = self.self_attn(output2, mask)
        alphas2 = torch.sigmoid(self.cif_output2(output2))
        alphas2 = torch.nn.functional.relu(alphas2 * self.smooth_factor2 - self.noise_threshold2)
        # repeat the mask in T demension to match the upsampled length
        if mask is not None:
            mask2 = mask.repeat(1, self.upsample_times, 1).transpose(-1, -2).reshape(alphas2.shape[0], -1)
            mask2 = mask2.unsqueeze(-1)
            alphas2 = alphas2 * mask2
        alphas2 = alphas2.squeeze(-1)
        _token_num = alphas2.sum(-1)
        if token_num is not None:
            alphas2 *= (token_num / _token_num)[:, None].repeat(1, alphas2.size(1))
        # re-downsample
        ds_alphas = alphas2.reshape(b, -1, self.upsample_times).sum(-1)
        ds_cif_peak = cif_wo_hidden(ds_alphas, self.threshold - 1e-4)
        # upsampled alphas and cif_peak
        us_alphas = alphas2
        us_cif_peak = cif_wo_hidden(us_alphas, self.threshold - 1e-4)
        return ds_alphas, ds_cif_peak, us_alphas, us_cif_peak

    def tail_process_fn(self, hidden, alphas, token_num=None, mask=None):
        b, t, d = hidden.size()
        tail_threshold = self.tail_threshold
        if mask is not None:
            zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
            ones_t = torch.ones_like(zeros_t)
            mask_1 = torch.cat([mask, zeros_t], dim=1)
            mask_2 = torch.cat([ones_t, mask], dim=1)
            mask = mask_2 - mask_1
            tail_threshold = mask * tail_threshold
            alphas = torch.cat([alphas, zeros_t], dim=1)
            alphas = torch.add(alphas, tail_threshold)
        else:
            tail_threshold = torch.tensor([tail_threshold], dtype=alphas.dtype).to(alphas.device)
            tail_threshold = torch.reshape(tail_threshold, (1, 1))
            alphas = torch.cat([alphas, tail_threshold], dim=1)
        zeros = torch.zeros((b, 1, d), dtype=hidden.dtype).to(hidden.device)
        hidden = torch.cat([hidden, zeros], dim=1)
        token_num = alphas.sum(dim=-1)
        token_num_floor = torch.floor(token_num)

        return hidden, alphas, token_num_floor

    def gen_frame_alignments(self,
                             alphas: torch.Tensor = None,
                             encoder_sequence_length: torch.Tensor = None):
        batch_size, maximum_length = alphas.size()
        int_type = torch.int32

        is_training = self.training
        if is_training:
            token_num = torch.round(torch.sum(alphas, dim=1)).type(int_type)
        else:
            token_num = torch.floor(torch.sum(alphas, dim=1)).type(int_type)

        max_token_num = torch.max(token_num).item()

        alphas_cumsum = torch.cumsum(alphas, dim=1)
        alphas_cumsum = torch.floor(alphas_cumsum).type(int_type)
        alphas_cumsum = alphas_cumsum[:, None, :].repeat(1, max_token_num, 1)

        index = torch.ones([batch_size, max_token_num], dtype=int_type)
        index = torch.cumsum(index, dim=1)
        index = index[:, :, None].repeat(1, 1, maximum_length).to(alphas_cumsum.device)

        index_div = torch.floor(torch.true_divide(alphas_cumsum, index)).type(int_type)
        index_div_bool_zeros = index_div.eq(0)
        index_div_bool_zeros_count = torch.sum(index_div_bool_zeros, dim=-1) + 1
        index_div_bool_zeros_count = torch.clamp(index_div_bool_zeros_count, 0, encoder_sequence_length.max())
        token_num_mask = (~make_pad_mask(token_num, maxlen=max_token_num)).to(token_num.device)
        index_div_bool_zeros_count *= token_num_mask

        index_div_bool_zeros_count_tile = index_div_bool_zeros_count[:, :, None].repeat(1, 1, maximum_length)
        ones = torch.ones_like(index_div_bool_zeros_count_tile)
        zeros = torch.zeros_like(index_div_bool_zeros_count_tile)
        ones = torch.cumsum(ones, dim=2)
        cond = index_div_bool_zeros_count_tile == ones
        index_div_bool_zeros_count_tile = torch.where(cond, zeros, ones)

        index_div_bool_zeros_count_tile_bool = index_div_bool_zeros_count_tile.type(torch.bool)
        index_div_bool_zeros_count_tile = 1 - index_div_bool_zeros_count_tile_bool.type(int_type)
        index_div_bool_zeros_count_tile_out = torch.sum(index_div_bool_zeros_count_tile, dim=1)
        index_div_bool_zeros_count_tile_out = index_div_bool_zeros_count_tile_out.type(int_type)
        predictor_mask = (~make_pad_mask(encoder_sequence_length, maxlen=encoder_sequence_length.max())).type(
            int_type).to(encoder_sequence_length.device)
        index_div_bool_zeros_count_tile_out = index_div_bool_zeros_count_tile_out * predictor_mask

        predictor_alignments = index_div_bool_zeros_count_tile_out
        predictor_alignments_length = predictor_alignments.sum(-1).type(encoder_sequence_length.dtype)
        return predictor_alignments.detach(), predictor_alignments_length.detach()

class BATPredictor(nn.Module):
    def __init__(self, idim, l_order, r_order, threshold=1.0, dropout=0.1, smooth_factor=1.0, noise_threshold=0, return_accum=False):
        super(BATPredictor, self).__init__()

        self.pad = nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = nn.Conv1d(idim, idim, l_order + r_order + 1, groups=idim)
        self.cif_output = nn.Linear(idim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.threshold = threshold
        self.smooth_factor = smooth_factor
        self.noise_threshold = noise_threshold
        self.return_accum = return_accum

    def cif(
        self,
        input: Tensor,
        alpha: Tensor,
        beta: float = 1.0,
        return_accum: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        B, S, C = input.size()
        assert tuple(alpha.size()) == (B, S), f"{alpha.size()} != {(B, S)}"

        dtype = alpha.dtype
        alpha = alpha.float()

        alpha_sum = alpha.sum(1)
        feat_lengths = (alpha_sum / beta).floor().long()
        T = feat_lengths.max()

        # aggregate and integrate
        csum = alpha.cumsum(-1)
        with torch.no_grad():
            # indices used for scattering
            right_idx = (csum / beta).floor().long().clip(max=T)
            left_idx = right_idx.roll(1, dims=1)
            left_idx[:, 0] = 0

            # count # of fires from each source
            fire_num = right_idx - left_idx
            extra_weights = (fire_num - 1).clip(min=0)
            # The extra entry in last dim is for
            output = input.new_zeros((B, T + 1, C))
            source_range = torch.arange(1, 1 + S).unsqueeze(0).type_as(input)
            zero = alpha.new_zeros((1,))

        # right scatter
        fire_mask = fire_num > 0
        right_weight = torch.where(
            fire_mask,
            csum - right_idx.type_as(alpha) * beta,
            zero
        ).type_as(input)
        # assert right_weight.ge(0).all(), f"{right_weight} should be non-negative."
        output.scatter_add_(
            1,
            right_idx.unsqueeze(-1).expand(-1, -1, C),
            right_weight.unsqueeze(-1) * input
        )

        # left scatter
        left_weight = (
            alpha - right_weight - extra_weights.type_as(alpha) * beta
        ).type_as(input)
        output.scatter_add_(
            1,
            left_idx.unsqueeze(-1).expand(-1, -1, C),
            left_weight.unsqueeze(-1) * input
        )

         # extra scatters
        if extra_weights.ge(0).any():
            extra_steps = extra_weights.max().item()
            tgt_idx = left_idx
            src_feats = input * beta
            for _ in range(extra_steps):
                tgt_idx = (tgt_idx + 1).clip(max=T)
                # (B, S, 1)
                src_mask = (extra_weights > 0)
                output.scatter_add_(
                    1,
                    tgt_idx.unsqueeze(-1).expand(-1, -1, C),
                    src_feats * src_mask.unsqueeze(2)
                )
                extra_weights -= 1

        output = output[:, :T, :]

        if return_accum:
            return output, csum
        else:
            return output, alpha

    def forward(self, hidden, target_label=None, mask=None, ignore_id=-1, mask_chunk_predictor=None, target_label_length=None):
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        memory = self.cif_conv1d(queries)
        output = memory + context
        output = self.dropout(output)
        output = output.transpose(1, 2)
        output = torch.relu(output)
        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(alphas*self.smooth_factor - self.noise_threshold)
        if mask is not None:
            alphas = alphas * mask.transpose(-1, -2).float()
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor
        alphas = alphas.squeeze(-1)
        if target_label_length is not None:
            target_length = target_label_length
        elif target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
            # logging.info("target_length: {}".format(target_length))
        else:
            target_length = None
        token_num = alphas.sum(-1)
        if target_length is not None:
            # length_noise = torch.rand(alphas.size(0), device=alphas.device) - 0.5
            # target_length = length_noise + target_length
            alphas *= ((target_length + 1e-4) / token_num)[:, None].repeat(1, alphas.size(1))
        acoustic_embeds, cif_peak = self.cif(hidden, alphas, self.threshold, self.return_accum)
        return acoustic_embeds, token_num, alphas, cif_peak
