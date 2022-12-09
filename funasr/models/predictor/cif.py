import torch
from torch import nn

from funasr.modules.nets_utils import make_pad_mask
from funasr.modules.streaming_utils.utils import sequence_mask

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
            alphas = alphas * mask.transpose(-1, -2).float()
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor
        alphas = alphas.squeeze(-1)
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
            alphas = torch.cat([alphas, tail_threshold], dim=1)
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
    def __init__(self, idim, l_order, r_order, threshold=1.0, dropout=0.1, smooth_factor=1.0, noise_threshold=0,
                 tail_threshold=0.0):
        super(CifPredictorV2, self).__init__()

        self.pad = nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = nn.Conv1d(idim, idim, l_order + r_order + 1)
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
            alphas = torch.cat([alphas, tail_threshold], dim=1)
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
