import math
import torch
import numpy as np
import torch.nn.functional as F

from funasr.models.scama.utils import sequence_mask
from funasr.models.transformer.utils.nets_utils import make_pad_mask


class overlap_chunk:
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    San-m: Memory equipped self-attention for end-to-end speech recognition
    https://arxiv.org/abs/2006.01713

    """

    def __init__(
        self,
        chunk_size: tuple = (16,),
        stride: tuple = (10,),
        pad_left: tuple = (0,),
        encoder_att_look_back_factor: tuple = (1,),
        shfit_fsmn: int = 0,
        decoder_att_look_back_factor: tuple = (1,),
    ):

        pad_left = self.check_chunk_size_args(chunk_size, pad_left)
        encoder_att_look_back_factor = self.check_chunk_size_args(
            chunk_size, encoder_att_look_back_factor
        )
        decoder_att_look_back_factor = self.check_chunk_size_args(
            chunk_size, decoder_att_look_back_factor
        )
        (
            self.chunk_size,
            self.stride,
            self.pad_left,
            self.encoder_att_look_back_factor,
            self.decoder_att_look_back_factor,
        ) = (
            chunk_size,
            stride,
            pad_left,
            encoder_att_look_back_factor,
            decoder_att_look_back_factor,
        )
        self.shfit_fsmn = shfit_fsmn
        self.x_add_mask = None
        self.x_rm_mask = None
        self.x_len = None
        self.mask_shfit_chunk = None
        self.mask_chunk_predictor = None
        self.mask_att_chunk_encoder = None
        self.mask_shift_att_chunk_decoder = None
        self.chunk_outs = None
        (
            self.chunk_size_cur,
            self.stride_cur,
            self.pad_left_cur,
            self.encoder_att_look_back_factor_cur,
            self.chunk_size_pad_shift_cur,
        ) = (None, None, None, None, None)

    def check_chunk_size_args(self, chunk_size, x):
        if len(x) < len(chunk_size):
            x = [x[0] for i in chunk_size]
        return x

    def get_chunk_size(self, ind: int = 0):
        # with torch.no_grad:
        chunk_size, stride, pad_left, encoder_att_look_back_factor, decoder_att_look_back_factor = (
            self.chunk_size[ind],
            self.stride[ind],
            self.pad_left[ind],
            self.encoder_att_look_back_factor[ind],
            self.decoder_att_look_back_factor[ind],
        )
        (
            self.chunk_size_cur,
            self.stride_cur,
            self.pad_left_cur,
            self.encoder_att_look_back_factor_cur,
            self.chunk_size_pad_shift_cur,
            self.decoder_att_look_back_factor_cur,
        ) = (
            chunk_size,
            stride,
            pad_left,
            encoder_att_look_back_factor,
            chunk_size + self.shfit_fsmn,
            decoder_att_look_back_factor,
        )
        return (
            self.chunk_size_cur,
            self.stride_cur,
            self.pad_left_cur,
            self.encoder_att_look_back_factor_cur,
            self.chunk_size_pad_shift_cur,
        )

    def random_choice(self, training=True, decoding_ind=None):
        chunk_num = len(self.chunk_size)
        ind = 0
        if training and chunk_num > 1:
            ind = torch.randint(0, chunk_num, ()).cpu().item()
        if not training and decoding_ind is not None:
            ind = int(decoding_ind)

        return ind

    def gen_chunk_mask(self, x_len, ind=0, num_units=1, num_units_predictor=1):

        with torch.no_grad():
            x_len = x_len.cpu().numpy()
            x_len_max = x_len.max()

            chunk_size, stride, pad_left, encoder_att_look_back_factor, chunk_size_pad_shift = (
                self.get_chunk_size(ind)
            )
            shfit_fsmn = self.shfit_fsmn
            pad_right = chunk_size - stride - pad_left

            chunk_num_batch = np.ceil(x_len / stride).astype(np.int32)
            x_len_chunk = (
                (chunk_num_batch - 1) * chunk_size_pad_shift
                + shfit_fsmn
                + pad_left
                + 0
                + x_len
                - (chunk_num_batch - 1) * stride
            )
            x_len_chunk = x_len_chunk.astype(x_len.dtype)
            x_len_chunk_max = x_len_chunk.max()

            chunk_num = int(math.ceil(x_len_max / stride))
            dtype = np.int32
            max_len_for_x_mask_tmp = max(chunk_size, x_len_max + pad_left)
            x_add_mask = np.zeros([0, max_len_for_x_mask_tmp], dtype=dtype)
            x_rm_mask = np.zeros([max_len_for_x_mask_tmp, 0], dtype=dtype)
            mask_shfit_chunk = np.zeros([0, num_units], dtype=dtype)
            mask_chunk_predictor = np.zeros([0, num_units_predictor], dtype=dtype)
            mask_shift_att_chunk_decoder = np.zeros([0, 1], dtype=dtype)
            mask_att_chunk_encoder = np.zeros([0, chunk_num * chunk_size_pad_shift], dtype=dtype)
            for chunk_ids in range(chunk_num):
                # x_mask add
                fsmn_padding = np.zeros((shfit_fsmn, max_len_for_x_mask_tmp), dtype=dtype)
                x_mask_cur = np.diag(np.ones(chunk_size, dtype=np.float32))
                x_mask_pad_left = np.zeros((chunk_size, chunk_ids * stride), dtype=dtype)
                x_mask_pad_right = np.zeros((chunk_size, max_len_for_x_mask_tmp), dtype=dtype)
                x_cur_pad = np.concatenate([x_mask_pad_left, x_mask_cur, x_mask_pad_right], axis=1)
                x_cur_pad = x_cur_pad[:chunk_size, :max_len_for_x_mask_tmp]
                x_add_mask_fsmn = np.concatenate([fsmn_padding, x_cur_pad], axis=0)
                x_add_mask = np.concatenate([x_add_mask, x_add_mask_fsmn], axis=0)

                # x_mask rm
                fsmn_padding = np.zeros((max_len_for_x_mask_tmp, shfit_fsmn), dtype=dtype)
                padding_mask_left = np.zeros((max_len_for_x_mask_tmp, pad_left), dtype=dtype)
                padding_mask_right = np.zeros((max_len_for_x_mask_tmp, pad_right), dtype=dtype)
                x_mask_cur = np.diag(np.ones(stride, dtype=dtype))
                x_mask_cur_pad_top = np.zeros((chunk_ids * stride, stride), dtype=dtype)
                x_mask_cur_pad_bottom = np.zeros((max_len_for_x_mask_tmp, stride), dtype=dtype)
                x_rm_mask_cur = np.concatenate(
                    [x_mask_cur_pad_top, x_mask_cur, x_mask_cur_pad_bottom], axis=0
                )
                x_rm_mask_cur = x_rm_mask_cur[:max_len_for_x_mask_tmp, :stride]
                x_rm_mask_cur_fsmn = np.concatenate(
                    [fsmn_padding, padding_mask_left, x_rm_mask_cur, padding_mask_right], axis=1
                )
                x_rm_mask = np.concatenate([x_rm_mask, x_rm_mask_cur_fsmn], axis=1)

                # fsmn_padding_mask
                pad_shfit_mask = np.zeros([shfit_fsmn, num_units], dtype=dtype)
                ones_1 = np.ones([chunk_size, num_units], dtype=dtype)
                mask_shfit_chunk_cur = np.concatenate([pad_shfit_mask, ones_1], axis=0)
                mask_shfit_chunk = np.concatenate([mask_shfit_chunk, mask_shfit_chunk_cur], axis=0)

                # predictor mask
                zeros_1 = np.zeros([shfit_fsmn + pad_left, num_units_predictor], dtype=dtype)
                ones_2 = np.ones([stride, num_units_predictor], dtype=dtype)
                zeros_3 = np.zeros(
                    [chunk_size - stride - pad_left, num_units_predictor], dtype=dtype
                )
                ones_zeros = np.concatenate([ones_2, zeros_3], axis=0)
                mask_chunk_predictor_cur = np.concatenate([zeros_1, ones_zeros], axis=0)
                mask_chunk_predictor = np.concatenate(
                    [mask_chunk_predictor, mask_chunk_predictor_cur], axis=0
                )

                # encoder att mask
                zeros_1_top = np.zeros([shfit_fsmn, chunk_num * chunk_size_pad_shift], dtype=dtype)

                zeros_2_num = max(chunk_ids - encoder_att_look_back_factor, 0)
                zeros_2 = np.zeros([chunk_size, zeros_2_num * chunk_size_pad_shift], dtype=dtype)

                encoder_att_look_back_num = max(chunk_ids - zeros_2_num, 0)
                zeros_2_left = np.zeros([chunk_size, shfit_fsmn], dtype=dtype)
                ones_2_mid = np.ones([stride, stride], dtype=dtype)
                zeros_2_bottom = np.zeros([chunk_size - stride, stride], dtype=dtype)
                zeros_2_right = np.zeros([chunk_size, chunk_size - stride], dtype=dtype)
                ones_2 = np.concatenate([ones_2_mid, zeros_2_bottom], axis=0)
                ones_2 = np.concatenate([zeros_2_left, ones_2, zeros_2_right], axis=1)
                ones_2 = np.tile(ones_2, [1, encoder_att_look_back_num])

                zeros_3_left = np.zeros([chunk_size, shfit_fsmn], dtype=dtype)
                ones_3_right = np.ones([chunk_size, chunk_size], dtype=dtype)
                ones_3 = np.concatenate([zeros_3_left, ones_3_right], axis=1)

                zeros_remain_num = max(chunk_num - 1 - chunk_ids, 0)
                zeros_remain = np.zeros(
                    [chunk_size, zeros_remain_num * chunk_size_pad_shift], dtype=dtype
                )

                ones2_bottom = np.concatenate([zeros_2, ones_2, ones_3, zeros_remain], axis=1)
                mask_att_chunk_encoder_cur = np.concatenate([zeros_1_top, ones2_bottom], axis=0)
                mask_att_chunk_encoder = np.concatenate(
                    [mask_att_chunk_encoder, mask_att_chunk_encoder_cur], axis=0
                )

                # decoder fsmn_shift_att_mask
                zeros_1 = np.zeros([shfit_fsmn, 1])
                ones_1 = np.ones([chunk_size, 1])
                mask_shift_att_chunk_decoder_cur = np.concatenate([zeros_1, ones_1], axis=0)
                mask_shift_att_chunk_decoder = np.concatenate(
                    [mask_shift_att_chunk_decoder, mask_shift_att_chunk_decoder_cur], axis=0
                )

            self.x_add_mask = x_add_mask[:x_len_chunk_max, : x_len_max + pad_left]
            self.x_len_chunk = x_len_chunk
            self.x_rm_mask = x_rm_mask[:x_len_max, :x_len_chunk_max]
            self.x_len = x_len
            self.mask_shfit_chunk = mask_shfit_chunk[:x_len_chunk_max, :]
            self.mask_chunk_predictor = mask_chunk_predictor[:x_len_chunk_max, :]
            self.mask_att_chunk_encoder = mask_att_chunk_encoder[:x_len_chunk_max, :x_len_chunk_max]
            self.mask_shift_att_chunk_decoder = mask_shift_att_chunk_decoder[:x_len_chunk_max, :]
            self.chunk_outs = (
                self.x_add_mask,
                self.x_len_chunk,
                self.x_rm_mask,
                self.x_len,
                self.mask_shfit_chunk,
                self.mask_chunk_predictor,
                self.mask_att_chunk_encoder,
                self.mask_shift_att_chunk_decoder,
            )

        return self.chunk_outs

    def split_chunk(self, x, x_len, chunk_outs):
        """
        :param x: (b, t, d)
        :param x_length: (b)
        :param ind: int
        :return:
        """
        x = x[:, : x_len.max(), :]
        b, t, d = x.size()
        x_len_mask = (~make_pad_mask(x_len, maxlen=t)).to(x.device)
        x *= x_len_mask[:, :, None]

        x_add_mask = self.get_x_add_mask(chunk_outs, x.device, dtype=x.dtype)
        x_len_chunk = self.get_x_len_chunk(chunk_outs, x_len.device, dtype=x_len.dtype)
        pad = (0, 0, self.pad_left_cur, 0)
        x = F.pad(x, pad, "constant", 0.0)
        b, t, d = x.size()
        x = torch.transpose(x, 1, 0)
        x = torch.reshape(x, [t, -1])
        x_chunk = torch.mm(x_add_mask, x)
        x_chunk = torch.reshape(x_chunk, [-1, b, d]).transpose(1, 0)

        return x_chunk, x_len_chunk

    def remove_chunk(self, x_chunk, x_len_chunk, chunk_outs):
        x_chunk = x_chunk[:, : x_len_chunk.max(), :]
        b, t, d = x_chunk.size()
        x_len_chunk_mask = (~make_pad_mask(x_len_chunk, maxlen=t)).to(x_chunk.device)
        x_chunk *= x_len_chunk_mask[:, :, None]

        x_rm_mask = self.get_x_rm_mask(chunk_outs, x_chunk.device, dtype=x_chunk.dtype)
        x_len = self.get_x_len(chunk_outs, x_len_chunk.device, dtype=x_len_chunk.dtype)
        x_chunk = torch.transpose(x_chunk, 1, 0)
        x_chunk = torch.reshape(x_chunk, [t, -1])
        x = torch.mm(x_rm_mask, x_chunk)
        x = torch.reshape(x, [-1, b, d]).transpose(1, 0)

        return x, x_len

    def get_x_add_mask(self, chunk_outs=None, device="cpu", idx=0, dtype=torch.float32):
        with torch.no_grad():
            x = chunk_outs[idx] if chunk_outs is not None else self.chunk_outs[idx]
            x = torch.from_numpy(x).type(dtype).to(device)
        return x

    def get_x_len_chunk(self, chunk_outs=None, device="cpu", idx=1, dtype=torch.float32):
        with torch.no_grad():
            x = chunk_outs[idx] if chunk_outs is not None else self.chunk_outs[idx]
            x = torch.from_numpy(x).type(dtype).to(device)
        return x

    def get_x_rm_mask(self, chunk_outs=None, device="cpu", idx=2, dtype=torch.float32):
        with torch.no_grad():
            x = chunk_outs[idx] if chunk_outs is not None else self.chunk_outs[idx]
            x = torch.from_numpy(x).type(dtype).to(device)
        return x

    def get_x_len(self, chunk_outs=None, device="cpu", idx=3, dtype=torch.float32):
        with torch.no_grad():
            x = chunk_outs[idx] if chunk_outs is not None else self.chunk_outs[idx]
            x = torch.from_numpy(x).type(dtype).to(device)
        return x

    def get_mask_shfit_chunk(
        self, chunk_outs=None, device="cpu", batch_size=1, num_units=1, idx=4, dtype=torch.float32
    ):
        with torch.no_grad():
            x = chunk_outs[idx] if chunk_outs is not None else self.chunk_outs[idx]
            x = np.tile(
                x[
                    None,
                    :,
                    :,
                ],
                [batch_size, 1, num_units],
            )
            x = torch.from_numpy(x).type(dtype).to(device)
        return x

    def get_mask_chunk_predictor(
        self, chunk_outs=None, device="cpu", batch_size=1, num_units=1, idx=5, dtype=torch.float32
    ):
        with torch.no_grad():
            x = chunk_outs[idx] if chunk_outs is not None else self.chunk_outs[idx]
            x = np.tile(
                x[
                    None,
                    :,
                    :,
                ],
                [batch_size, 1, num_units],
            )
            x = torch.from_numpy(x).type(dtype).to(device)
        return x

    def get_mask_att_chunk_encoder(
        self, chunk_outs=None, device="cpu", batch_size=1, idx=6, dtype=torch.float32
    ):
        with torch.no_grad():
            x = chunk_outs[idx] if chunk_outs is not None else self.chunk_outs[idx]
            x = np.tile(
                x[
                    None,
                    :,
                    :,
                ],
                [batch_size, 1, 1],
            )
            x = torch.from_numpy(x).type(dtype).to(device)
        return x

    def get_mask_shift_att_chunk_decoder(
        self, chunk_outs=None, device="cpu", batch_size=1, idx=7, dtype=torch.float32
    ):
        with torch.no_grad():
            x = chunk_outs[idx] if chunk_outs is not None else self.chunk_outs[idx]
            x = np.tile(x[None, None, :, 0], [batch_size, 1, 1])
            x = torch.from_numpy(x).type(dtype).to(device)
        return x


def build_scama_mask_for_cross_attention_decoder(
    predictor_alignments: torch.Tensor,
    encoder_sequence_length: torch.Tensor,
    chunk_size: int = 5,
    encoder_chunk_size: int = 5,
    attention_chunk_center_bias: int = 0,
    attention_chunk_size: int = 1,
    attention_chunk_type: str = "chunk",
    step=None,
    predictor_mask_chunk_hopping: torch.Tensor = None,
    decoder_att_look_back_factor: int = 1,
    mask_shift_att_chunk_decoder: torch.Tensor = None,
    target_length: torch.Tensor = None,
    is_training=True,
    dtype: torch.dtype = torch.float32,
):
    with torch.no_grad():
        device = predictor_alignments.device
        batch_size, chunk_num = predictor_alignments.size()
        maximum_encoder_length = encoder_sequence_length.max().item()
        int_type = predictor_alignments.dtype
        if not is_training:
            target_length = predictor_alignments.sum(dim=-1).type(encoder_sequence_length.dtype)
        maximum_target_length = target_length.max()
        predictor_alignments_cumsum = torch.cumsum(predictor_alignments, dim=1)
        predictor_alignments_cumsum = predictor_alignments_cumsum[:, None, :].repeat(
            1, maximum_target_length, 1
        )

        index = torch.ones([batch_size, maximum_target_length], dtype=int_type).to(device)
        index = torch.cumsum(index, dim=1)
        index = index[:, :, None].repeat(1, 1, chunk_num)

        index_div = torch.floor(torch.divide(predictor_alignments_cumsum, index)).type(int_type)
        index_div_bool_zeros = index_div == 0
        index_div_bool_zeros_count = torch.sum(index_div_bool_zeros.type(int_type), dim=-1) + 1

        index_div_bool_zeros_count = torch.clip(index_div_bool_zeros_count, min=1, max=chunk_num)

        index_div_bool_zeros_count *= chunk_size
        index_div_bool_zeros_count += attention_chunk_center_bias
        index_div_bool_zeros_count = torch.clip(
            index_div_bool_zeros_count - 1, min=0, max=maximum_encoder_length
        )
        index_div_bool_zeros_count_ori = index_div_bool_zeros_count

        index_div_bool_zeros_count = (
            torch.floor(index_div_bool_zeros_count / encoder_chunk_size) + 1
        ) * encoder_chunk_size
        max_len_chunk = math.ceil(maximum_encoder_length / encoder_chunk_size) * encoder_chunk_size

        mask_flip, mask_flip2 = None, None
        if attention_chunk_size is not None:
            index_div_bool_zeros_count_beg = index_div_bool_zeros_count - attention_chunk_size
            index_div_bool_zeros_count_beg = torch.clip(
                index_div_bool_zeros_count_beg, 0, max_len_chunk
            )
            index_div_bool_zeros_count_beg_mask = sequence_mask(
                index_div_bool_zeros_count_beg, maxlen=max_len_chunk, dtype=int_type, device=device
            )
            mask_flip = 1 - index_div_bool_zeros_count_beg_mask
            attention_chunk_size2 = attention_chunk_size * (decoder_att_look_back_factor + 1)
            index_div_bool_zeros_count_beg = index_div_bool_zeros_count - attention_chunk_size2

            index_div_bool_zeros_count_beg = torch.clip(
                index_div_bool_zeros_count_beg, 0, max_len_chunk
            )
            index_div_bool_zeros_count_beg_mask = sequence_mask(
                index_div_bool_zeros_count_beg, maxlen=max_len_chunk, dtype=int_type, device=device
            )
            mask_flip2 = 1 - index_div_bool_zeros_count_beg_mask

        mask = sequence_mask(
            index_div_bool_zeros_count, maxlen=max_len_chunk, dtype=dtype, device=device
        )

        if predictor_mask_chunk_hopping is not None:
            b, k, t = mask.size()
            predictor_mask_chunk_hopping = predictor_mask_chunk_hopping[:, None, :, 0].repeat(
                1, k, 1
            )

            mask_mask_flip = mask
            if mask_flip is not None:
                mask_mask_flip = mask_flip * mask

            def _fn():
                mask_sliced = mask[:b, :k, encoder_chunk_size:t]
                zero_pad_right = torch.zeros(
                    [b, k, encoder_chunk_size], dtype=mask_sliced.dtype
                ).to(device)
                mask_sliced = torch.cat([mask_sliced, zero_pad_right], dim=2)
                _, _, tt = predictor_mask_chunk_hopping.size()
                pad_right_p = max_len_chunk - tt
                predictor_mask_chunk_hopping_pad = torch.nn.functional.pad(
                    predictor_mask_chunk_hopping, [0, pad_right_p], "constant", 0
                )
                masked = mask_sliced * predictor_mask_chunk_hopping_pad

                mask_true = mask_mask_flip + masked
                return mask_true

            mask = _fn() if t > chunk_size else mask_mask_flip

        if mask_flip2 is not None:
            mask *= mask_flip2

        mask_target = sequence_mask(
            target_length, maxlen=maximum_target_length, dtype=mask.dtype, device=device
        )
        mask = mask[:, :maximum_target_length, :] * mask_target[:, :, None]

        mask_len = sequence_mask(
            encoder_sequence_length, maxlen=maximum_encoder_length, dtype=mask.dtype, device=device
        )
        mask = mask[:, :, :maximum_encoder_length] * mask_len[:, None, :]

        if attention_chunk_type == "full":
            mask = torch.ones_like(mask).to(device)
        if mask_shift_att_chunk_decoder is not None:
            mask = mask * mask_shift_att_chunk_decoder
        mask = mask[:, :maximum_target_length, :maximum_encoder_length].type(dtype).to(device)

    return mask
