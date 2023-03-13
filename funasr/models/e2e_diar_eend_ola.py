# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import Tuple

import numpy as np
import torch
import torch.nn as  nn
from typeguard import check_argument_types

from funasr.modules.eend_ola.encoder import TransformerEncoder
from funasr.modules.eend_ola.encoder_decoder_attractor import EncoderDecoderAttractor
from funasr.modules.eend_ola.utils.power import generate_mapping_dict
from funasr.torch_utils.device_funcs import force_gatherable
from funasr.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    pass
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class DiarEENDOLAModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
            self,
            encoder: TransformerEncoder,
            eda: EncoderDecoderAttractor,
            n_units: int = 256,
            max_n_speaker: int = 8,
            attractor_loss_weight: float = 1.0,
            mapping_dict=None,
            **kwargs,
    ):
        assert check_argument_types()

        super().__init__()
        self.encoder = encoder
        self.eda = eda
        self.attractor_loss_weight = attractor_loss_weight
        self.max_n_speaker = max_n_speaker
        if mapping_dict is None:
            mapping_dict = generate_mapping_dict(max_speaker_num=self.max_n_speaker)
            self.mapping_dict = mapping_dict
        # PostNet
        self.PostNet = nn.LSTM(self.max_n_speaker, n_units, 1, batch_first=True)
        self.output_layer = nn.Linear(n_units, mapping_dict['oov'] + 1)

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
                speech.shape[0]
                == speech_lengths.shape[0]
                == text.shape[0]
                == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text, text_lengths
                )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (
                               1 - self.interctc_weight
                       ) * loss_ctc + self.interctc_weight * loss_interctc

        # 2b. Attention decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def estimate_sequential(self,
                            speech: torch.Tensor,
                            speech_lengths: torch.Tensor,
                            n_speakers: int,
                            shuffle: bool,
                            threshold: float,
                            **kwargs):
        speech = [s[:s_len] for s, s_len in zip(speech, speech_lengths)]
        emb = self.forward_core(speech)  # list, [(T1, C1), ..., (T1, C1)]
        if shuffle:
            orders = [np.arange(e.shape[0]) for e in emb]
            for order in orders:
                np.random.shuffle(order)
            # e[order]: shuffle后的embeddings, list, [(T1, C1), ..., (T1, C1)]  每个sample的T维度已进行随机顺序交换
            # attractors, list, hts(论文里的as), [(max_n_speakers, n_units), ..., (max_n_speakers, n_units)]
            # probs, list, [(max_n_speakers, ), ..., (max_n_speakers, ]
            attractors, probs = self.eda.estimate(
                [e[torch.from_numpy(order).to(torch.long).to(xs[0].device)] for e, order in zip(emb, orders)])
        else:
            attractors, probs = self.eda.estimate(emb)
        attractors_active = []
        for p, att, e in zip(probs, attractors, emb):
            if n_speakers and n_speakers >= 0:  # 根据指定说话人数, 选择对应数量的ys
                # TODO：在测试有不同数量speaker数的数据集时，考虑改成根据sample来确定具体的speaker数，而不是直接指定
                # raise NotImplementedError
                att = att[:n_speakers, ]
                attractors_active.append(att)
            elif threshold is not None:
                silence = torch.nonzero(p < threshold)[0]  # 找到第一个输出概率小于阈值的索引, 作为结束, 且值刚好等于说话人数
                n_spk = silence[0] if silence.size else None
                att = att[:n_spk, ]
                attractors_active.append(att)
            else:
                NotImplementedError('n_speakers or th has to be given.')
        raw_n_speakers = [att.shape[0] for att in attractors_active]  # [C1, C2, ..., CB]
        attractors = [
            pad_attractor(att, self.max_n_speaker) if att.shape[0] <= self.max_n_speaker else att[:self.max_n_speaker]
            for att in attractors_active]
        ys = [torch.matmul(e, att.permute(1, 0)) for e, att in zip(emb, attractors)]
        # ys_eda = [torch.sigmoid(y[:, :n_spk]) for y,n_spk in zip(ys, raw_n_speakers)]
        logits = self.cal_postnet(ys, self.max_n_speaker)
        ys = [self.recover_y_from_powerlabel(logit, raw_n_speaker) for logit, raw_n_speaker in
              zip(logits, raw_n_speakers)]

        return ys, emb, attractors, raw_n_speakers

    def recover_y_from_powerlabel(self, logit, n_speaker):
        pred = torch.argmax(torch.softmax(logit, dim=-1), dim=-1)  # (T, )
        oov_index = torch.where(pred == self.mapping_dict['oov'])[0]
        for i in oov_index:
            if i > 0:
                pred[i] = pred[i - 1]
            else:
                pred[i] = 0
        pred = [self.reporter.inv_mapping_func(i, self.mapping_dict) for i in pred]
        # print(pred)
        decisions = [bin(num)[2:].zfill(self.max_n_speaker)[::-1] for num in pred]
        decisions = torch.from_numpy(
            np.stack([np.array([int(i) for i in dec]) for dec in decisions], axis=0)).to(logit.device).to(
            torch.float32)
        decisions = decisions[:, :n_speaker]
        return decisions
