from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from funasr.frontends.wav_frontend import WavFrontendMel23
from funasr.models.eend.encoder import EENDOLATransformerEncoder
from funasr.models.eend.encoder_decoder_attractor import EncoderDecoderAttractor
from funasr.models.eend.utils.losses import (
    standard_loss,
    cal_power_loss,
    fast_batch_pit_n_speaker_loss,
)
from funasr.models.eend.utils.power import create_powerlabel
from funasr.models.eend.utils.power import generate_mapping_dict
from funasr.train_utils.device_funcs import force_gatherable

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    pass
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


def pad_attractor(att, max_n_speakers):
    C, D = att.shape
    if C < max_n_speakers:
        att = torch.cat(
            [att, torch.zeros(max_n_speakers - C, D).to(torch.float32).to(att.device)], dim=0
        )
    return att


def pad_labels(ts, out_size):
    for i, t in enumerate(ts):
        if t.shape[1] < out_size:
            ts[i] = F.pad(t, (0, out_size - t.shape[1], 0, 0), mode="constant", value=0.0)
    return ts


def pad_results(ys, out_size):
    ys_padded = []
    for i, y in enumerate(ys):
        if y.shape[1] < out_size:
            ys_padded.append(
                torch.cat(
                    [
                        y,
                        torch.zeros(y.shape[0], out_size - y.shape[1])
                        .to(torch.float32)
                        .to(y.device),
                    ],
                    dim=1,
                )
            )
        else:
            ys_padded.append(y)
    return ys_padded


class DiarEENDOLAModel(nn.Module):
    """EEND-OLA diarization model"""

    def __init__(
        self,
        frontend: Optional[WavFrontendMel23],
        encoder: EENDOLATransformerEncoder,
        encoder_decoder_attractor: EncoderDecoderAttractor,
        n_units: int = 256,
        max_n_speaker: int = 8,
        attractor_loss_weight: float = 1.0,
        mapping_dict=None,
        **kwargs,
    ):
        super().__init__()
        self.frontend = frontend
        self.enc = encoder
        self.encoder_decoder_attractor = encoder_decoder_attractor
        self.attractor_loss_weight = attractor_loss_weight
        self.max_n_speaker = max_n_speaker
        if mapping_dict is None:
            mapping_dict = generate_mapping_dict(max_speaker_num=self.max_n_speaker)
            self.mapping_dict = mapping_dict
        # PostNet
        self.postnet = nn.LSTM(self.max_n_speaker, n_units, 1, batch_first=True)
        self.output_layer = nn.Linear(n_units, mapping_dict["oov"] + 1)

    def forward_encoder(self, xs, ilens):
        xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=-1)
        pad_shape = xs.shape
        xs_mask = [torch.ones(ilen).to(xs.device) for ilen in ilens]
        xs_mask = torch.nn.utils.rnn.pad_sequence(
            xs_mask, batch_first=True, padding_value=0
        ).unsqueeze(-2)
        emb = self.enc(xs, xs_mask)
        emb = torch.split(emb.view(pad_shape[0], pad_shape[1], -1), 1, dim=0)
        emb = [e[0][:ilen] for e, ilen in zip(emb, ilens)]
        return emb

    def forward_post_net(self, logits, ilens):
        maxlen = torch.max(ilens).to(torch.int).item()
        logits = nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=-1)
        logits = nn.utils.rnn.pack_padded_sequence(
            logits, ilens.cpu().to(torch.int64), batch_first=True, enforce_sorted=False
        )
        outputs, (_, _) = self.postnet(logits)
        outputs = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, padding_value=-1, total_length=maxlen
        )[0]
        outputs = [output[: ilens[i].to(torch.int).item()] for i, output in enumerate(outputs)]
        outputs = [self.output_layer(output) for output in outputs]
        return outputs

    def forward(
        self,
        speech: List[torch.Tensor],
        speaker_labels: List[torch.Tensor],
        orders: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        # Check that batch_size is unified
        assert len(speech) == len(speaker_labels), (len(speech), len(speaker_labels))
        speech_lengths = torch.tensor([len(sph) for sph in speech]).to(torch.int64)
        speaker_labels_lengths = torch.tensor([spk.shape[-1] for spk in speaker_labels]).to(
            torch.int64
        )
        batch_size = len(speech)

        # Encoder
        encoder_out = self.forward_encoder(speech, speech_lengths)

        # Encoder-decoder attractor
        attractor_loss, attractors = self.encoder_decoder_attractor(
            [e[order] for e, order in zip(encoder_out, orders)], speaker_labels_lengths
        )
        speaker_logits = [
            torch.matmul(e, att.permute(1, 0)) for e, att in zip(encoder_out, attractors)
        ]

        # pit loss
        pit_speaker_labels = fast_batch_pit_n_speaker_loss(speaker_logits, speaker_labels)
        pit_loss = standard_loss(speaker_logits, pit_speaker_labels)

        # pse loss
        with torch.no_grad():
            power_ts = [
                create_powerlabel(label.cpu().numpy(), self.mapping_dict, self.max_n_speaker).to(
                    encoder_out[0].device, non_blocking=True
                )
                for label in pit_speaker_labels
            ]
        pad_attractors = [pad_attractor(att, self.max_n_speaker) for att in attractors]
        pse_speaker_logits = [
            torch.matmul(e, pad_att.permute(1, 0))
            for e, pad_att in zip(encoder_out, pad_attractors)
        ]
        pse_speaker_logits = self.forward_post_net(pse_speaker_logits, speech_lengths)
        pse_loss = cal_power_loss(pse_speaker_logits, power_ts)

        loss = pse_loss + pit_loss + self.attractor_loss_weight * attractor_loss

        stats = dict()
        stats["pse_loss"] = pse_loss.detach()
        stats["pit_loss"] = pit_loss.detach()
        stats["attractor_loss"] = attractor_loss.detach()
        stats["batch_size"] = batch_size

        # Collect total loss stats
        stats["loss"] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def estimate_sequential(
        self,
        speech: torch.Tensor,
        n_speakers: int = None,
        shuffle: bool = True,
        threshold: float = 0.5,
        **kwargs,
    ):
        speech_lengths = torch.tensor([len(sph) for sph in speech]).to(torch.int64)
        emb = self.forward_encoder(speech, speech_lengths)
        if shuffle:
            orders = [np.arange(e.shape[0]) for e in emb]
            for order in orders:
                np.random.shuffle(order)
            attractors, probs = self.encoder_decoder_attractor.estimate(
                [
                    e[torch.from_numpy(order).to(torch.long).to(speech[0].device)]
                    for e, order in zip(emb, orders)
                ]
            )
        else:
            attractors, probs = self.encoder_decoder_attractor.estimate(emb)
        attractors_active = []
        for p, att, e in zip(probs, attractors, emb):
            if n_speakers and n_speakers >= 0:
                att = att[:n_speakers,]
                attractors_active.append(att)
            elif threshold is not None:
                silence = torch.nonzero(p < threshold)[0]
                n_spk = silence[0] if silence.size else None
                att = att[:n_spk,]
                attractors_active.append(att)
            else:
                NotImplementedError("n_speakers or threshold has to be given.")
        raw_n_speakers = [att.shape[0] for att in attractors_active]
        attractors = [
            (
                pad_attractor(att, self.max_n_speaker)
                if att.shape[0] <= self.max_n_speaker
                else att[: self.max_n_speaker]
            )
            for att in attractors_active
        ]
        ys = [torch.matmul(e, att.permute(1, 0)) for e, att in zip(emb, attractors)]
        logits = self.forward_post_net(ys, speech_lengths)
        ys = [
            self.recover_y_from_powerlabel(logit, raw_n_speaker)
            for logit, raw_n_speaker in zip(logits, raw_n_speakers)
        ]

        return ys, emb, attractors, raw_n_speakers

    def recover_y_from_powerlabel(self, logit, n_speaker):
        pred = torch.argmax(torch.softmax(logit, dim=-1), dim=-1)
        oov_index = torch.where(pred == self.mapping_dict["oov"])[0]
        for i in oov_index:
            if i > 0:
                pred[i] = pred[i - 1]
            else:
                pred[i] = 0
        pred = [self.inv_mapping_func(i) for i in pred]
        decisions = [bin(num)[2:].zfill(self.max_n_speaker)[::-1] for num in pred]
        decisions = (
            torch.from_numpy(
                np.stack([np.array([int(i) for i in dec]) for dec in decisions], axis=0)
            )
            .to(logit.device)
            .to(torch.float32)
        )
        decisions = decisions[:, :n_speaker]
        return decisions

    def inv_mapping_func(self, label):

        if not isinstance(label, int):
            label = int(label)
        if label in self.mapping_dict["label2dec"].keys():
            num = self.mapping_dict["label2dec"][label]
        else:
            num = -1
        return num

    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass
