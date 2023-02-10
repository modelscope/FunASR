#!/usr/bin/env python3
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from contextlib import contextmanager
from distutils.version import LooseVersion
from itertools import permutations
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch.nn import functional as F
from typeguard import check_argument_types

from funasr.modules.nets_utils import to_device
from funasr.modules.nets_utils import make_pad_mask
from funasr.models.decoder.abs_decoder import AbsDecoder
from funasr.models.encoder.abs_encoder import AbsEncoder
from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.layers.abs_normalize import AbsNormalize
from funasr.torch_utils.device_funcs import force_gatherable
from funasr.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class DiarSondModel(AbsESPnetModel):
    """Speaker overlap-aware neural diarization model
    reference: https://arxiv.org/abs/2211.10243
    """

    def __init__(
        self,
        vocab_size: int,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: AbsEncoder,
        speaker_encoder: AbsEncoder,
        ci_scorer: torch.nn.Module,
        cd_scorer: torch.nn.Module,
        decoder: torch.nn.Module,
        token_list: list,
        lsm_weight: float = 0.1,
        length_normalized_loss: bool = False,
        max_spk_num: int = 16,
        label_aggregator: Optional[torch.nn.Module] = None,
        normlize_speech_speaker: bool = False,
    ):
        assert check_argument_types()

        super().__init__()

        self.encoder = encoder
        self.speaker_encoder = speaker_encoder
        self.ci_scorer = ci_scorer
        self.cd_scorer = cd_scorer
        self.normalize = normalize
        self.frontend = frontend
        self.specaug = specaug
        self.label_aggregator = label_aggregator
        self.decoder = decoder
        self.token_list = token_list
        self.max_spk_num = max_spk_num
        self.normalize_speech_speaker = normlize_speech_speaker

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        profile: torch.Tensor = None,
        profile_lengths: torch.Tensor = None,
        spk_labels: torch.Tensor = None,
        spk_labels_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Speaker Encoder + CI Scorer + CD Scorer + Decoder + Calc loss

        Args:
            speech: (Batch, samples)
            speech_lengths: (Batch,) default None for chunk interator,
                                     because the chunk-iterator does not
                                     have the speech_lengths returned.
                                     see in
                                     espnet2/iterators/chunk_iter_factory.py
            profile: (Batch, N_spk, dim)
            profile_lengths: (Batch,)
            spk_labels: (Batch, )
        """
        assert speech.shape[0] == spk_labels.shape[0], (speech.shape, spk_labels.shape)
        batch_size = speech.shape[0]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        if self.attractor is None:
            # 2a. Decoder (baiscally a predction layer after encoder_out)
            pred = self.decoder(encoder_out, encoder_out_lens)
        else:
            # 2b. Encoder Decoder Attractors
            # Shuffle the chronological order of encoder_out, then calculate attractor
            encoder_out_shuffled = encoder_out.clone()
            for i in range(len(encoder_out_lens)):
                encoder_out_shuffled[i, : encoder_out_lens[i], :] = encoder_out[
                    i, torch.randperm(encoder_out_lens[i]), :
                ]
            attractor, att_prob = self.attractor(
                encoder_out_shuffled,
                encoder_out_lens,
                to_device(
                    self,
                    torch.zeros(
                        encoder_out.size(0), spk_labels.size(2) + 1, encoder_out.size(2)
                    ),
                ),
            )
            # Remove the final attractor which does not correspond to a speaker
            # Then multiply the attractors and encoder_out
            pred = torch.bmm(encoder_out, attractor[:, :-1, :].permute(0, 2, 1))
        # 3. Aggregate time-domain labels
        if self.label_aggregator is not None:
            spk_labels, spk_labels_lengths = self.label_aggregator(
                spk_labels, spk_labels_lengths
            )

        # If encoder uses conv* as input_layer (i.e., subsampling),
        # the sequence length of 'pred' might be slighly less than the
        # length of 'spk_labels'. Here we force them to be equal.
        length_diff_tolerance = 2
        length_diff = spk_labels.shape[1] - pred.shape[1]
        if length_diff > 0 and length_diff <= length_diff_tolerance:
            spk_labels = spk_labels[:, 0 : pred.shape[1], :]

        if self.attractor is None:
            loss_pit, loss_att = None, None
            loss, perm_idx, perm_list, label_perm = self.pit_loss(
                pred, spk_labels, encoder_out_lens
            )
        else:
            loss_pit, perm_idx, perm_list, label_perm = self.pit_loss(
                pred, spk_labels, encoder_out_lens
            )
            loss_att = self.attractor_loss(att_prob, spk_labels)
            loss = loss_pit + self.attractor_weight * loss_att
        (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        ) = self.calc_diarization_error(pred, label_perm, encoder_out_lens)

        if speech_scored > 0 and num_frames > 0:
            sad_mr, sad_fr, mi, fa, cf, acc, der = (
                speech_miss / speech_scored,
                speech_falarm / speech_scored,
                speaker_miss / speaker_scored,
                speaker_falarm / speaker_scored,
                speaker_error / speaker_scored,
                correct / num_frames,
                (speaker_miss + speaker_falarm + speaker_error) / speaker_scored,
            )
        else:
            sad_mr, sad_fr, mi, fa, cf, acc, der = 0, 0, 0, 0, 0, 0, 0

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_pit=loss_pit.detach() if loss_pit is not None else None,
            sad_mr=sad_mr,
            sad_fr=sad_fr,
            mi=mi,
            fa=fa,
            cf=cf,
            acc=acc,
            der=der,
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spk_labels: torch.Tensor = None,
        spk_labels_lengths: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode_speaker(
            self,
            profile: torch.Tensor,
            profile_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with autocast(False):
            if profile.shape[1] < self.max_spk_num:
                profile = F.pad(profile, [0, 0, 0, self.max_spk_num-profile.shape[1], 0, 0], "constant", 0.0)
            profile_mask = (torch.linalg.norm(profile, ord=2, dim=2, keepdim=True) > 0).float()
            profile = F.normalize(profile, dim=2)
            if self.speaker_encoder is not None:
                profile = self.speaker_encoder(profile, profile_lengths)[0]
                return profile * profile_mask, profile_lengths
            else:
                return profile, profile_lengths

    def encode_speech(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.encoder is not None:
            speech, speech_lengths = self.encode(speech, speech_lengths)
            speech_mask = ~make_pad_mask(speech_lengths, maxlen=speech.shape[1])
            speech_mask = speech_mask.to(speech.device).unsqueeze(-1).float()
            return speech * speech_mask, speech_lengths
        else:
            return speech, speech_lengths

    @staticmethod
    def concate_speech_ivc(
            speech: torch.Tensor,
            ivc: torch.Tensor
    ) -> torch.Tensor:
        nn, tt = ivc.shape[1], speech.shape[1]
        speech = speech.unsqueeze(dim=1)        # B x 1 x T x D
        speech = speech.expand(-1, nn, -1, -1)  # B x N x T x D
        ivc = ivc.unsqueeze(dim=2)              # B x N x 1 x D
        ivc = ivc.expand(-1, -1, tt, -1)        # B x N x T x D
        sd_in = torch.cat([speech, ivc], dim=3)  # B x N x T x 2D
        return sd_in

    def calc_similarity(
            self,
            speech_encoder_outputs: torch.Tensor,
            speaker_encoder_outputs: torch.Tensor,
            seq_len: torch.Tensor = None,
            spk_len: torch.Tensor = None,
    ) -> torch.Tensor:
        bb, tt = speech_encoder_outputs.shape[0], speech_encoder_outputs.shape[1]
        d_sph, d_spk = speech_encoder_outputs.shape[2], speaker_encoder_outputs.shape[2]
        if self.normalize_speech_speaker:
            speech_encoder_outputs = F.normalize(speech_encoder_outputs, dim=2)
            speaker_encoder_outputs = F.normalize(speaker_encoder_outputs, dim=2)
        ge_in = self.concate_speech_ivc(speech_encoder_outputs, speaker_encoder_outputs)
        ge_in = torch.reshape(ge_in, [bb * self.max_spk_num, tt, d_sph + d_spk])
        ge_len = seq_len.unsqueeze(1).expand(-1, self.max_spk_num)
        ge_len = torch.reshape(ge_len, [bb * self.max_spk_num])
        cd_simi = self.cd_scorer(ge_in, ge_len)[0]
        cd_simi = torch.reshape(cd_simi, [bb, self.max_spk_num, tt, 1])
        cd_simi = cd_simi.squeeze(dim=3).permute([0, 2, 1])

        if isinstance(self.ci_scorer, AbsEncoder):
            ci_simi = self.ci_scorer(ge_in, ge_len)[0]
        else:
            ci_simi = self.ci_scorer(speech_encoder_outputs, speaker_encoder_outputs)
        simi = torch.cat([cd_simi, ci_simi], dim=2)

        return simi

    def post_net_forward(self, simi, seq_len):
        logits = self.decoder(simi, seq_len)[0]

        return logits

    def prediction_forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            profile: torch.Tensor,
            profile_lengths: torch.Tensor,
    ) -> torch.Tensor:
        # speech encoding
        speech, speech_lengths = self.encode_speech(speech, speech_lengths)
        # speaker encoding
        profile, profile_lengths = self.encode_speaker(profile, profile_lengths)
        # calculating similarity
        similarity = self.calc_similarity(speech, profile, speech_lengths, profile_lengths)
        # post net forward
        logits = self.post_net_forward(similarity, speech_lengths)

        return logits

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

            # 4. Forward encoder
            # feats: (Batch, Length, Dim)
            # -> encoder_out: (Batch, Length2, Dim)
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = speech.shape[0]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else torch.ones(batch_size).int() * speech.shape[1]
        )

        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    @staticmethod
    def calc_diarization_error(pred, label, length):
        # Note (jiatong): Credit to https://github.com/hitachi-speech/EEND

        (batch_size, max_len, num_output) = label.size()
        # mask the padding part
        mask = np.zeros((batch_size, max_len, num_output))
        for i in range(batch_size):
            mask[i, : length[i], :] = 1

        # pred and label have the shape (batch_size, max_len, num_output)
        label_np = label.data.cpu().numpy().astype(int)
        pred_np = (pred.data.cpu().numpy() > 0).astype(int)
        label_np = label_np * mask
        pred_np = pred_np * mask
        length = length.data.cpu().numpy()

        # compute speech activity detection error
        n_ref = np.sum(label_np, axis=2)
        n_sys = np.sum(pred_np, axis=2)
        speech_scored = float(np.sum(n_ref > 0))
        speech_miss = float(np.sum(np.logical_and(n_ref > 0, n_sys == 0)))
        speech_falarm = float(np.sum(np.logical_and(n_ref == 0, n_sys > 0)))

        # compute speaker diarization error
        speaker_scored = float(np.sum(n_ref))
        speaker_miss = float(np.sum(np.maximum(n_ref - n_sys, 0)))
        speaker_falarm = float(np.sum(np.maximum(n_sys - n_ref, 0)))
        n_map = np.sum(np.logical_and(label_np == 1, pred_np == 1), axis=2)
        speaker_error = float(np.sum(np.minimum(n_ref, n_sys) - n_map))
        correct = float(1.0 * np.sum((label_np == pred_np) * mask) / num_output)
        num_frames = np.sum(length)
        return (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        )
