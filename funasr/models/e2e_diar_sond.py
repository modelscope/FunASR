#!/usr/bin/env python3
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from contextlib import contextmanager
from distutils.version import LooseVersion
from itertools import permutations
from typing import Dict
from typing import Optional
from typing import Tuple, List

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
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss, SequenceBinaryCrossEntropy
from funasr.utils.misc import int2vec

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
        encoder: torch.nn.Module,
        speaker_encoder: Optional[torch.nn.Module],
        ci_scorer: torch.nn.Module,
        cd_scorer: Optional[torch.nn.Module],
        decoder: torch.nn.Module,
        token_list: list,
        lsm_weight: float = 0.1,
        length_normalized_loss: bool = False,
        max_spk_num: int = 16,
        label_aggregator: Optional[torch.nn.Module] = None,
        normalize_speech_speaker: bool = False,
        ignore_id: int = -1,
        speaker_discrimination_loss_weight: float = 1.0,
        inter_score_loss_weight: float = 0.0
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
        self.normalize_speech_speaker = normalize_speech_speaker
        self.ignore_id = ignore_id
        self.criterion_diar = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.criterion_bce = SequenceBinaryCrossEntropy(normalize_length=length_normalized_loss)
        self.pse_embedding = self.generate_pse_embedding()
        # self.register_buffer("pse_embedding", pse_embedding)
        self.power_weight = torch.from_numpy(2 ** np.arange(max_spk_num)[np.newaxis, np.newaxis, :]).float()
        # self.register_buffer("power_weight", power_weight)
        self.int_token_arr = torch.from_numpy(np.array(self.token_list).astype(int)[np.newaxis, np.newaxis, :]).int()
        # self.register_buffer("int_token_arr", int_token_arr)
        self.speaker_discrimination_loss_weight = speaker_discrimination_loss_weight
        self.inter_score_loss_weight = inter_score_loss_weight
        self.forward_steps = 0

    def generate_pse_embedding(self):
        embedding = np.zeros((len(self.token_list), self.max_spk_num), dtype=np.float)
        for idx, pse_label in enumerate(self.token_list):
            emb = int2vec(int(pse_label), vec_dim=self.max_spk_num, dtype=np.float)
            embedding[idx] = emb
        return torch.from_numpy(embedding)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        profile: torch.Tensor = None,
        profile_lengths: torch.Tensor = None,
        binary_labels: torch.Tensor = None,
        binary_labels_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Speaker Encoder + CI Scorer + CD Scorer + Decoder + Calc loss

        Args:
            speech: (Batch, samples) or (Batch, frames, input_size)
            speech_lengths: (Batch,) default None for chunk interator,
                                     because the chunk-iterator does not
                                     have the speech_lengths returned.
                                     see in
                                     espnet2/iterators/chunk_iter_factory.py
            profile: (Batch, N_spk, dim)
            profile_lengths: (Batch,)
            binary_labels: (Batch, frames, max_spk_num)
            binary_labels_lengths: (Batch,)
        """
        assert speech.shape[0] == binary_labels.shape[0], (speech.shape, binary_labels.shape)
        batch_size = speech.shape[0]
        self.forward_steps = self.forward_steps + 1
        # 1. Network forward
        pred, inter_outputs = self.prediction_forward(
            speech, speech_lengths,
            profile, profile_lengths,
            return_inter_outputs=True
        )
        (speech, speech_lengths), (profile, profile_lengths), (ci_score, cd_score) = inter_outputs

        # 2. Aggregate time-domain labels to match forward outputs
        if self.label_aggregator is not None:
            binary_labels, binary_labels_lengths = self.label_aggregator(
                binary_labels, binary_labels_lengths
            )
        # 2. Calculate power-set encoding (PSE) labels
        raw_pse_labels = torch.sum(binary_labels * self.power_weight, dim=2, keepdim=True)
        pse_labels = torch.argmax((raw_pse_labels.int() == self.int_token_arr).float(), dim=2)

        # If encoder uses conv* as input_layer (i.e., subsampling),
        # the sequence length of 'pred' might be slightly less than the
        # length of 'spk_labels'. Here we force them to be equal.
        length_diff_tolerance = 2
        length_diff = pse_labels.shape[1] - pred.shape[1]
        if 0 < length_diff <= length_diff_tolerance:
            pse_labels = pse_labels[:, 0: pred.shape[1]]

        loss_diar = self.classification_loss(pred, pse_labels, binary_labels_lengths)
        loss_spk_dis = self.speaker_discrimination_loss(profile, profile_lengths)
        loss_inter_ci, loss_inter_cd = self.internal_score_loss(cd_score, ci_score, pse_labels, binary_labels_lengths)
        label_mask = make_pad_mask(binary_labels_lengths, maxlen=pse_labels.shape[1]).to(pse_labels.device)
        loss = (loss_diar + self.speaker_discrimination_loss_weight * loss_spk_dis
                + self.inter_score_loss_weight * (loss_inter_ci + loss_inter_cd))

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
        ) = self.calc_diarization_error(
            pred=F.embedding(pred.argmax(dim=2) * (~label_mask), self.pse_embedding),
            label=F.embedding(pse_labels * (~label_mask), self.pse_embedding),
            length=binary_labels_lengths
        )

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
            loss_diar=loss_diar.detach() if loss_diar is not None else None,
            loss_spk_dis=loss_spk_dis.detach() if loss_spk_dis is not None else None,
            loss_inter_ci=loss_inter_ci.detach() if loss_inter_ci is not None else None,
            loss_inter_cd=loss_inter_cd.detach() if loss_inter_cd is not None else None,
            sad_mr=sad_mr,
            sad_fr=sad_fr,
            mi=mi,
            fa=fa,
            cf=cf,
            acc=acc,
            der=der,
            forward_steps=self.forward_steps,
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def classification_loss(
            self,
            predictions: torch.Tensor,
            labels: torch.Tensor,
            prediction_lengths: torch.Tensor
    ) -> torch.Tensor:
        mask = make_pad_mask(prediction_lengths, maxlen=labels.shape[1])
        pad_labels = labels.masked_fill(
            mask.to(predictions.device),
            value=self.ignore_id
        )
        loss = self.criterion_diar(predictions.contiguous(), pad_labels)

        return loss

    def speaker_discrimination_loss(
            self,
            profile: torch.Tensor,
            profile_lengths: torch.Tensor
    ) -> torch.Tensor:
        profile_mask = (torch.linalg.norm(profile, ord=2, dim=2, keepdim=True) > 0).float()  # (B, N, 1)
        mask = torch.matmul(profile_mask, profile_mask.transpose(1, 2))  # (B, N, N)
        mask = mask * (1.0 - torch.eye(self.max_spk_num).unsqueeze(0).to(mask))

        eps = 1e-12
        coding_norm = torch.linalg.norm(
            profile * profile_mask + (1 - profile_mask) * eps,
            dim=2, keepdim=True
        ) * profile_mask
        # profile: Batch, N, dim
        cos_theta = F.cosine_similarity(profile.unsqueeze(2), profile.unsqueeze(1), dim=-1, eps=eps) * mask
        cos_theta = torch.clip(cos_theta, -1 + eps, 1 - eps)
        loss = (F.relu(mask * coding_norm * (cos_theta - 0.0))).sum() / mask.sum()

        return loss

    def calculate_multi_labels(self, pse_labels, pse_labels_lengths):
        mask = make_pad_mask(pse_labels_lengths, maxlen=pse_labels.shape[1])
        padding_labels = pse_labels.masked_fill(
            mask.to(pse_labels.device),
            value=0
        ).to(pse_labels)
        multi_labels = F.embedding(padding_labels, self.pse_embedding)

        return multi_labels

    def internal_score_loss(
            self,
            cd_score: torch.Tensor,
            ci_score: torch.Tensor,
            pse_labels: torch.Tensor,
            pse_labels_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        multi_labels = self.calculate_multi_labels(pse_labels, pse_labels_lengths)
        ci_loss = self.criterion_bce(ci_score, multi_labels, pse_labels_lengths)
        cd_loss = self.criterion_bce(cd_score, multi_labels, pse_labels_lengths)
        return ci_loss, cd_loss

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        profile: torch.Tensor = None,
        profile_lengths: torch.Tensor = None,
        binary_labels: torch.Tensor = None,
        binary_labels_lengths: torch.Tensor = None,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        return ci_simi, cd_simi

    def post_net_forward(self, simi, seq_len):
        logits = self.decoder(simi, seq_len)[0]

        return logits

    def prediction_forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            profile: torch.Tensor,
            profile_lengths: torch.Tensor,
            return_inter_outputs: bool = False,
    ) -> [torch.Tensor, Optional[list]]:
        # speech encoding
        speech, speech_lengths = self.encode_speech(speech, speech_lengths)
        # speaker encoding
        profile, profile_lengths = self.encode_speaker(profile, profile_lengths)
        # calculating similarity
        ci_simi, cd_simi = self.calc_similarity(speech, profile, speech_lengths, profile_lengths)
        similarity = torch.cat([cd_simi, ci_simi], dim=2)
        # post net forward
        logits = self.post_net_forward(similarity, speech_lengths)

        if return_inter_outputs:
            return logits, [(speech, speech_lengths), (profile, profile_lengths), (ci_simi, cd_simi)]
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
            encoder_outputs = self.encoder(feats, feats_lengths)
            encoder_out, encoder_out_lens = encoder_outputs[:2]

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
        mask = ~make_pad_mask(length, maxlen=label.shape[1]).unsqueeze(-1).numpy()

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
