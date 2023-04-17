"""ESPnet2 ASR Transducer model."""

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.models.rnnt_predictor.abs_decoder import AbsDecoder
from funasr.models.encoder.conformer_encoder import ConformerChunkEncoder as Encoder
from funasr.models.joint_net.joint_network import JointNetwork
from funasr.modules.nets_utils import get_transducer_task_io
from funasr.layers.abs_normalize import AbsNormalize
from funasr.torch_utils.device_funcs import force_gatherable
from funasr.train.abs_espnet_model import AbsESPnetModel
from funasr.modules.add_sos_eos import add_sos_eos
from funasr.models.decoder.abs_decoder import AbsDecoder as AbsAttDecoder
from funasr.modules.nets_utils import th_accuracy
from funasr.losses.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)
from funasr.modules.e2e_asr_common import ErrorCalculatorTransducer as ErrorCalculator
if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:

    @contextmanager
    def autocast(enabled=True):
        yield


class UnifiedTransducerModel(AbsESPnetModel):
    """ESPnet2ASRTransducerModel module definition.

    Args:
        vocab_size: Size of complete vocabulary (w/ EOS and blank included).
        token_list: List of token
        frontend: Frontend module.
        specaug: SpecAugment module.
        normalize: Normalization module.
        encoder: Encoder module.
        decoder: Decoder module.
        joint_network: Joint Network module.
        transducer_weight: Weight of the Transducer loss.
        fastemit_lambda: FastEmit lambda value.
        auxiliary_ctc_weight: Weight of auxiliary CTC loss.
        auxiliary_ctc_dropout_rate: Dropout rate for auxiliary CTC loss inputs.
        auxiliary_lm_loss_weight: Weight of auxiliary LM loss.
        auxiliary_lm_loss_smoothing: Smoothing rate for LM loss' label smoothing.
        ignore_id: Initial padding ID.
        sym_space: Space symbol.
        sym_blank: Blank Symbol
        report_cer: Whether to report Character Error Rate during validation.
        report_wer: Whether to report Word Error Rate during validation.
        extract_feats_in_collect_stats: Whether to use extract_feats stats collection.

    """

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: Encoder,
        decoder: AbsDecoder,
        att_decoder: Optional[AbsAttDecoder],
        joint_network: JointNetwork,
        transducer_weight: float = 1.0,
        fastemit_lambda: float = 0.0,
        auxiliary_ctc_weight: float = 0.0,
        auxiliary_att_weight: float = 0.0,
        auxiliary_ctc_dropout_rate: float = 0.0,
        auxiliary_lm_loss_weight: float = 0.0,
        auxiliary_lm_loss_smoothing: float = 0.0,
        ignore_id: int = -1,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        report_cer: bool = True,
        report_wer: bool = True,
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
    ) -> None:
        """Construct an ESPnetASRTransducerModel object."""
        super().__init__()

        assert check_argument_types()

        # The following labels ID are reserved: 0 (blank) and vocab_size - 1 (sos/eos)
        self.blank_id = 0

        if sym_sos in token_list:
            self.sos = token_list.index(sym_sos)
        else:
            self.sos = vocab_size - 1
        if sym_eos in token_list:
            self.eos = token_list.index(sym_eos)
        else:
            self.eos = vocab_size - 1

        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.token_list = token_list.copy()

        self.sym_space = sym_space
        self.sym_blank = sym_blank

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize

        self.encoder = encoder
        self.decoder = decoder
        self.joint_network = joint_network

        self.criterion_transducer = None
        self.error_calculator = None

        self.use_auxiliary_ctc = auxiliary_ctc_weight > 0
        self.use_auxiliary_att = auxiliary_att_weight > 0
        self.use_auxiliary_lm_loss = auxiliary_lm_loss_weight > 0

        if self.use_auxiliary_ctc:
            self.ctc_lin = torch.nn.Linear(encoder.output_size, vocab_size)
            self.ctc_dropout_rate = auxiliary_ctc_dropout_rate

        if self.use_auxiliary_att:
            self.att_decoder = att_decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

        if self.use_auxiliary_lm_loss:
            self.lm_lin = torch.nn.Linear(decoder.output_size, vocab_size)
            self.lm_loss_smoothing = auxiliary_lm_loss_smoothing

        self.transducer_weight = transducer_weight
        self.fastemit_lambda = fastemit_lambda

        self.auxiliary_ctc_weight = auxiliary_ctc_weight
        self.auxiliary_att_weight = auxiliary_att_weight
        self.auxiliary_lm_loss_weight = auxiliary_lm_loss_weight

        self.report_cer = report_cer
        self.report_wer = report_wer

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Forward architecture and compute loss(es).

        Args:
            speech: Speech sequences. (B, S)
            speech_lengths: Speech sequences lengths. (B,)
            text: Label ID sequences. (B, L)
            text_lengths: Label ID sequences lengths. (B,)
            kwargs: Contains "utts_id".

        Return:
            loss: Main loss value.
            stats: Task statistics.
            weight: Task weights.

        """
        assert text_lengths.dim() == 1, text_lengths.shape
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        batch_size = speech.shape[0]
        text = text[:, : text_lengths.max()]
        #print(speech.shape)
        # 1. Encoder
        encoder_out, encoder_out_chunk, encoder_out_lens = self.encode(speech, speech_lengths)

        loss_att, loss_att_chunk = 0.0, 0.0

        if self.use_auxiliary_att:
            loss_att, _ = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
            loss_att_chunk, _ = self._calc_att_loss(
                encoder_out_chunk, encoder_out_lens, text, text_lengths
            )

        # 2. Transducer-related I/O preparation
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            text,
            encoder_out_lens,
            ignore_id=self.ignore_id,
        )
        
        # 3. Decoder
        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in, u_len)

        # 4. Joint Network
        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )
        
        joint_out_chunk = self.joint_network(
            encoder_out_chunk.unsqueeze(2), decoder_out.unsqueeze(1)
        )

        # 5. Losses
        loss_trans_utt, cer_trans, wer_trans = self._calc_transducer_loss(
            encoder_out,
            joint_out,
            target,
            t_len,
            u_len,
        )
        
        loss_trans_chunk, cer_trans_chunk, wer_trans_chunk = self._calc_transducer_loss(
            encoder_out_chunk,
            joint_out_chunk,
            target,
            t_len,
            u_len,
        )

        loss_ctc, loss_ctc_chunk, loss_lm = 0.0, 0.0, 0.0

        if self.use_auxiliary_ctc:
            loss_ctc = self._calc_ctc_loss(
                encoder_out,
                target,
                t_len,
                u_len,
            )
            loss_ctc_chunk = self._calc_ctc_loss(
                encoder_out_chunk,
                target,
                t_len,
                u_len,
            )

        if self.use_auxiliary_lm_loss:
            loss_lm = self._calc_lm_loss(decoder_out, target)

        loss_trans = loss_trans_utt + loss_trans_chunk
        loss_ctc = loss_ctc + loss_ctc_chunk 
        loss_ctc = loss_att + loss_att_chunk

        loss = (
            self.transducer_weight * loss_trans
            + self.auxiliary_ctc_weight * loss_ctc
            + self.auxiliary_att_weight * loss_att
            + self.auxiliary_lm_loss_weight * loss_lm
        )

        stats = dict(
            loss=loss.detach(),
            loss_transducer=loss_trans_utt.detach(),
            loss_transducer_chunk=loss_trans_chunk.detach(),
            aux_ctc_loss=loss_ctc.detach() if loss_ctc > 0.0 else None,
            aux_ctc_loss_chunk=loss_ctc_chunk.detach() if loss_ctc_chunk > 0.0 else None,
            aux_att_loss=loss_att.detach() if loss_att > 0.0 else None,
            aux_att_loss_chunk=loss_att_chunk.detach() if loss_att_chunk > 0.0 else None,
            aux_lm_loss=loss_lm.detach() if loss_lm > 0.0 else None,
            cer_transducer=cer_trans,
            wer_transducer=wer_trans,
            cer_transducer_chunk=cer_trans_chunk,
            wer_transducer_chunk=wer_trans_chunk,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Collect features sequences and features lengths sequences.

        Args:
            speech: Speech sequences. (B, S)
            speech_lengths: Speech sequences lengths. (B,)
            text: Label ID sequences. (B, L)
            text_lengths: Label ID sequences lengths. (B,)
            kwargs: Contains "utts_id".

        Return:
            {}: "feats": Features sequences. (B, T, D_feats),
                "feats_lengths": Features sequences lengths. (B,)

        """
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )

            feats, feats_lengths = speech, speech_lengths

        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder speech sequences.

        Args:
            speech: Speech sequences. (B, S)
            speech_lengths: Speech sequences lengths. (B,)

        Return:
            encoder_out: Encoder outputs. (B, T, D_enc)
            encoder_out_lens: Encoder outputs lengths. (B,)

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
        encoder_out, encoder_out_chunk, encoder_out_lens = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_chunk, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features sequences and features sequences lengths.

        Args:
            speech: Speech sequences. (B, S)
            speech_lengths: Speech sequences lengths. (B,)

        Return:
            feats: Features sequences. (B, T, D_feats)
            feats_lengths: Features sequences lengths. (B,)

        """
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            feats, feats_lengths = speech, speech_lengths

        return feats, feats_lengths

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        joint_out: torch.Tensor,
        target: torch.Tensor,
        t_len: torch.Tensor,
        u_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[float], Optional[float]]:
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            joint_out: Joint Network output sequences (B, T, U, D_joint)
            target: Target label ID sequences. (B, L)
            t_len: Encoder output sequences lengths. (B,)
            u_len: Target label ID sequences lengths. (B,)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        if self.criterion_transducer is None:
            try:
                # from warprnnt_pytorch import RNNTLoss
	        # self.criterion_transducer = RNNTLoss(
                    # reduction="mean",
                    # fastemit_lambda=self.fastemit_lambda,
                # )
                from warp_rnnt import rnnt_loss as RNNTLoss
                self.criterion_transducer = RNNTLoss

            except ImportError:
                logging.error(
                    "warp-rnnt was not installed."
                    "Please consult the installation documentation."
                )
                exit(1)

        # loss_transducer = self.criterion_transducer(
        #     joint_out,
        #     target,
        #     t_len,
        #     u_len,
        # )
        log_probs = torch.log_softmax(joint_out, dim=-1)

        loss_transducer = self.criterion_transducer(
                log_probs,
                target,
                t_len,
                u_len,
                reduction="mean",
                blank=self.blank_id,
                fastemit_lambda=self.fastemit_lambda,
                gather=True,
        )

        if not self.training and (self.report_cer or self.report_wer):
            if self.error_calculator is None:
                self.error_calculator = ErrorCalculator(
                    self.decoder,
                    self.joint_network,
                    self.token_list,
                    self.sym_space,
                    self.sym_blank,
                    report_cer=self.report_cer,
                    report_wer=self.report_wer,
                )

            cer_transducer, wer_transducer = self.error_calculator(encoder_out, target)
            return loss_transducer, cer_transducer, wer_transducer

        return loss_transducer, None, None

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        target: torch.Tensor,
        t_len: torch.Tensor,
        u_len: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            target: Target label ID sequences. (B, L)
            t_len: Encoder output sequences lengths. (B,)
            u_len: Target label ID sequences lengths. (B,)

        Return:
            loss_ctc: CTC loss value.

        """
        ctc_in = self.ctc_lin(
            torch.nn.functional.dropout(encoder_out, p=self.ctc_dropout_rate)
        )
        ctc_in = torch.log_softmax(ctc_in.transpose(0, 1), dim=-1)

        target_mask = target != 0
        ctc_target = target[target_mask].cpu()

        with torch.backends.cudnn.flags(deterministic=True):
            loss_ctc = torch.nn.functional.ctc_loss(
                ctc_in,
                ctc_target,
                t_len,
                u_len,
                zero_infinity=True,
                reduction="sum",
            )
        loss_ctc /= target.size(0)

        return loss_ctc

    def _calc_lm_loss(
        self,
        decoder_out: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LM loss.

        Args:
            decoder_out: Decoder output sequences. (B, U, D_dec)
            target: Target label ID sequences. (B, L)

        Return:
            loss_lm: LM loss value.

        """
        lm_loss_in = self.lm_lin(decoder_out[:, :-1, :]).view(-1, self.vocab_size)
        lm_target = target.view(-1).type(torch.int64)

        with torch.no_grad():
            true_dist = lm_loss_in.clone()
            true_dist.fill_(self.lm_loss_smoothing / (self.vocab_size - 1))

            # Ignore blank ID (0)
            ignore = lm_target == 0
            lm_target = lm_target.masked_fill(ignore, 0)

            true_dist.scatter_(1, lm_target.unsqueeze(1), (1 - self.lm_loss_smoothing))

        loss_lm = torch.nn.functional.kl_div(
            torch.log_softmax(lm_loss_in, dim=1),
            true_dist,
            reduction="none",
        )
        loss_lm = loss_lm.masked_fill(ignore.unsqueeze(1), 0).sum() / decoder_out.size(
            0
        )

        return loss_lm

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        if hasattr(self, "lang_token_id") and self.lang_token_id is not None:
            ys_pad = torch.cat(
                [
                    self.lang_token_id.repeat(ys_pad.size(0), 1).to(ys_pad.device),
                    ys_pad,
                ],
                dim=1,
            )
            ys_pad_lens += 1

        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.att_decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        return loss_att, acc_att
