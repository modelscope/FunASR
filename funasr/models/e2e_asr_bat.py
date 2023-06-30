"""Boundary Aware Transducer (BAT) model."""

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from funasr.losses.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.models.decoder.rnnt_decoder import RNNTDecoder
from funasr.models.decoder.abs_decoder import AbsDecoder as AbsAttDecoder
from funasr.models.encoder.abs_encoder import AbsEncoder
from funasr.models.joint_net.joint_network import JointNetwork
from funasr.modules.nets_utils import get_transducer_task_io
from funasr.modules.nets_utils import th_accuracy
from funasr.modules.nets_utils import make_pad_mask
from funasr.modules.add_sos_eos import add_sos_eos
from funasr.layers.abs_normalize import AbsNormalize
from funasr.torch_utils.device_funcs import force_gatherable
from funasr.models.base_model import FunASRModel

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:

    @contextmanager
    def autocast(enabled=True):
        yield


class BATModel(FunASRModel):
    """BATModel module definition.

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
        encoder: AbsEncoder,
        decoder: RNNTDecoder,
        joint_network: JointNetwork,
        att_decoder: Optional[AbsAttDecoder] = None,
        predictor = None,
        transducer_weight: float = 1.0,
        predictor_weight: float = 1.0,
        cif_weight: float = 1.0,
        fastemit_lambda: float = 0.0,
        auxiliary_ctc_weight: float = 0.0,
        auxiliary_ctc_dropout_rate: float = 0.0,
        auxiliary_lm_loss_weight: float = 0.0,
        auxiliary_lm_loss_smoothing: float = 0.0,
        ignore_id: int = -1,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        report_cer: bool = True,
        report_wer: bool = True,
        extract_feats_in_collect_stats: bool = True,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        r_d: int = 5,
        r_u: int = 5,
    ) -> None:
        """Construct an BATModel object."""
        super().__init__()

        # The following labels ID are reserved: 0 (blank) and vocab_size - 1 (sos/eos)
        self.blank_id = 0
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
        self.use_auxiliary_lm_loss = auxiliary_lm_loss_weight > 0

        if self.use_auxiliary_ctc:
            self.ctc_lin = torch.nn.Linear(encoder.output_size(), vocab_size)
            self.ctc_dropout_rate = auxiliary_ctc_dropout_rate

        if self.use_auxiliary_lm_loss:
            self.lm_lin = torch.nn.Linear(decoder.output_size, vocab_size)
            self.lm_loss_smoothing = auxiliary_lm_loss_smoothing

        self.transducer_weight = transducer_weight
        self.fastemit_lambda = fastemit_lambda

        self.auxiliary_ctc_weight = auxiliary_ctc_weight
        self.auxiliary_lm_loss_weight = auxiliary_lm_loss_weight

        self.report_cer = report_cer
        self.report_wer = report_wer

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        self.criterion_pre = torch.nn.L1Loss()
        self.predictor_weight = predictor_weight
        self.predictor = predictor
        
        self.cif_weight = cif_weight
        if self.cif_weight > 0:
            self.cif_output_layer = torch.nn.Linear(encoder.output_size(), vocab_size)
            self.criterion_cif = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )        
        self.r_d = r_d
        self.r_u = r_u

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

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if hasattr(self.encoder, 'overlap_chunk_cls') and self.encoder.overlap_chunk_cls is not None:
            encoder_out, encoder_out_lens = self.encoder.overlap_chunk_cls.remove_chunk(encoder_out, encoder_out_lens,
                                                                                        chunk_outs=None)

        encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(encoder_out.device)
        # 2. Transducer-related I/O preparation
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            text,
            encoder_out_lens,
            ignore_id=self.ignore_id,
        )

        # 3. Decoder
        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in, u_len)

        pre_acoustic_embeds, pre_token_length, _, pre_peak_index = self.predictor(encoder_out, text, encoder_out_mask, ignore_id=self.ignore_id)
        loss_pre = self.criterion_pre(text_lengths.type_as(pre_token_length), pre_token_length)

        if self.cif_weight > 0.0:
            cif_predict = self.cif_output_layer(pre_acoustic_embeds)
            loss_cif = self.criterion_cif(cif_predict, text)
        else:
            loss_cif = 0.0

        # 5. Losses
        boundary = torch.zeros((encoder_out.size(0), 4), dtype=torch.int64, device=encoder_out.device)
        boundary[:, 2] = u_len.long().detach()
        boundary[:, 3] = t_len.long().detach()

        pre_peak_index = torch.floor(pre_peak_index).long()
        s_begin = pre_peak_index - self.r_d

        T = encoder_out.size(1)
        B = encoder_out.size(0)
        U = decoder_out.size(1)

        mask = torch.arange(0, T, device=encoder_out.device).reshape(1, T).expand(B, T)
        mask = mask <= boundary[:, 3].reshape(B, 1) - 1

        s_begin_padding = boundary[:, 2].reshape(B, 1) - (self.r_u+self.r_d) + 1
        # handle the cases where `len(symbols) < s_range`
        s_begin_padding = torch.clamp(s_begin_padding, min=0)

        s_begin = torch.where(mask, s_begin, s_begin_padding)
        
        mask2 = s_begin <  boundary[:, 2].reshape(B, 1) - (self.r_u+self.r_d) + 1

        s_begin = torch.where(mask2, s_begin, boundary[:, 2].reshape(B, 1) - (self.r_u+self.r_d) + 1)

        s_begin = torch.clamp(s_begin, min=0)
        
        ranges = s_begin.reshape((B, T, 1)).expand((B, T, min(self.r_u+self.r_d, min(u_len)))) + torch.arange(min(self.r_d+self.r_u, min(u_len)), device=encoder_out.device)

        import fast_rnnt
        am_pruned, lm_pruned = fast_rnnt.do_rnnt_pruning(
            am=self.joint_network.lin_enc(encoder_out),
            lm=self.joint_network.lin_dec(decoder_out),
            ranges=ranges,
        )

        logits = self.joint_network(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            loss_trans = fast_rnnt.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=target.long(),
                ranges=ranges,
                termination_symbol=self.blank_id,
                boundary=boundary,
                reduction="sum",
            )

        cer_trans, wer_trans = None, None
        if not self.training and (self.report_cer or self.report_wer):
            if self.error_calculator is None:
                from funasr.modules.e2e_asr_common import ErrorCalculatorTransducer as ErrorCalculator
                self.error_calculator = ErrorCalculator(
                    self.decoder,
                    self.joint_network,
                    self.token_list,
                    self.sym_space,
                    self.sym_blank,
                    report_cer=self.report_cer,
                    report_wer=self.report_wer,
                )
            cer_trans, wer_trans = self.error_calculator(encoder_out, target, t_len)

        loss_ctc, loss_lm = 0.0, 0.0

        if self.use_auxiliary_ctc:
            loss_ctc = self._calc_ctc_loss(
                encoder_out,
                target,
                t_len,
                u_len,
            )

        if self.use_auxiliary_lm_loss:
            loss_lm = self._calc_lm_loss(decoder_out, target)

        loss = (
            self.transducer_weight * loss_trans
            + self.auxiliary_ctc_weight * loss_ctc
            + self.auxiliary_lm_loss_weight * loss_lm
            + self.predictor_weight * loss_pre
            + self.cif_weight * loss_cif
        )

        stats = dict(
            loss=loss.detach(),
            loss_transducer=loss_trans.detach(),
            loss_pre=loss_pre.detach(),
            loss_cif=loss_cif.detach() if loss_cif > 0.0 else None,
            aux_ctc_loss=loss_ctc.detach() if loss_ctc > 0.0 else None,
            aux_lm_loss=loss_lm.detach() if loss_lm > 0.0 else None,
            cer_transducer=cer_trans,
            wer_transducer=wer_trans,
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
