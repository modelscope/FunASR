#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import time
import torch
import logging
from contextlib import contextmanager
from typing import Dict, Optional, Tuple
from distutils.version import LooseVersion

from funasr.register import tables
from funasr.utils import postprocess_utils
from funasr.utils.datadir_writer import DatadirWriter
from funasr.train_utils.device_funcs import force_gatherable
from funasr.models.transformer.scorers.ctc import CTCPrefixScorer
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.transformer.scorers.length_bonus import LengthBonus
from funasr.models.transformer.utils.nets_utils import get_transducer_task_io
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.models.transducer.beam_search_transducer import BeamSearchTransducer


if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


@tables.register("model_classes", "Transducer")
class Transducer(torch.nn.Module):
    def __init__(
        self,
        frontend: Optional[str] = None,
        frontend_conf: Optional[Dict] = None,
        specaug: Optional[str] = None,
        specaug_conf: Optional[Dict] = None,
        normalize: str = None,
        normalize_conf: Optional[Dict] = None,
        encoder: str = None,
        encoder_conf: Optional[Dict] = None,
        decoder: str = None,
        decoder_conf: Optional[Dict] = None,
        joint_network: str = None,
        joint_network_conf: Optional[Dict] = None,
        transducer_weight: float = 1.0,
        fastemit_lambda: float = 0.0,
        auxiliary_ctc_weight: float = 0.0,
        auxiliary_ctc_dropout_rate: float = 0.0,
        auxiliary_lm_loss_weight: float = 0.0,
        auxiliary_lm_loss_smoothing: float = 0.0,
        input_size: int = 80,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        # report_cer: bool = True,
        # report_wer: bool = True,
        # sym_space: str = "<space>",
        # sym_blank: str = "<blank>",
        # extract_feats_in_collect_stats: bool = True,
        share_embedding: bool = False,
        # preencoder: Optional[AbsPreEncoder] = None,
        # postencoder: Optional[AbsPostEncoder] = None,
        **kwargs,
    ):

        super().__init__()

        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)
        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        decoder_class = tables.decoder_classes.get(decoder)
        decoder = decoder_class(
            vocab_size=vocab_size,
            **decoder_conf,
        )
        decoder_output_size = decoder.output_size

        joint_network_class = tables.joint_network_classes.get(joint_network)
        joint_network = joint_network_class(
            vocab_size,
            encoder_output_size,
            decoder_output_size,
            **joint_network_conf,
        )

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
        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.decoder = decoder
        self.joint_network = joint_network

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None
        self.ctc = None
        self.ctc_weight = 0.0

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]
        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if (
            hasattr(self.encoder, "overlap_chunk_cls")
            and self.encoder.overlap_chunk_cls is not None
        ):
            encoder_out, encoder_out_lens = self.encoder.overlap_chunk_cls.remove_chunk(
                encoder_out, encoder_out_lens, chunk_outs=None
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
        joint_out = self.joint_network(encoder_out.unsqueeze(2), decoder_out.unsqueeze(1))

        # 5. Losses
        loss_trans, cer_trans, wer_trans = self._calc_transducer_loss(
            encoder_out,
            joint_out,
            target,
            t_len,
            u_len,
        )

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
        )

        stats = dict(
            loss=loss.detach(),
            loss_transducer=loss_trans.detach(),
            aux_ctc_loss=loss_ctc.detach() if loss_ctc > 0.0 else None,
            aux_lm_loss=loss_lm.detach() if loss_lm > 0.0 else None,
            cer_transducer=cer_trans,
            wer_transducer=wer_trans,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """
        with autocast(False):

            # Data augmentation
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)

            # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        # Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens

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
                from warp_rnnt import rnnt_loss as RNNTLoss

                self.criterion_transducer = RNNTLoss

            except ImportError:
                logging.error(
                    "warp-rnnt was not installed." "Please consult the installation documentation."
                )
                exit(1)

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
                from funasr.metrics import ErrorCalculatorTransducer as ErrorCalculator

                self.error_calculator = ErrorCalculator(
                    self.decoder,
                    self.joint_network,
                    self.token_list,
                    self.sym_space,
                    self.sym_blank,
                    report_cer=self.report_cer,
                    report_wer=self.report_wer,
                )

            cer_transducer, wer_transducer = self.error_calculator(encoder_out, target, t_len)

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
        ctc_in = self.ctc_lin(torch.nn.functional.dropout(encoder_out, p=self.ctc_dropout_rate))
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
        loss_lm = loss_lm.masked_fill(ignore.unsqueeze(1), 0).sum() / decoder_out.size(0)

        return loss_lm

    def init_beam_search(
        self,
        **kwargs,
    ):

        # 1. Build ASR model
        scorers = {}

        if self.ctc != None:
            ctc = CTCPrefixScorer(ctc=self.ctc, eos=self.eos)
            scorers.update(ctc=ctc)
        token_list = kwargs.get("token_list")
        scorers.update(
            length_bonus=LengthBonus(len(token_list)),
        )

        # 3. Build ngram model
        # ngram is not supported now
        ngram = None
        scorers["ngram"] = ngram

        beam_search = BeamSearchTransducer(
            self.decoder,
            self.joint_network,
            kwargs.get("beam_size", 2),
            nbest=1,
        )
        # beam_search.to(device=kwargs.get("device", "cpu"), dtype=getattr(torch, kwargs.get("dtype", "float32"))).eval()
        # for scorer in scorers.values():
        #     if isinstance(scorer, torch.nn.Module):
        #         scorer.to(device=kwargs.get("device", "cpu"), dtype=getattr(torch, kwargs.get("dtype", "float32"))).eval()
        self.beam_search = beam_search

    def inference(
        self,
        data_in: list,
        data_lengths: list = None,
        key: list = None,
        tokenizer=None,
        **kwargs,
    ):

        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        # init beamsearch
        is_use_ctc = kwargs.get("decoding_ctc_weight", 0.0) > 0.00001 and self.ctc != None
        is_use_lm = (
            kwargs.get("lm_weight", 0.0) > 0.00001 and kwargs.get("lm_file", None) is not None
        )
        # if self.beam_search is None and (is_use_lm or is_use_ctc):
        logging.info("enable beam_search")
        self.init_beam_search(**kwargs)
        self.nbest = kwargs.get("nbest", 1)

        meta_data = {}
        # extract fbank feats
        time1 = time.perf_counter()
        audio_sample_list = load_audio_text_image_video(
            data_in, fs=self.frontend.fs, audio_fs=kwargs.get("fs", 16000)
        )
        time2 = time.perf_counter()
        meta_data["load_data"] = f"{time2 - time1:0.3f}"
        speech, speech_lengths = extract_fbank(
            audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=self.frontend
        )
        time3 = time.perf_counter()
        meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
        meta_data["batch_data_time"] = (
            speech_lengths.sum().item() * self.frontend.frame_shift * self.frontend.lfr_n / 1000
        )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        # Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # c. Passed the encoder result and the beam search
        nbest_hyps = self.beam_search(encoder_out[0], is_final=True)
        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        b, n, d = encoder_out.size()
        for i in range(b):

            for nbest_idx, hyp in enumerate(nbest_hyps):
                ibest_writer = None
                if kwargs.get("output_dir") is not None:
                    if not hasattr(self, "writer"):
                        self.writer = DatadirWriter(kwargs.get("output_dir"))
                    ibest_writer = self.writer[f"{nbest_idx + 1}best_recog"]
                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq  # [1:last_pos]
                else:
                    token_int = hyp.yseq  # [1:last_pos].tolist()

                # remove blank symbol id, which is assumed to be 0
                token_int = list(
                    filter(
                        lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int
                    )
                )

                # Change integer-ids to tokens
                token = tokenizer.ids2tokens(token_int)
                text = tokenizer.tokens2text(token)

                text_postprocessed, _ = postprocess_utils.sentence_postprocess(token)
                result_i = {
                    "key": key[i],
                    "token": token,
                    "text": text,
                    "text_postprocessed": text_postprocessed,
                }
                results.append(result_i)

                if ibest_writer is not None:
                    ibest_writer["token"][key[i]] = " ".join(token)
                    ibest_writer["text"][key[i]] = text
                    ibest_writer["text_postprocessed"][key[i]] = text_postprocessed

        return results, meta_data
