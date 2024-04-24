#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import time
import torch
import logging
from torch.cuda.amp import autocast
from typing import Union, Dict, List, Tuple, Optional

from funasr.register import tables
from funasr.models.ctc.ctc import CTC
from funasr.utils import postprocess_utils
from funasr.metrics.compute_acc import th_accuracy
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.paraformer.cif_predictor import mae_loss
from funasr.train_utils.device_funcs import force_gatherable
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.models.transformer.utils.nets_utils import make_pad_mask, pad_list
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.models.scama.utils import sequence_mask


@tables.register("model_classes", "UniASR")
class UniASR(torch.nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    """

    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        encoder: str = None,
        encoder_conf: dict = None,
        encoder2: str = None,
        encoder2_conf: dict = None,
        decoder: str = None,
        decoder_conf: dict = None,
        decoder2: str = None,
        decoder2_conf: dict = None,
        predictor: str = None,
        predictor_conf: dict = None,
        predictor_bias: int = 0,
        predictor_weight: float = 0.0,
        predictor2: str = None,
        predictor2_conf: dict = None,
        predictor2_bias: int = 0,
        predictor2_weight: float = 0.0,
        ctc: str = None,
        ctc_conf: dict = None,
        ctc_weight: float = 0.5,
        ctc2: str = None,
        ctc2_conf: dict = None,
        ctc2_weight: float = 0.5,
        decoder_attention_chunk_type: str = "chunk",
        decoder_attention_chunk_type2: str = "chunk",
        stride_conv=None,
        stride_conv_conf: dict = None,
        loss_weight_model1: float = 0.5,
        input_size: int = 80,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        share_embedding: bool = False,
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
            encoder_output_size=encoder_output_size,
            **decoder_conf,
        )
        predictor_class = tables.predictor_classes.get(predictor)
        predictor = predictor_class(**predictor_conf)

        from funasr.models.transformer.utils.subsampling import Conv1dSubsampling

        stride_conv = Conv1dSubsampling(
            **stride_conv_conf,
            idim=input_size + encoder_output_size,
            odim=input_size + encoder_output_size,
        )
        stride_conv_output_size = stride_conv.output_size()

        encoder_class = tables.encoder_classes.get(encoder2)
        encoder2 = encoder_class(input_size=stride_conv_output_size, **encoder2_conf)
        encoder2_output_size = encoder2.output_size()

        decoder_class = tables.decoder_classes.get(decoder2)
        decoder2 = decoder_class(
            vocab_size=vocab_size,
            encoder_output_size=encoder2_output_size,
            **decoder2_conf,
        )
        predictor_class = tables.predictor_classes.get(predictor2)
        predictor2 = predictor_class(**predictor2_conf)

        self.blank_id = blank_id
        self.sos = sos
        self.eos = eos
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.ctc2_weight = ctc2_weight

        self.specaug = specaug
        self.normalize = normalize

        self.encoder = encoder

        self.error_calculator = None

        self.decoder = decoder
        self.ctc = None
        self.ctc2 = None

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.predictor = predictor
        self.predictor_weight = predictor_weight
        self.criterion_pre = mae_loss(normalize_length=length_normalized_loss)
        self.encoder1_encoder2_joint_training = kwargs.get("encoder1_encoder2_joint_training", True)

        if self.encoder.overlap_chunk_cls is not None:
            from funasr.models.scama.chunk_utilis import (
                build_scama_mask_for_cross_attention_decoder,
            )

            self.build_scama_mask_for_cross_attention_decoder_fn = (
                build_scama_mask_for_cross_attention_decoder
            )
            self.decoder_attention_chunk_type = decoder_attention_chunk_type

        self.encoder2 = encoder2
        self.decoder2 = decoder2
        self.ctc2_weight = ctc2_weight

        self.predictor2 = predictor2
        self.predictor2_weight = predictor2_weight
        self.decoder_attention_chunk_type2 = decoder_attention_chunk_type2
        self.stride_conv = stride_conv
        self.loss_weight_model1 = loss_weight_model1
        if self.encoder2.overlap_chunk_cls is not None:
            from funasr.models.scama.chunk_utilis import (
                build_scama_mask_for_cross_attention_decoder,
            )

            self.build_scama_mask_for_cross_attention_decoder_fn2 = (
                build_scama_mask_for_cross_attention_decoder
            )
            self.decoder_attention_chunk_type2 = decoder_attention_chunk_type2

        self.length_normalized_loss = length_normalized_loss
        self.enable_maas_finetune = kwargs.get("enable_maas_finetune", False)
        self.freeze_encoder2 = kwargs.get("freeze_encoder2", False)
        self.beam_search = None

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss
        Args:
                        speech: (Batch, Length, ...)
                        speech_lengths: (Batch, )
                        text: (Batch, Length)
                        text_lengths: (Batch,)
        """
        decoding_ind = kwargs.get("decoding_ind", None)
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        ind = self.encoder.overlap_chunk_cls.random_choice(self.training, decoding_ind)
        # 1. Encoder
        if self.enable_maas_finetune:
            with torch.no_grad():
                speech_raw, encoder_out, encoder_out_lens = self.encode(
                    speech, speech_lengths, ind=ind
                )
        else:
            speech_raw, encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, ind=ind)

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        stats = dict()
        loss_pre = None
        loss, loss1, loss2 = 0.0, 0.0, 0.0

        if self.loss_weight_model1 > 0.0:
            ## model1
            # 1. CTC branch
            if self.enable_maas_finetune:
                with torch.no_grad():

                    loss_att, acc_att, cer_att, wer_att, loss_pre = self._calc_att_predictor_loss(
                        encoder_out, encoder_out_lens, text, text_lengths
                    )

                    loss = loss_att + loss_pre * self.predictor_weight

                    # Collect Attn branch stats
                    stats["loss_att"] = loss_att.detach() if loss_att is not None else None
                    stats["acc"] = acc_att
                    stats["cer"] = cer_att
                    stats["wer"] = wer_att
                    stats["loss_pre"] = loss_pre.detach().cpu() if loss_pre is not None else None
            else:

                loss_att, acc_att, cer_att, wer_att, loss_pre = self._calc_att_predictor_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

                loss = loss_att + loss_pre * self.predictor_weight

                # Collect Attn branch stats
                stats["loss_att"] = loss_att.detach() if loss_att is not None else None
                stats["acc"] = acc_att
                stats["cer"] = cer_att
                stats["wer"] = wer_att
                stats["loss_pre"] = loss_pre.detach().cpu() if loss_pre is not None else None

        loss1 = loss

        if self.loss_weight_model1 < 1.0:
            ## model2

            # encoder2
            if self.freeze_encoder2:
                with torch.no_grad():
                    encoder_out, encoder_out_lens = self.encode2(
                        encoder_out, encoder_out_lens, speech_raw, speech_lengths, ind=ind
                    )
            else:
                encoder_out, encoder_out_lens = self.encode2(
                    encoder_out, encoder_out_lens, speech_raw, speech_lengths, ind=ind
                )

            intermediate_outs = None
            if isinstance(encoder_out, tuple):
                intermediate_outs = encoder_out[1]
                encoder_out = encoder_out[0]

            loss_att, acc_att, cer_att, wer_att, loss_pre = self._calc_att_predictor_loss2(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            loss = loss_att + loss_pre * self.predictor2_weight

            # Collect Attn branch stats
            stats["loss_att2"] = loss_att.detach() if loss_att is not None else None
            stats["acc2"] = acc_att
            stats["cer2"] = cer_att
            stats["wer2"] = wer_att
            stats["loss_pre2"] = loss_pre.detach().cpu() if loss_pre is not None else None

        loss2 = loss

        loss = loss1 * self.loss_weight_model1 + loss2 * (1 - self.loss_weight_model1)
        stats["loss1"] = torch.clone(loss1.detach())
        stats["loss2"] = torch.clone(loss2.detach())
        stats["loss"] = torch.clone(loss.detach())
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
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
        **kwargs,
    ):
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
                        speech: (Batch, Length, ...)
                        speech_lengths: (Batch, )
        """
        ind = kwargs.get("ind", 0)
        with autocast(False):
            # Data augmentation
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)

            # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        speech_raw = speech.clone().to(speech.device)

        # 4. Forward encoder
        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths, ind=ind)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return speech_raw, encoder_out, encoder_out_lens

    def encode2(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ):
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
                        speech: (Batch, Length, ...)
                        speech_lengths: (Batch, )
        """

        ind = kwargs.get("ind", 0)
        encoder_out_rm, encoder_out_lens_rm = self.encoder.overlap_chunk_cls.remove_chunk(
            encoder_out,
            encoder_out_lens,
            chunk_outs=None,
        )
        # residual_input
        encoder_out = torch.cat((speech, encoder_out_rm), dim=-1)
        encoder_out_lens = encoder_out_lens_rm
        if self.stride_conv is not None:
            speech, speech_lengths = self.stride_conv(encoder_out, encoder_out_lens)
        if not self.encoder1_encoder2_joint_training:
            speech = speech.detach()
            speech_lengths = speech_lengths.detach()
        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)

        encoder_out, encoder_out_lens, _ = self.encoder2(speech, speech_lengths, ind=ind)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, encoder_out_lens

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder
        Normally, this function is called in batchify_nll.
        Args:
                        encoder_out: (Batch, Length, Dim)
                        encoder_out_lens: (Batch,)
                        ys_pad: (Batch, Length)
                        ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder
        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
                        encoder_out: (Batch, Length, Dim)
                        encoder_out_lens: (Batch,)
                        ys_pad: (Batch, Length)
                        ys_pad_lens: (Batch,)
                        batch_size: int, samples each batch contain when computing nll,
                                                                        you may change this to avoid OOM or increase
                                                                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_att_predictor_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        encoder_out_mask = sequence_mask(
            encoder_out_lens,
            maxlen=encoder_out.size(1),
            dtype=encoder_out.dtype,
            device=encoder_out.device,
        )[:, None, :]
        mask_chunk_predictor = None
        if self.encoder.overlap_chunk_cls is not None:
            mask_chunk_predictor = self.encoder.overlap_chunk_cls.get_mask_chunk_predictor(
                None, device=encoder_out.device, batch_size=encoder_out.size(0)
            )
            mask_shfit_chunk = self.encoder.overlap_chunk_cls.get_mask_shfit_chunk(
                None, device=encoder_out.device, batch_size=encoder_out.size(0)
            )
            encoder_out = encoder_out * mask_shfit_chunk
        pre_acoustic_embeds, pre_token_length, pre_alphas, _ = self.predictor(
            encoder_out,
            ys_out_pad,
            encoder_out_mask,
            ignore_id=self.ignore_id,
            mask_chunk_predictor=mask_chunk_predictor,
            target_label_length=ys_in_lens,
        )
        predictor_alignments, predictor_alignments_len = self.predictor.gen_frame_alignments(
            pre_alphas, encoder_out_lens
        )

        scama_mask = None
        if (
            self.encoder.overlap_chunk_cls is not None
            and self.decoder_attention_chunk_type == "chunk"
        ):
            encoder_chunk_size = self.encoder.overlap_chunk_cls.chunk_size_pad_shift_cur
            attention_chunk_center_bias = 0
            attention_chunk_size = encoder_chunk_size
            decoder_att_look_back_factor = (
                self.encoder.overlap_chunk_cls.decoder_att_look_back_factor_cur
            )
            mask_shift_att_chunk_decoder = (
                self.encoder.overlap_chunk_cls.get_mask_shift_att_chunk_decoder(
                    None, device=encoder_out.device, batch_size=encoder_out.size(0)
                )
            )
            scama_mask = self.build_scama_mask_for_cross_attention_decoder_fn(
                predictor_alignments=predictor_alignments,
                encoder_sequence_length=encoder_out_lens,
                chunk_size=1,
                encoder_chunk_size=encoder_chunk_size,
                attention_chunk_center_bias=attention_chunk_center_bias,
                attention_chunk_size=attention_chunk_size,
                attention_chunk_type=self.decoder_attention_chunk_type,
                step=None,
                predictor_mask_chunk_hopping=mask_chunk_predictor,
                decoder_att_look_back_factor=decoder_att_look_back_factor,
                mask_shift_att_chunk_decoder=mask_shift_att_chunk_decoder,
                target_length=ys_in_lens,
                is_training=self.training,
            )
        elif self.encoder.overlap_chunk_cls is not None:
            encoder_out, encoder_out_lens = self.encoder.overlap_chunk_cls.remove_chunk(
                encoder_out, encoder_out_lens, chunk_outs=None
            )
        # try:
        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out,
            encoder_out_lens,
            ys_in_pad,
            ys_in_lens,
            chunk_mask=scama_mask,
            pre_acoustic_embeds=pre_acoustic_embeds,
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        # predictor loss
        loss_pre = self.criterion_pre(ys_in_lens.type_as(pre_token_length), pre_token_length)
        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att, loss_pre

    def _calc_att_predictor_loss2(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        encoder_out_mask = sequence_mask(
            encoder_out_lens,
            maxlen=encoder_out.size(1),
            dtype=encoder_out.dtype,
            device=encoder_out.device,
        )[:, None, :]
        mask_chunk_predictor = None
        if self.encoder2.overlap_chunk_cls is not None:
            mask_chunk_predictor = self.encoder2.overlap_chunk_cls.get_mask_chunk_predictor(
                None, device=encoder_out.device, batch_size=encoder_out.size(0)
            )
            mask_shfit_chunk = self.encoder2.overlap_chunk_cls.get_mask_shfit_chunk(
                None, device=encoder_out.device, batch_size=encoder_out.size(0)
            )
            encoder_out = encoder_out * mask_shfit_chunk
        pre_acoustic_embeds, pre_token_length, pre_alphas, _ = self.predictor2(
            encoder_out,
            ys_out_pad,
            encoder_out_mask,
            ignore_id=self.ignore_id,
            mask_chunk_predictor=mask_chunk_predictor,
            target_label_length=ys_in_lens,
        )
        predictor_alignments, predictor_alignments_len = self.predictor2.gen_frame_alignments(
            pre_alphas, encoder_out_lens
        )

        scama_mask = None
        if (
            self.encoder2.overlap_chunk_cls is not None
            and self.decoder_attention_chunk_type2 == "chunk"
        ):
            encoder_chunk_size = self.encoder2.overlap_chunk_cls.chunk_size_pad_shift_cur
            attention_chunk_center_bias = 0
            attention_chunk_size = encoder_chunk_size
            decoder_att_look_back_factor = (
                self.encoder2.overlap_chunk_cls.decoder_att_look_back_factor_cur
            )
            mask_shift_att_chunk_decoder = (
                self.encoder2.overlap_chunk_cls.get_mask_shift_att_chunk_decoder(
                    None, device=encoder_out.device, batch_size=encoder_out.size(0)
                )
            )
            scama_mask = self.build_scama_mask_for_cross_attention_decoder_fn2(
                predictor_alignments=predictor_alignments,
                encoder_sequence_length=encoder_out_lens,
                chunk_size=1,
                encoder_chunk_size=encoder_chunk_size,
                attention_chunk_center_bias=attention_chunk_center_bias,
                attention_chunk_size=attention_chunk_size,
                attention_chunk_type=self.decoder_attention_chunk_type2,
                step=None,
                predictor_mask_chunk_hopping=mask_chunk_predictor,
                decoder_att_look_back_factor=decoder_att_look_back_factor,
                mask_shift_att_chunk_decoder=mask_shift_att_chunk_decoder,
                target_length=ys_in_lens,
                is_training=self.training,
            )
        elif self.encoder2.overlap_chunk_cls is not None:
            encoder_out, encoder_out_lens = self.encoder2.overlap_chunk_cls.remove_chunk(
                encoder_out, encoder_out_lens, chunk_outs=None
            )
        # try:
        # 1. Forward decoder
        decoder_out, _ = self.decoder2(
            encoder_out,
            encoder_out_lens,
            ys_in_pad,
            ys_in_lens,
            chunk_mask=scama_mask,
            pre_acoustic_embeds=pre_acoustic_embeds,
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        # predictor loss
        loss_pre = self.criterion_pre(ys_in_lens.type_as(pre_token_length), pre_token_length)
        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att, loss_pre

    def calc_predictor_mask(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor = None,
        ys_pad_lens: torch.Tensor = None,
    ):
        # ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        # ys_in_lens = ys_pad_lens + 1
        ys_out_pad, ys_in_lens = None, None

        encoder_out_mask = sequence_mask(
            encoder_out_lens,
            maxlen=encoder_out.size(1),
            dtype=encoder_out.dtype,
            device=encoder_out.device,
        )[:, None, :]
        mask_chunk_predictor = None
        if self.encoder.overlap_chunk_cls is not None:
            mask_chunk_predictor = self.encoder.overlap_chunk_cls.get_mask_chunk_predictor(
                None, device=encoder_out.device, batch_size=encoder_out.size(0)
            )
            mask_shfit_chunk = self.encoder.overlap_chunk_cls.get_mask_shfit_chunk(
                None, device=encoder_out.device, batch_size=encoder_out.size(0)
            )
            encoder_out = encoder_out * mask_shfit_chunk
        pre_acoustic_embeds, pre_token_length, pre_alphas, _ = self.predictor(
            encoder_out,
            ys_out_pad,
            encoder_out_mask,
            ignore_id=self.ignore_id,
            mask_chunk_predictor=mask_chunk_predictor,
            target_label_length=ys_in_lens,
        )
        predictor_alignments, predictor_alignments_len = self.predictor.gen_frame_alignments(
            pre_alphas, encoder_out_lens
        )

        scama_mask = None
        if (
            self.encoder.overlap_chunk_cls is not None
            and self.decoder_attention_chunk_type == "chunk"
        ):
            encoder_chunk_size = self.encoder.overlap_chunk_cls.chunk_size_pad_shift_cur
            attention_chunk_center_bias = 0
            attention_chunk_size = encoder_chunk_size
            decoder_att_look_back_factor = (
                self.encoder.overlap_chunk_cls.decoder_att_look_back_factor_cur
            )
            mask_shift_att_chunk_decoder = (
                self.encoder.overlap_chunk_cls.get_mask_shift_att_chunk_decoder(
                    None, device=encoder_out.device, batch_size=encoder_out.size(0)
                )
            )
            scama_mask = self.build_scama_mask_for_cross_attention_decoder_fn(
                predictor_alignments=predictor_alignments,
                encoder_sequence_length=encoder_out_lens,
                chunk_size=1,
                encoder_chunk_size=encoder_chunk_size,
                attention_chunk_center_bias=attention_chunk_center_bias,
                attention_chunk_size=attention_chunk_size,
                attention_chunk_type=self.decoder_attention_chunk_type,
                step=None,
                predictor_mask_chunk_hopping=mask_chunk_predictor,
                decoder_att_look_back_factor=decoder_att_look_back_factor,
                mask_shift_att_chunk_decoder=mask_shift_att_chunk_decoder,
                target_length=ys_in_lens,
                is_training=self.training,
            )
        elif self.encoder.overlap_chunk_cls is not None:
            encoder_out, encoder_out_lens = self.encoder.overlap_chunk_cls.remove_chunk(
                encoder_out, encoder_out_lens, chunk_outs=None
            )

        return (
            pre_acoustic_embeds,
            pre_token_length,
            predictor_alignments,
            predictor_alignments_len,
            scama_mask,
        )

    def calc_predictor_mask2(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor = None,
        ys_pad_lens: torch.Tensor = None,
    ):
        # ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        # ys_in_lens = ys_pad_lens + 1
        ys_out_pad, ys_in_lens = None, None

        encoder_out_mask = sequence_mask(
            encoder_out_lens,
            maxlen=encoder_out.size(1),
            dtype=encoder_out.dtype,
            device=encoder_out.device,
        )[:, None, :]
        mask_chunk_predictor = None
        if self.encoder2.overlap_chunk_cls is not None:
            mask_chunk_predictor = self.encoder2.overlap_chunk_cls.get_mask_chunk_predictor(
                None, device=encoder_out.device, batch_size=encoder_out.size(0)
            )
            mask_shfit_chunk = self.encoder2.overlap_chunk_cls.get_mask_shfit_chunk(
                None, device=encoder_out.device, batch_size=encoder_out.size(0)
            )
            encoder_out = encoder_out * mask_shfit_chunk
        pre_acoustic_embeds, pre_token_length, pre_alphas, _ = self.predictor2(
            encoder_out,
            ys_out_pad,
            encoder_out_mask,
            ignore_id=self.ignore_id,
            mask_chunk_predictor=mask_chunk_predictor,
            target_label_length=ys_in_lens,
        )
        predictor_alignments, predictor_alignments_len = self.predictor2.gen_frame_alignments(
            pre_alphas, encoder_out_lens
        )

        scama_mask = None
        if (
            self.encoder2.overlap_chunk_cls is not None
            and self.decoder_attention_chunk_type2 == "chunk"
        ):
            encoder_chunk_size = self.encoder2.overlap_chunk_cls.chunk_size_pad_shift_cur
            attention_chunk_center_bias = 0
            attention_chunk_size = encoder_chunk_size
            decoder_att_look_back_factor = (
                self.encoder2.overlap_chunk_cls.decoder_att_look_back_factor_cur
            )
            mask_shift_att_chunk_decoder = (
                self.encoder2.overlap_chunk_cls.get_mask_shift_att_chunk_decoder(
                    None, device=encoder_out.device, batch_size=encoder_out.size(0)
                )
            )
            scama_mask = self.build_scama_mask_for_cross_attention_decoder_fn2(
                predictor_alignments=predictor_alignments,
                encoder_sequence_length=encoder_out_lens,
                chunk_size=1,
                encoder_chunk_size=encoder_chunk_size,
                attention_chunk_center_bias=attention_chunk_center_bias,
                attention_chunk_size=attention_chunk_size,
                attention_chunk_type=self.decoder_attention_chunk_type2,
                step=None,
                predictor_mask_chunk_hopping=mask_chunk_predictor,
                decoder_att_look_back_factor=decoder_att_look_back_factor,
                mask_shift_att_chunk_decoder=mask_shift_att_chunk_decoder,
                target_length=ys_in_lens,
                is_training=self.training,
            )
        elif self.encoder2.overlap_chunk_cls is not None:
            encoder_out, encoder_out_lens = self.encoder2.overlap_chunk_cls.remove_chunk(
                encoder_out, encoder_out_lens, chunk_outs=None
            )

        return (
            pre_acoustic_embeds,
            pre_token_length,
            predictor_alignments,
            predictor_alignments_len,
            scama_mask,
        )

    def init_beam_search(
        self,
        **kwargs,
    ):
        from funasr.models.uniasr.beam_search import BeamSearchScama
        from funasr.models.transformer.scorers.ctc import CTCPrefixScorer
        from funasr.models.transformer.scorers.length_bonus import LengthBonus

        decoding_mode = kwargs.get("decoding_mode", "model1")
        if decoding_mode == "model1":
            decoder = self.decoder
        else:
            decoder = self.decoder2
        # 1. Build ASR model
        scorers = {}

        if self.ctc != None:
            ctc = CTCPrefixScorer(ctc=self.ctc, eos=self.eos)
            scorers.update(ctc=ctc)
        token_list = kwargs.get("token_list")
        scorers.update(
            decoder=decoder,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 3. Build ngram model
        # ngram is not supported now
        ngram = None
        scorers["ngram"] = ngram

        weights = dict(
            decoder=1.0 - kwargs.get("decoding_ctc_weight", 0.0),
            ctc=kwargs.get("decoding_ctc_weight", 0.0),
            lm=kwargs.get("lm_weight", 0.0),
            ngram=kwargs.get("ngram_weight", 0.0),
            length_bonus=kwargs.get("penalty", 0.0),
        )
        beam_search = BeamSearchScama(
            beam_size=kwargs.get("beam_size", 5),
            weights=weights,
            scorers=scorers,
            sos=self.sos,
            eos=self.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if self.ctc_weight == 1.0 else "full",
        )

        self.beam_search = beam_search

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):

        decoding_model = kwargs.get("decoding_model", "normal")
        token_num_relax = kwargs.get("token_num_relax", 5)
        if decoding_model == "fast":
            decoding_ind = 0
            decoding_mode = "model1"
        elif decoding_model == "offline":
            decoding_ind = 1
            decoding_mode = "model2"
        else:
            decoding_ind = 0
            decoding_mode = "model2"
        # init beamsearch

        if self.beam_search is None:
            logging.info("enable beam_search")
            self.init_beam_search(decoding_mode=decoding_mode, **kwargs)
            self.nbest = kwargs.get("nbest", 1)

        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = (
                speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])
        speech_raw = speech.clone().to(device=kwargs["device"])
        # Encoder
        _, encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, ind=decoding_ind)
        if decoding_mode == "model1":
            predictor_outs = self.calc_predictor_mask(encoder_out, encoder_out_lens)
        else:
            encoder_out, encoder_out_lens = self.encode2(
                encoder_out, encoder_out_lens, speech_raw, speech_lengths, ind=decoding_ind
            )
            predictor_outs = self.calc_predictor_mask2(encoder_out, encoder_out_lens)

        scama_mask = predictor_outs[4]
        pre_token_length = predictor_outs[1]
        pre_acoustic_embeds = predictor_outs[0]
        maxlen = pre_token_length.sum().item() + token_num_relax
        minlen = max(0, pre_token_length.sum().item() - token_num_relax)
        # c. Passed the encoder result and the beam search
        nbest_hyps = self.beam_search(
            x=encoder_out[0],
            scama_mask=scama_mask,
            pre_acoustic_embeds=pre_acoustic_embeds,
            maxlenratio=0.0,
            minlenratio=0.0,
            maxlen=int(maxlen),
            minlen=int(minlen),
        )

        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:

            # remove sos/eos and get results
            last_pos = -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[1:last_pos]
            else:
                token_int = hyp.yseq[1:last_pos].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = tokenizer.ids2tokens(token_int)
            text_postprocessed = tokenizer.tokens2text(token)
            if not hasattr(tokenizer, "bpemodel"):
                text_postprocessed, _ = postprocess_utils.sentence_postprocess(token)

            result_i = {"key": key[0], "text": text_postprocessed}
            results.append(result_i)

        return results, meta_data
