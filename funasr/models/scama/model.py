#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import time
import torch
import torch.nn as nn
import torch.functional as F
import logging
from typing import Dict, Tuple
from contextlib import contextmanager
from distutils.version import LooseVersion

from funasr.register import tables
from funasr.models.ctc.ctc import CTC
from funasr.utils import postprocess_utils
from funasr.metrics.compute_acc import th_accuracy
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.paraformer.model import Paraformer
from funasr.models.paraformer.search import Hypothesis
from funasr.models.paraformer.cif_predictor import mae_loss
from funasr.train_utils.device_funcs import force_gatherable
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.models.transformer.utils.nets_utils import make_pad_mask, pad_list
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.models.scama.utils import sequence_mask

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


@tables.register("model_classes", "SCAMA")
class SCAMA(nn.Module):
    """
    Author: Shiliang Zhang, Zhifu Gao, Haoneng Luo, Ming Lei, Jie Gao, Zhijie Yan, Lei Xie
    SCAMA: Streaming chunk-aware multihead attention for online end-to-end speech recognition
    https://arxiv.org/abs/2006.01712
    """

    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        encoder: str = None,
        encoder_conf: dict = None,
        decoder: str = None,
        decoder_conf: dict = None,
        ctc: str = None,
        ctc_conf: dict = None,
        ctc_weight: float = 0.5,
        predictor: str = None,
        predictor_conf: dict = None,
        predictor_bias: int = 0,
        predictor_weight: float = 0.0,
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
        if ctc_weight > 0.0:

            if ctc_conf is None:
                ctc_conf = {}

            ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)

        predictor_class = tables.predictor_classes.get(predictor)
        predictor = predictor_class(**predictor_conf)

        # note that eos is the same as sos (equivalent ID)
        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight

        self.specaug = specaug
        self.normalize = normalize

        self.encoder = encoder

        if ctc_weight == 1.0:
            self.decoder = None
        else:
            self.decoder = decoder

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.predictor = predictor
        self.predictor_weight = predictor_weight
        self.predictor_bias = predictor_bias

        self.criterion_pre = mae_loss(normalize_length=length_normalized_loss)

        self.share_embedding = share_embedding
        if self.share_embedding:
            self.decoder.embed = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None
        self.error_calculator = None

        if self.encoder.overlap_chunk_cls is not None:
            from funasr.models.scama.chunk_utilis import (
                build_scama_mask_for_cross_attention_decoder,
            )

            self.build_scama_mask_for_cross_attention_decoder_fn = (
                build_scama_mask_for_cross_attention_decoder
            )
            self.decoder_attention_chunk_type = kwargs.get("decoder_attention_chunk_type", "chunk")

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

        decoding_ind = kwargs.get("decoding_ind")
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        # Encoder
        ind = self.encoder.overlap_chunk_cls.random_choice(self.training, decoding_ind)
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, ind=ind)

        loss_ctc, cer_ctc = None, None
        loss_pre = None
        stats = dict()

        # decoder: CTC branch

        if self.ctc_weight > 0.0:

            encoder_out_ctc, encoder_out_lens_ctc = self.encoder.overlap_chunk_cls.remove_chunk(
                encoder_out, encoder_out_lens, chunk_outs=None
            )

            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out_ctc, encoder_out_lens_ctc, text, text_lengths
            )
            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # decoder: Attention decoder branch
        loss_att, acc_att, cer_att, wer_att, loss_pre = self._calc_att_predictor_loss(
            encoder_out, encoder_out_lens, text, text_lengths
        )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att + loss_pre * self.predictor_weight
        else:
            loss = (
                self.ctc_weight * loss_ctc
                + (1 - self.ctc_weight) * loss_att
                + loss_pre * self.predictor_weight
            )

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att
        stats["loss_pre"] = loss_pre.detach().cpu() if loss_pre is not None else None

        stats["loss"] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = (text_lengths + self.predictor_bias).sum()
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder. Note that this method is used by asr_inference.py
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
        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, encoder_out_lens

    def encode_chunk(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        cache: dict = None,
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
        encoder_out, encoder_out_lens, _ = self.encoder.forward_chunk(
            speech, speech_lengths, cache=cache["encoder"]
        )
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, torch.tensor([encoder_out.size(1)])

    def calc_predictor_chunk(self, encoder_out, encoder_out_lens, cache=None, **kwargs):
        is_final = kwargs.get("is_final", False)

        return self.predictor.forward_chunk(encoder_out, cache["encoder"], is_final=is_final)

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

        from funasr.models.scama.beam_search import BeamSearchScamaStreaming

        from funasr.models.transformer.scorers.ctc import CTCPrefixScorer
        from funasr.models.transformer.scorers.length_bonus import LengthBonus

        # 1. Build ASR model
        scorers = {}

        if self.ctc != None:
            ctc = CTCPrefixScorer(ctc=self.ctc, eos=self.eos)
            scorers.update(ctc=ctc)
        token_list = kwargs.get("token_list")
        scorers.update(
            decoder=self.decoder,
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

        beam_search = BeamSearchScamaStreaming(
            beam_size=kwargs.get("beam_size", 2),
            weights=weights,
            scorers=scorers,
            sos=self.sos,
            eos=self.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if self.ctc_weight == 1.0 else "full",
        )
        # beam_search.to(device=kwargs.get("device", "cpu"), dtype=getattr(torch, kwargs.get("dtype", "float32"))).eval()
        # for scorer in scorers.values():
        #     if isinstance(scorer, torch.nn.Module):
        #         scorer.to(device=kwargs.get("device", "cpu"), dtype=getattr(torch, kwargs.get("dtype", "float32"))).eval()
        self.beam_search = beam_search

    def generate_chunk(
        self,
        speech,
        speech_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        cache = kwargs.get("cache", {})
        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        # Encoder
        encoder_out, encoder_out_lens = self.encode_chunk(
            speech, speech_lengths, cache=cache, is_final=kwargs.get("is_final", False)
        )
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        if "running_hyps" not in cache:
            running_hyps = self.beam_search.init_hyp(encoder_out)
            cache["running_hyps"] = running_hyps

        # predictor
        predictor_outs = self.calc_predictor_chunk(
            encoder_out,
            encoder_out_lens,
            cache=cache,
            is_final=kwargs.get("is_final", False),
        )
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = (
            predictor_outs[0],
            predictor_outs[1],
            predictor_outs[2],
            predictor_outs[3],
        )
        pre_token_length = pre_token_length.round().long()

        if torch.max(pre_token_length) < 1:
            return []
        maxlen = minlen = pre_token_length
        if kwargs.get("is_final", False):
            maxlen += kwargs.get("token_num_relax", 5)
            minlen = max(0, minlen - kwargs.get("token_num_relax", 5))
        # c. Passed the encoder result and the beam search
        nbest_hyps = self.beam_search(
            x=encoder_out[0],
            scama_mask=None,
            pre_acoustic_embeds=pre_acoustic_embeds,
            maxlen=int(maxlen),
            minlen=int(minlen),
            cache=cache,
        )

        cache["running_hyps"] = nbest_hyps
        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            # assert isinstance(hyp, (Hypothesis)), type(hyp)

            # remove sos/eos and get results
            last_pos = -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[1:last_pos]
            else:
                token_int = hyp.yseq[1:last_pos].tolist()

                # remove blank symbol id, which is assumed to be 0
                token_int = list(
                    filter(
                        lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int
                    )
                )

                # Change integer-ids to tokens
                token = tokenizer.ids2tokens(token_int)
                # text = tokenizer.tokens2text(token)

                result_i = token

                results.extend(result_i)

        return results

    def init_cache(self, cache: dict = {}, **kwargs):
        device = kwargs.get("device", "cuda")

        chunk_size = kwargs.get("chunk_size", [0, 10, 5])
        encoder_chunk_look_back = kwargs.get("encoder_chunk_look_back", 0)
        decoder_chunk_look_back = kwargs.get("decoder_chunk_look_back", 0)
        batch_size = 1

        enc_output_size = kwargs["encoder_conf"]["output_size"]
        feats_dims = kwargs["frontend_conf"]["n_mels"] * kwargs["frontend_conf"]["lfr_m"]

        cache_encoder = {
            "start_idx": 0,
            "cif_hidden": torch.zeros((batch_size, 1, enc_output_size)).to(device=device),
            "cif_alphas": torch.zeros((batch_size, 1)).to(device=device),
            "chunk_size": chunk_size,
            "encoder_chunk_look_back": encoder_chunk_look_back,
            "last_chunk": False,
            "opt": None,
            "feats": torch.zeros((batch_size, chunk_size[0] + chunk_size[2], feats_dims)).to(
                device=device
            ),
            "tail_chunk": False,
        }
        cache["encoder"] = cache_encoder

        cache_decoder = {
            "decode_fsmn": None,
            "decoder_chunk_look_back": decoder_chunk_look_back,
            "opt": None,
            "chunk_size": chunk_size,
        }
        cache["decoder"] = cache_decoder
        cache["frontend"] = {}

        cache["prev_samples"] = torch.empty(0).to(device=device)

        return cache

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        cache: dict = {},
        **kwargs,
    ):

        # init beamsearch
        is_use_ctc = kwargs.get("decoding_ctc_weight", 0.0) > 0.00001 and self.ctc != None
        is_use_lm = (
            kwargs.get("lm_weight", 0.0) > 0.00001 and kwargs.get("lm_file", None) is not None
        )

        if self.beam_search is None:

            logging.info("enable beam_search")
            self.init_beam_search(**kwargs)
            self.nbest = kwargs.get("nbest", 1)

        if len(cache) == 0:
            self.init_cache(cache, **kwargs)

        meta_data = {}
        chunk_size = kwargs.get("chunk_size", [0, 10, 5])
        chunk_stride_samples = int(chunk_size[1] * 960)  # 600ms

        time1 = time.perf_counter()
        cfg = {"is_final": kwargs.get("is_final", False)}
        audio_sample_list = load_audio_text_image_video(
            data_in,
            fs=frontend.fs,
            audio_fs=kwargs.get("fs", 16000),
            data_type=kwargs.get("data_type", "sound"),
            tokenizer=tokenizer,
            cache=cfg,
        )
        _is_final = cfg["is_final"]  # if data_in is a file or url, set is_final=True

        time2 = time.perf_counter()
        meta_data["load_data"] = f"{time2 - time1:0.3f}"
        assert len(audio_sample_list) == 1, "batch_size must be set 1"

        audio_sample = torch.cat((cache["prev_samples"], audio_sample_list[0]))

        n = int(len(audio_sample) // chunk_stride_samples + int(_is_final))
        m = int(len(audio_sample) % chunk_stride_samples * (1 - int(_is_final)))
        tokens = []
        for i in range(n):
            kwargs["is_final"] = _is_final and i == n - 1
            audio_sample_i = audio_sample[i * chunk_stride_samples : (i + 1) * chunk_stride_samples]

            # extract fbank feats
            speech, speech_lengths = extract_fbank(
                [audio_sample_i],
                data_type=kwargs.get("data_type", "sound"),
                frontend=frontend,
                cache=cache["frontend"],
                is_final=kwargs["is_final"],
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = (
                speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )

            tokens_i = self.generate_chunk(
                speech,
                speech_lengths,
                key=key,
                tokenizer=tokenizer,
                cache=cache,
                frontend=frontend,
                **kwargs,
            )
            tokens.extend(tokens_i)

        text_postprocessed, _ = postprocess_utils.sentence_postprocess(tokens)

        result_i = {"key": key[0], "text": text_postprocessed}
        result = [result_i]

        cache["prev_samples"] = audio_sample[:-m]
        if _is_final:
            self.init_cache(cache, **kwargs)

        if kwargs.get("output_dir"):
            writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = writer[f"{1}best_recog"]
            ibest_writer["token"][key[0]] = " ".join(tokens)
            ibest_writer["text"][key[0]] = text_postprocessed

        return result, meta_data
