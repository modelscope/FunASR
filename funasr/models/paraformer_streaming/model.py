#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import time
import torch
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


if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


@tables.register("model_classes", "ParaformerStreaming")
class ParaformerStreaming(Paraformer):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        # import pdb;
        # pdb.set_trace()
        self.sampling_ratio = kwargs.get("sampling_ratio", 0.2)

        self.scama_mask = None
        if (
            hasattr(self.encoder, "overlap_chunk_cls")
            and self.encoder.overlap_chunk_cls is not None
        ):
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
        # import pdb;
        # pdb.set_trace()
        decoding_ind = kwargs.get("decoding_ind")
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        # Encoder
        if hasattr(self.encoder, "overlap_chunk_cls"):
            ind = self.encoder.overlap_chunk_cls.random_choice(self.training, decoding_ind)
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, ind=ind)
        else:
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        loss_ctc, cer_ctc = None, None
        loss_pre = None
        stats = dict()

        # decoder: CTC branch

        if self.ctc_weight > 0.0:
            if hasattr(self.encoder, "overlap_chunk_cls"):
                encoder_out_ctc, encoder_out_lens_ctc = self.encoder.overlap_chunk_cls.remove_chunk(
                    encoder_out, encoder_out_lens, chunk_outs=None
                )
            else:
                encoder_out_ctc, encoder_out_lens_ctc = encoder_out, encoder_out_lens

            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out_ctc, encoder_out_lens_ctc, text, text_lengths
            )
            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # decoder: Attention decoder branch
        loss_att, acc_att, cer_att, wer_att, loss_pre, pre_loss_att = self._calc_att_predictor_loss(
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
        stats["pre_loss_att"] = pre_loss_att.detach() if pre_loss_att is not None else None
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

    def _calc_att_predictor_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        if self.predictor_bias == 1:
            _, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = ys_pad_lens + self.predictor_bias
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
            ys_pad,
            encoder_out_mask,
            ignore_id=self.ignore_id,
            mask_chunk_predictor=mask_chunk_predictor,
            target_label_length=ys_pad_lens,
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
                target_length=ys_pad_lens,
                is_training=self.training,
            )
        elif self.encoder.overlap_chunk_cls is not None:
            encoder_out, encoder_out_lens = self.encoder.overlap_chunk_cls.remove_chunk(
                encoder_out, encoder_out_lens, chunk_outs=None
            )
        # 0. sampler
        decoder_out_1st = None
        pre_loss_att = None
        if self.sampling_ratio > 0.0:

            if self.use_1st_decoder_loss:
                sematic_embeds, decoder_out_1st, pre_loss_att = self.sampler_with_grad(
                    encoder_out,
                    encoder_out_lens,
                    ys_pad,
                    ys_pad_lens,
                    pre_acoustic_embeds,
                    scama_mask,
                )
            else:
                sematic_embeds, decoder_out_1st = self.sampler(
                    encoder_out,
                    encoder_out_lens,
                    ys_pad,
                    ys_pad_lens,
                    pre_acoustic_embeds,
                    scama_mask,
                )
        else:
            sematic_embeds = pre_acoustic_embeds

        # 1. Forward decoder
        decoder_outs = self.decoder(
            encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens, scama_mask
        )
        decoder_out, _ = decoder_outs[0], decoder_outs[1]

        if decoder_out_1st is None:
            decoder_out_1st = decoder_out
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(
            decoder_out_1st.view(-1, self.vocab_size),
            ys_pad,
            ignore_label=self.ignore_id,
        )
        loss_pre = self.criterion_pre(ys_pad_lens.type_as(pre_token_length), pre_token_length)

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out_1st.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att, loss_pre, pre_loss_att

    def sampler(
        self,
        encoder_out,
        encoder_out_lens,
        ys_pad,
        ys_pad_lens,
        pre_acoustic_embeds,
        chunk_mask=None,
    ):

        tgt_mask = (~make_pad_mask(ys_pad_lens, maxlen=ys_pad_lens.max())[:, :, None]).to(
            ys_pad.device
        )
        ys_pad_masked = ys_pad * tgt_mask[:, :, 0]
        if self.share_embedding:
            ys_pad_embed = self.decoder.output_layer.weight[ys_pad_masked]
        else:
            ys_pad_embed = self.decoder.embed(ys_pad_masked)
        with torch.no_grad():
            decoder_outs = self.decoder(
                encoder_out, encoder_out_lens, pre_acoustic_embeds, ys_pad_lens, chunk_mask
            )
            decoder_out, _ = decoder_outs[0], decoder_outs[1]
            pred_tokens = decoder_out.argmax(-1)
            nonpad_positions = ys_pad.ne(self.ignore_id)
            seq_lens = (nonpad_positions).sum(1)
            same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
            input_mask = torch.ones_like(nonpad_positions)
            bsz, seq_len = ys_pad.size()
            for li in range(bsz):
                target_num = (
                    ((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio
                ).long()
                if target_num > 0:
                    input_mask[li].scatter_(
                        dim=0, index=torch.randperm(seq_lens[li])[:target_num].cuda(), value=0
                    )
            input_mask = input_mask.eq(1)
            input_mask = input_mask.masked_fill(~nonpad_positions, False)
            input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)

        sematic_embeds = pre_acoustic_embeds.masked_fill(
            ~input_mask_expand_dim, 0
        ) + ys_pad_embed.masked_fill(input_mask_expand_dim, 0)
        return sematic_embeds * tgt_mask, decoder_out * tgt_mask

    def calc_predictor(self, encoder_out, encoder_out_lens):

        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        mask_chunk_predictor = None
        if self.encoder.overlap_chunk_cls is not None:
            mask_chunk_predictor = self.encoder.overlap_chunk_cls.get_mask_chunk_predictor(
                None, device=encoder_out.device, batch_size=encoder_out.size(0)
            )
            mask_shfit_chunk = self.encoder.overlap_chunk_cls.get_mask_shfit_chunk(
                None, device=encoder_out.device, batch_size=encoder_out.size(0)
            )
            encoder_out = encoder_out * mask_shfit_chunk
        pre_acoustic_embeds, pre_token_length, pre_alphas, pre_peak_index = self.predictor(
            encoder_out,
            None,
            encoder_out_mask,
            ignore_id=self.ignore_id,
            mask_chunk_predictor=mask_chunk_predictor,
            target_label_length=None,
        )
        predictor_alignments, predictor_alignments_len = self.predictor.gen_frame_alignments(
            pre_alphas,
            encoder_out_lens + 1 if self.predictor.tail_threshold > 0.0 else encoder_out_lens,
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
                target_length=None,
                is_training=self.training,
            )
        self.scama_mask = scama_mask

        return pre_acoustic_embeds, pre_token_length, pre_alphas, pre_peak_index

    def calc_predictor_chunk(self, encoder_out, encoder_out_lens, cache=None, **kwargs):
        is_final = kwargs.get("is_final", False)

        return self.predictor.forward_chunk(encoder_out, cache["encoder"], is_final=is_final)

    def cal_decoder_with_predictor(
        self, encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens
    ):
        decoder_outs = self.decoder(
            encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens, self.scama_mask
        )
        decoder_out = decoder_outs[0]
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        return decoder_out, ys_pad_lens

    def cal_decoder_with_predictor_chunk(
        self, encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens, cache=None
    ):
        decoder_outs = self.decoder.forward_chunk(encoder_out, sematic_embeds, cache["decoder"])
        decoder_out = decoder_outs
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        return decoder_out, ys_pad_lens

    def init_cache(self, cache: dict = {}, **kwargs):
        chunk_size = kwargs.get("chunk_size", [0, 10, 5])
        encoder_chunk_look_back = kwargs.get("encoder_chunk_look_back", 0)
        decoder_chunk_look_back = kwargs.get("decoder_chunk_look_back", 0)
        batch_size = 1

        enc_output_size = kwargs["encoder_conf"]["output_size"]
        feats_dims = kwargs["frontend_conf"]["n_mels"] * kwargs["frontend_conf"]["lfr_m"]
        cache_encoder = {
            "start_idx": 0,
            "cif_hidden": torch.zeros((batch_size, 1, enc_output_size)),
            "cif_alphas": torch.zeros((batch_size, 1)),
            "chunk_size": chunk_size,
            "encoder_chunk_look_back": encoder_chunk_look_back,
            "last_chunk": False,
            "opt": None,
            "feats": torch.zeros((batch_size, chunk_size[0] + chunk_size[2], feats_dims)),
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
        cache["prev_samples"] = torch.empty(0)

        return cache

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

        # predictor
        predictor_outs = self.calc_predictor_chunk(
            encoder_out, encoder_out_lens, cache=cache, is_final=kwargs.get("is_final", False)
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
        decoder_outs = self.cal_decoder_with_predictor_chunk(
            encoder_out, encoder_out_lens, pre_acoustic_embeds, pre_token_length, cache=cache
        )
        decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]

        results = []
        b, n, d = decoder_out.size()
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        for i in range(b):
            x = encoder_out[i, : encoder_out_lens[i], :]
            am_scores = decoder_out[i, : pre_token_length[i], :]
            if self.beam_search is not None:
                nbest_hyps = self.beam_search(
                    x=x,
                    am_scores=am_scores,
                    maxlenratio=kwargs.get("maxlenratio", 0.0),
                    minlenratio=kwargs.get("minlenratio", 0.0),
                )

                nbest_hyps = nbest_hyps[: self.nbest]
            else:

                yseq = am_scores.argmax(dim=-1)
                score = am_scores.max(dim=-1)[0]
                score = torch.sum(score, dim=-1)
                # pad with mask tokens to ensure compatibility with sos/eos tokens
                yseq = torch.tensor([self.sos] + yseq.tolist() + [self.eos], device=yseq.device)
                nbest_hyps = [Hypothesis(yseq=yseq, score=score)]
            for nbest_idx, hyp in enumerate(nbest_hyps):

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
        if self.beam_search is None and (is_use_lm or is_use_ctc):
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
            if kwargs["is_final"] and len(audio_sample_i) < 960:
                cache["encoder"]["tail_chunk"] = True
                speech = cache["encoder"]["feats"]
                speech_lengths = torch.tensor([speech.shape[1]], dtype=torch.int64).to(
                    speech.device
                )
            else:
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
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"{1}best_recog"]
            ibest_writer["token"][key[0]] = " ".join(tokens)
            ibest_writer["text"][key[0]] = text_postprocessed

        return result, meta_data

    def export(self, **kwargs):
        from .export_meta import export_rebuild_model

        models = export_rebuild_model(model=self, **kwargs)
        return models
