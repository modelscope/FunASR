#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os
import re
import time
import torch
import codecs
import logging
import tempfile
import requests
import numpy as np
from typing import Dict, Tuple
from contextlib import contextmanager
from distutils.version import LooseVersion

from funasr.register import tables
from funasr.utils import postprocess_utils
from funasr.metrics.compute_acc import th_accuracy
from funasr.models.paraformer.model import Paraformer
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.paraformer.search import Hypothesis
from funasr.train_utils.device_funcs import force_gatherable
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


@tables.register("model_classes", "ContextualParaformer")
class ContextualParaformer(Paraformer):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    FunASR: A Fundamental End-to-End Speech Recognition Toolkit
    https://arxiv.org/abs/2305.11013
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.target_buffer_length = kwargs.get("target_buffer_length", -1)
        inner_dim = kwargs.get("inner_dim", 256)
        bias_encoder_type = kwargs.get("bias_encoder_type", "lstm")
        use_decoder_embedding = kwargs.get("use_decoder_embedding", False)
        crit_attn_weight = kwargs.get("crit_attn_weight", 0.0)
        crit_attn_smooth = kwargs.get("crit_attn_smooth", 0.0)
        bias_encoder_dropout_rate = kwargs.get("bias_encoder_dropout_rate", 0.0)

        if bias_encoder_type == "lstm":
            self.bias_encoder = torch.nn.LSTM(
                inner_dim, inner_dim, 1, batch_first=True, dropout=bias_encoder_dropout_rate
            )
            self.bias_embed = torch.nn.Embedding(self.vocab_size, inner_dim)
        elif bias_encoder_type == "mean":
            self.bias_embed = torch.nn.Embedding(self.vocab_size, inner_dim)
        else:
            logging.error("Unsupport bias encoder type: {}".format(bias_encoder_type))

        if self.target_buffer_length > 0:
            self.hotword_buffer = None
            self.length_record = []
            self.current_buffer_length = 0
        self.use_decoder_embedding = use_decoder_embedding
        self.crit_attn_weight = crit_attn_weight
        if self.crit_attn_weight > 0:
            self.attn_loss = torch.nn.L1Loss()
        self.crit_attn_smooth = crit_attn_smooth

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
        text_lengths = text_lengths.squeeze()
        speech_lengths = speech_lengths.squeeze()

        batch_size = speech.shape[0]

        hotword_pad = kwargs.get("hotword_pad")
        hotword_lengths = kwargs.get("hotword_lengths")
        # dha_pad = kwargs.get("dha_pad")

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

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

        # 2b. Attention decoder branch
        loss_att, acc_att, cer_att, wer_att, loss_pre, loss_ideal = self._calc_att_clas_loss(
            encoder_out, encoder_out_lens, text, text_lengths, hotword_pad, hotword_lengths
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

        if loss_ideal is not None:
            loss = loss + loss_ideal * self.crit_attn_weight
            stats["loss_ideal"] = loss_ideal.detach().cpu()

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att
        stats["loss_pre"] = loss_pre.detach().cpu() if loss_pre is not None else None

        stats["loss"] = torch.clone(loss.detach())
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + self.predictor_bias).sum())

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _calc_att_clas_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        hotword_pad: torch.Tensor,
        hotword_lengths: torch.Tensor,
    ):
        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)

        if self.predictor_bias == 1:
            _, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = ys_pad_lens + self.predictor_bias

        pre_acoustic_embeds, pre_token_length, _, _ = self.predictor(
            encoder_out, ys_pad, encoder_out_mask, ignore_id=self.ignore_id
        )
        # -1. bias encoder
        if self.use_decoder_embedding:
            hw_embed = self.decoder.embed(hotword_pad)
        else:
            hw_embed = self.bias_embed(hotword_pad)

        hw_embed, (_, _) = self.bias_encoder(hw_embed)
        _ind = np.arange(0, hotword_pad.shape[0]).tolist()
        selected = hw_embed[_ind, [i - 1 for i in hotword_lengths.detach().cpu().tolist()]]
        contextual_info = selected.squeeze(0).repeat(ys_pad.shape[0], 1, 1).to(ys_pad.device)

        # 0. sampler
        decoder_out_1st = None
        if self.sampling_ratio > 0.0:

            sematic_embeds, decoder_out_1st = self.sampler(
                encoder_out,
                encoder_out_lens,
                ys_pad,
                ys_pad_lens,
                pre_acoustic_embeds,
                contextual_info,
            )
        else:
            sematic_embeds = pre_acoustic_embeds

        # 1. Forward decoder
        decoder_outs = self.decoder(
            encoder_out,
            encoder_out_lens,
            sematic_embeds,
            ys_pad_lens,
            contextual_info=contextual_info,
        )
        decoder_out, _ = decoder_outs[0], decoder_outs[1]
        """
        if self.crit_attn_weight > 0 and attn.shape[-1] > 1:
            ideal_attn = ideal_attn + self.crit_attn_smooth / (self.crit_attn_smooth + 1.0)
            attn_non_blank = attn[:,:,:,:-1]
            ideal_attn_non_blank = ideal_attn[:,:,:-1]
            loss_ideal = self.attn_loss(attn_non_blank.max(1)[0], ideal_attn_non_blank.to(attn.device))
        else:
            loss_ideal = None
        """
        loss_ideal = None

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

        return loss_att, acc_att, cer_att, wer_att, loss_pre, loss_ideal

    def sampler(
        self,
        encoder_out,
        encoder_out_lens,
        ys_pad,
        ys_pad_lens,
        pre_acoustic_embeds,
        contextual_info,
    ):
        tgt_mask = (~make_pad_mask(ys_pad_lens, maxlen=ys_pad_lens.max())[:, :, None]).to(
            ys_pad.device
        )
        ys_pad = ys_pad * tgt_mask[:, :, 0]
        if self.share_embedding:
            ys_pad_embed = self.decoder.output_layer.weight[ys_pad]
        else:
            ys_pad_embed = self.decoder.embed(ys_pad)
        with torch.no_grad():
            decoder_outs = self.decoder(
                encoder_out,
                encoder_out_lens,
                pre_acoustic_embeds,
                ys_pad_lens,
                contextual_info=contextual_info,
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
                        dim=0,
                        index=torch.randperm(seq_lens[li])[:target_num].to(
                            pre_acoustic_embeds.device
                        ),
                        value=0,
                    )
            input_mask = input_mask.eq(1)
            input_mask = input_mask.masked_fill(~nonpad_positions, False)
            input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)

        sematic_embeds = pre_acoustic_embeds.masked_fill(
            ~input_mask_expand_dim, 0
        ) + ys_pad_embed.masked_fill(input_mask_expand_dim, 0)
        return sematic_embeds * tgt_mask, decoder_out * tgt_mask

    def cal_decoder_with_predictor(
        self,
        encoder_out,
        encoder_out_lens,
        sematic_embeds,
        ys_pad_lens,
        hw_list=None,
        clas_scale=1.0,
    ):
        if hw_list is None:
            hw_list = [torch.Tensor([1]).long().to(encoder_out.device)]  # empty hotword list
            hw_list_pad = pad_list(hw_list, 0)
            if self.use_decoder_embedding:
                hw_embed = self.decoder.embed(hw_list_pad)
            else:
                hw_embed = self.bias_embed(hw_list_pad)
            hw_embed, (h_n, _) = self.bias_encoder(hw_embed)
            hw_embed = h_n.repeat(encoder_out.shape[0], 1, 1)
        else:
            hw_lengths = [len(i) for i in hw_list]
            hw_list_pad = pad_list([torch.Tensor(i).long() for i in hw_list], 0).to(
                encoder_out.device
            )
            if self.use_decoder_embedding:
                hw_embed = self.decoder.embed(hw_list_pad)
            else:
                hw_embed = self.bias_embed(hw_list_pad)
            hw_embed = torch.nn.utils.rnn.pack_padded_sequence(
                hw_embed, hw_lengths, batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.bias_encoder(hw_embed)
            hw_embed = h_n.repeat(encoder_out.shape[0], 1, 1)

        decoder_outs = self.decoder(
            encoder_out,
            encoder_out_lens,
            sematic_embeds,
            ys_pad_lens,
            contextual_info=hw_embed,
            clas_scale=clas_scale,
        )

        decoder_out = decoder_outs[0]
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        return decoder_out, ys_pad_lens

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
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

        meta_data = {}

        # extract fbank feats
        time1 = time.perf_counter()

        audio_sample_list = load_audio_text_image_video(
            data_in, fs=frontend.fs, audio_fs=kwargs.get("fs", 16000)
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

        # hotword
        self.hotword_list = self.generate_hotwords_list(
            kwargs.get("hotword", None), tokenizer=tokenizer, frontend=frontend
        )

        # Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # predictor
        predictor_outs = self.calc_predictor(encoder_out, encoder_out_lens)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = (
            predictor_outs[0],
            predictor_outs[1],
            predictor_outs[2],
            predictor_outs[3],
        )
        pre_token_length = pre_token_length.round().long()
        if torch.max(pre_token_length) < 1:
            return []

        decoder_outs = self.cal_decoder_with_predictor(
            encoder_out,
            encoder_out_lens,
            pre_acoustic_embeds,
            pre_token_length,
            hw_list=self.hotword_list,
            clas_scale=kwargs.get("clas_scale", 1.0),
        )
        decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]

        results = []
        b, n, d = decoder_out.size()
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
                ibest_writer = None
                if kwargs.get("output_dir") is not None:
                    if not hasattr(self, "writer"):
                        self.writer = DatadirWriter(kwargs.get("output_dir"))
                    ibest_writer = self.writer[f"{nbest_idx + 1}best_recog"]

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

                if tokenizer is not None:
                    # Change integer-ids to tokens
                    token = tokenizer.ids2tokens(token_int)
                    text = tokenizer.tokens2text(token)

                    text_postprocessed, _ = postprocess_utils.sentence_postprocess(token)
                    result_i = {"key": key[i], "text": text_postprocessed}

                    if ibest_writer is not None:
                        ibest_writer["token"][key[i]] = " ".join(token)
                        ibest_writer["text"][key[i]] = text
                        ibest_writer["text_postprocessed"][key[i]] = text_postprocessed
                else:
                    result_i = {"key": key[i], "token_int": token_int}
                results.append(result_i)

        return results, meta_data

    def generate_hotwords_list(self, hotword_list_or_file, tokenizer=None, frontend=None):
        def load_seg_dict(seg_dict_file):
            seg_dict = {}
            assert isinstance(seg_dict_file, str)
            with open(seg_dict_file, "r", encoding="utf8") as f:
                lines = f.readlines()
                for line in lines:
                    s = line.strip().split()
                    key = s[0]
                    value = s[1:]
                    seg_dict[key] = " ".join(value)
            return seg_dict

        def seg_tokenize(txt, seg_dict):
            pattern = re.compile(r"^[\u4E00-\u9FA50-9]+$")
            out_txt = ""
            for word in txt:
                word = word.lower()
                if word in seg_dict:
                    out_txt += seg_dict[word] + " "
                else:
                    if pattern.match(word):
                        for char in word:
                            if char in seg_dict:
                                out_txt += seg_dict[char] + " "
                            else:
                                out_txt += "<unk>" + " "
                    else:
                        out_txt += "<unk>" + " "
            return out_txt.strip().split()

        seg_dict = None
        if frontend.cmvn_file is not None:
            model_dir = os.path.dirname(frontend.cmvn_file)
            seg_dict_file = os.path.join(model_dir, "seg_dict")
            if os.path.exists(seg_dict_file):
                seg_dict = load_seg_dict(seg_dict_file)
            else:
                seg_dict = None
        # for None
        if hotword_list_or_file is None:
            hotword_list = None
        # for local txt inputs
        elif os.path.exists(hotword_list_or_file) and hotword_list_or_file.endswith(".txt"):
            logging.info("Attempting to parse hotwords from local txt...")
            hotword_list = []
            hotword_str_list = []
            with codecs.open(hotword_list_or_file, "r") as fin:
                for line in fin.readlines():
                    hw = line.strip()
                    hw_list = hw.split()
                    if seg_dict is not None:
                        hw_list = seg_tokenize(hw_list, seg_dict)
                    hotword_str_list.append(hw)
                    hotword_list.append(tokenizer.tokens2ids(hw_list))
                hotword_list.append([self.sos])
                hotword_str_list.append("<s>")
            logging.info(
                "Initialized hotword list from file: {}, hotword list: {}.".format(
                    hotword_list_or_file, hotword_str_list
                )
            )
        # for url, download and generate txt
        elif hotword_list_or_file.startswith("http"):
            logging.info("Attempting to parse hotwords from url...")
            work_dir = tempfile.TemporaryDirectory().name
            if not os.path.exists(work_dir):
                os.makedirs(work_dir)
            text_file_path = os.path.join(work_dir, os.path.basename(hotword_list_or_file))
            local_file = requests.get(hotword_list_or_file)
            open(text_file_path, "wb").write(local_file.content)
            hotword_list_or_file = text_file_path
            hotword_list = []
            hotword_str_list = []
            with codecs.open(hotword_list_or_file, "r") as fin:
                for line in fin.readlines():
                    hw = line.strip()
                    hw_list = hw.split()
                    if seg_dict is not None:
                        hw_list = seg_tokenize(hw_list, seg_dict)
                    hotword_str_list.append(hw)
                    hotword_list.append(tokenizer.tokens2ids(hw_list))
                hotword_list.append([self.sos])
                hotword_str_list.append("<s>")
            logging.info(
                "Initialized hotword list from file: {}, hotword list: {}.".format(
                    hotword_list_or_file, hotword_str_list
                )
            )
        # for text str input
        elif not hotword_list_or_file.endswith(".txt"):
            logging.info("Attempting to parse hotwords as str...")
            hotword_list = []
            hotword_str_list = []
            for hw in hotword_list_or_file.strip().split():
                hotword_str_list.append(hw)
                hw_list = hw.strip().split()
                if seg_dict is not None:
                    hw_list = seg_tokenize(hw_list, seg_dict)
                hotword_list.append(tokenizer.tokens2ids(hw_list))
            hotword_list.append([self.sos])
            hotword_str_list.append("<s>")
            logging.info("Hotword list: {}.".format(hotword_str_list))
        else:
            hotword_list = None
        return hotword_list

    def export(
        self,
        **kwargs,
    ):
        if "max_seq_len" not in kwargs:
            kwargs["max_seq_len"] = 512
        from .export_meta import export_rebuild_model

        models = export_rebuild_model(model=self, **kwargs)
        return models
