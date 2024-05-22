#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os
import re
import time
import copy
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
from funasr.models.paraformer.model import Paraformer
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.paraformer.search import Hypothesis
from funasr.train_utils.device_funcs import force_gatherable
from funasr.models.bicif_paraformer.model import BiCifParaformer
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.utils.timestamp_tools import ts_prediction_lfr6_standard
from funasr.models.transformer.utils.nets_utils import make_pad_mask, pad_list
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank


if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


@tables.register("model_classes", "SeacoParaformer")
class SeacoParaformer(BiCifParaformer, Paraformer):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    SeACo-Paraformer: A Non-Autoregressive ASR System with Flexible and Effective Hotword Customization Ability
    https://arxiv.org/abs/2308.03266
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.inner_dim = kwargs.get("inner_dim", 256)
        self.bias_encoder_type = kwargs.get("bias_encoder_type", "lstm")
        bias_encoder_dropout_rate = kwargs.get("bias_encoder_dropout_rate", 0.0)
        bias_encoder_bid = kwargs.get("bias_encoder_bid", False)
        seaco_lsm_weight = kwargs.get("seaco_lsm_weight", 0.0)
        seaco_length_normalized_loss = kwargs.get("seaco_length_normalized_loss", True)

        # bias encoder
        if self.bias_encoder_type == "lstm":
            self.bias_encoder = torch.nn.LSTM(
                self.inner_dim,
                self.inner_dim,
                2,
                batch_first=True,
                dropout=bias_encoder_dropout_rate,
                bidirectional=bias_encoder_bid,
            )
            if bias_encoder_bid:
                self.lstm_proj = torch.nn.Linear(self.inner_dim * 2, self.inner_dim)
            else:
                self.lstm_proj = None
            # self.bias_embed = torch.nn.Embedding(self.vocab_size, self.inner_dim)
        elif self.bias_encoder_type == "mean":
            self.bias_embed = torch.nn.Embedding(self.vocab_size, self.inner_dim)
        else:
            logging.error("Unsupport bias encoder type: {}".format(self.bias_encoder_type))

        # seaco decoder
        seaco_decoder = kwargs.get("seaco_decoder", None)
        if seaco_decoder is not None:
            seaco_decoder_conf = kwargs.get("seaco_decoder_conf")
            seaco_decoder_class = tables.decoder_classes.get(seaco_decoder)
            self.seaco_decoder = seaco_decoder_class(
                vocab_size=self.vocab_size,
                encoder_output_size=self.inner_dim,
                **seaco_decoder_conf,
            )
        self.hotword_output_layer = torch.nn.Linear(self.inner_dim, self.vocab_size)
        self.criterion_seaco = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=self.ignore_id,
            smoothing=seaco_lsm_weight,
            normalize_length=seaco_length_normalized_loss,
        )
        self.train_decoder = kwargs.get("train_decoder", True)
        self.seaco_weight = kwargs.get("seaco_weight", 0.01)
        self.NO_BIAS = kwargs.get("NO_BIAS", 8377)
        self.predictor_name = kwargs.get("predictor")

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
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]
        # Check that batch_size is unified
        assert (
            speech.shape[0] == speech_lengths.shape[0] == text.shape[0] == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        hotword_pad = kwargs.get("hotword_pad")
        hotword_lengths = kwargs.get("hotword_lengths")
        seaco_label_pad = kwargs.get("seaco_label_pad")
        if len(hotword_lengths.size()) > 1:
            hotword_lengths = hotword_lengths[:, 0]

        batch_size = speech.shape[0]
        # for data-parallel
        text = text[:, : text_lengths.max()]
        speech = speech[:, : speech_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if self.predictor_bias == 1:
            _, ys_pad = add_sos_eos(text, self.sos, self.eos, self.ignore_id)
            ys_lengths = text_lengths + self.predictor_bias

        stats = dict()
        loss_seaco = self._calc_seaco_loss(
            encoder_out,
            encoder_out_lens,
            ys_pad,
            ys_lengths,
            hotword_pad,
            hotword_lengths,
            seaco_label_pad,
        )
        if self.train_decoder:
            loss_att, acc_att, _, _, _ = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
            loss = loss_seaco + loss_att * self.seaco_weight
            stats["loss_att"] = torch.clone(loss_att.detach())
            stats["acc_att"] = acc_att
        else:
            loss = loss_seaco

        stats["loss_seaco"] = torch.clone(loss_seaco.detach())
        stats["loss"] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = (text_lengths + self.predictor_bias).sum()
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _merge(self, cif_attended, dec_attended):
        return cif_attended + dec_attended

    def calc_predictor(self, encoder_out, encoder_out_lens):
        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        predictor_outs = self.predictor(
            encoder_out, None, encoder_out_mask, ignore_id=self.ignore_id
        )
        return predictor_outs[:4]

    def _calc_seaco_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_lengths: torch.Tensor,
        hotword_pad: torch.Tensor,
        hotword_lengths: torch.Tensor,
        seaco_label_pad: torch.Tensor,
    ):
        # predictor forward
        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        pre_acoustic_embeds = self.predictor(
            encoder_out, ys_pad, encoder_out_mask, ignore_id=self.ignore_id
        )[0]
        # decoder forward
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, pre_acoustic_embeds, ys_lengths, return_hidden=True
        )
        selected = self._hotword_representation(hotword_pad, hotword_lengths)
        contextual_info = (
            selected.squeeze(0).repeat(encoder_out.shape[0], 1, 1).to(encoder_out.device)
        )
        num_hot_word = contextual_info.shape[1]
        _contextual_length = (
            torch.Tensor([num_hot_word]).int().repeat(encoder_out.shape[0]).to(encoder_out.device)
        )
        # dha core
        cif_attended, _ = self.seaco_decoder(
            contextual_info, _contextual_length, pre_acoustic_embeds, ys_lengths
        )
        dec_attended, _ = self.seaco_decoder(
            contextual_info, _contextual_length, decoder_out, ys_lengths
        )
        merged = self._merge(cif_attended, dec_attended)
        dha_output = self.hotword_output_layer(
            merged[:, :-1]
        )  # remove the last token in loss calculation
        loss_att = self.criterion_seaco(dha_output, seaco_label_pad)
        return loss_att

    def _seaco_decode_with_ASF(
        self,
        encoder_out,
        encoder_out_lens,
        sematic_embeds,
        ys_pad_lens,
        hw_list,
        nfilter=50,
        seaco_weight=1.0,
    ):
        # decoder forward

        decoder_out, decoder_hidden, _ = self.decoder(
            encoder_out,
            encoder_out_lens,
            sematic_embeds,
            ys_pad_lens,
            return_hidden=True,
            return_both=True,
        )

        decoder_pred = torch.log_softmax(decoder_out, dim=-1)
        if hw_list is not None:
            hw_lengths = [len(i) for i in hw_list]
            hw_list_ = [torch.Tensor(i).long() for i in hw_list]
            hw_list_pad = pad_list(hw_list_, 0).to(encoder_out.device)
            selected = self._hotword_representation(
                hw_list_pad, torch.Tensor(hw_lengths).int().to(encoder_out.device)
            )

            contextual_info = (
                selected.squeeze(0).repeat(encoder_out.shape[0], 1, 1).to(encoder_out.device)
            )
            num_hot_word = contextual_info.shape[1]
            _contextual_length = (
                torch.Tensor([num_hot_word])
                .int()
                .repeat(encoder_out.shape[0])
                .to(encoder_out.device)
            )

            # ASF Core
            if nfilter > 0 and nfilter < num_hot_word:
                hotword_scores = self.seaco_decoder.forward_asf6(
                    contextual_info, _contextual_length, decoder_hidden, ys_pad_lens
                )
                hotword_scores = hotword_scores[0].sum(0).sum(0)
                # hotword_scores /= torch.sqrt(torch.tensor(hw_lengths)[:-1].float()).to(hotword_scores.device)
                dec_filter = torch.topk(hotword_scores, min(nfilter, num_hot_word - 1))[1].tolist()
                add_filter = dec_filter
                add_filter.append(len(hw_list_pad) - 1)
                # filter hotword embedding
                selected = selected[add_filter]
                # again
                contextual_info = (
                    selected.squeeze(0).repeat(encoder_out.shape[0], 1, 1).to(encoder_out.device)
                )
                num_hot_word = contextual_info.shape[1]
                _contextual_length = (
                    torch.Tensor([num_hot_word])
                    .int()
                    .repeat(encoder_out.shape[0])
                    .to(encoder_out.device)
                )

            # SeACo Core
            cif_attended, _ = self.seaco_decoder(
                contextual_info, _contextual_length, sematic_embeds, ys_pad_lens
            )
            dec_attended, _ = self.seaco_decoder(
                contextual_info, _contextual_length, decoder_hidden, ys_pad_lens
            )
            merged = self._merge(cif_attended, dec_attended)

            dha_output = self.hotword_output_layer(
                merged
            )  # remove the last token in loss calculation
            dha_pred = torch.log_softmax(dha_output, dim=-1)

            def _merge_res(dec_output, dha_output):
                lmbd = torch.Tensor([seaco_weight] * dha_output.shape[0])
                dha_ids = dha_output.max(-1)[-1]  # [0]
                dha_mask = (dha_ids == self.NO_BIAS).int().unsqueeze(-1)
                a = (1 - lmbd) / lmbd
                b = 1 / lmbd
                a, b = a.to(dec_output.device), b.to(dec_output.device)
                dha_mask = (dha_mask + a.reshape(-1, 1, 1)) / b.reshape(-1, 1, 1)
                # logits = dec_output * dha_mask + dha_output[:,:,:-1] * (1-dha_mask)
                logits = dec_output * dha_mask + dha_output[:, :, :] * (1 - dha_mask)
                return logits

            merged_pred = _merge_res(decoder_pred, dha_pred)
            return merged_pred
        else:
            return decoder_pred

    def _hotword_representation(self, hotword_pad, hotword_lengths):
        if self.bias_encoder_type != "lstm":
            logging.error("Unsupported bias encoder type")

        """
        hw_embed = self.decoder.embed(hotword_pad)
        hw_embed, (_, _) = self.bias_encoder(hw_embed)
        if self.lstm_proj is not None:
            hw_embed = self.lstm_proj(hw_embed)
        _ind = np.arange(0, hw_embed.shape[0]).tolist()
        selected = hw_embed[_ind, [i-1 for i in hotword_lengths.detach().cpu().tolist()]]
        return selected
        """

        # hw_embed = self.sac_embedding(hotword_pad)
        hw_embed = self.decoder.embed(hotword_pad)
        hw_embed = torch.nn.utils.rnn.pack_padded_sequence(
            hw_embed,
            hotword_lengths.cpu().type(torch.int64),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_rnn_output, _ = self.bias_encoder(hw_embed)
        rnn_output = torch.nn.utils.rnn.pad_packed_sequence(packed_rnn_output, batch_first=True)[0]
        if self.lstm_proj is not None:
            hw_hidden = self.lstm_proj(rnn_output)
        else:
            hw_hidden = rnn_output
        _ind = np.arange(0, hw_hidden.shape[0]).tolist()
        selected = hw_hidden[_ind, [i - 1 for i in hotword_lengths.detach().cpu().tolist()]]
        return selected

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
        pre_acoustic_embeds, pre_token_length = predictor_outs[0], predictor_outs[1]
        pre_token_length = pre_token_length.round().long()
        if torch.max(pre_token_length) < 1:
            return ([],)

        decoder_out = self._seaco_decode_with_ASF(
            encoder_out,
            encoder_out_lens,
            pre_acoustic_embeds,
            pre_token_length,
            hw_list=self.hotword_list,
        )

        # decoder_out, _ = decoder_outs[0], decoder_outs[1]
        if self.predictor_name == "CifPredictorV3":
            _, _, us_alphas, us_peaks = self.calc_predictor_timestamp(
                encoder_out, encoder_out_lens, pre_token_length
            )
        else:
            us_alphas = None

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
                    if us_alphas is not None:
                        _, timestamp = ts_prediction_lfr6_standard(
                            us_alphas[i][: encoder_out_lens[i] * 3],
                            us_peaks[i][: encoder_out_lens[i] * 3],
                            copy.copy(token),
                            vad_offset=kwargs.get("begin_time", 0),
                        )
                        text_postprocessed, time_stamp_postprocessed, _ = (
                            postprocess_utils.sentence_postprocess(token, timestamp)
                        )
                        result_i = {
                            "key": key[i],
                            "text": text_postprocessed,
                            "timestamp": time_stamp_postprocessed,
                        }
                        if ibest_writer is not None:
                            ibest_writer["token"][key[i]] = " ".join(token)
                            ibest_writer["timestamp"][key[i]] = time_stamp_postprocessed
                            ibest_writer["text"][key[i]] = text_postprocessed
                    else:
                        text_postprocessed, _ = postprocess_utils.sentence_postprocess(token)
                        result_i = {"key": key[i], "text": text_postprocessed}
                        if ibest_writer is not None:
                            ibest_writer["token"][key[i]] = " ".join(token)
                            ibest_writer["text"][key[i]] = text_postprocessed
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
