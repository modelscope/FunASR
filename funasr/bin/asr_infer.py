#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)


import codecs
import copy
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import requests
import torch
from packaging.version import parse as V
from typeguard import check_argument_types
from typeguard import check_return_type

from funasr.models.e2e_asr_contextual_paraformer import NeatContextualParaformer
from funasr.models.e2e_asr_paraformer import BiCifParaformer, ContextualParaformer
from funasr.models.frontend.wav_frontend import WavFrontend, WavFrontendOnline
from funasr.modules.beam_search.beam_search import BeamSearch
from funasr.modules.beam_search.beam_search import Hypothesis
from funasr.modules.beam_search.beam_search_sa_asr import Hypothesis as HypothesisSAASR
from funasr.modules.beam_search.beam_search_transducer import BeamSearchTransducer
from funasr.modules.beam_search.beam_search_transducer import Hypothesis as HypothesisTransducer
from funasr.modules.scorers.ctc import CTCPrefixScorer
from funasr.modules.scorers.length_bonus import LengthBonus
from funasr.tasks.asr import ASRTask
from funasr.tasks.asr import frontend_choices
from funasr.tasks.lm import LMTask
from funasr.text.build_tokenizer import build_tokenizer
from funasr.text.token_id_converter import TokenIDConverter
from funasr.torch_utils.device_funcs import to_device
from funasr.utils.timestamp_tools import ts_prediction_lfr6_standard


class Speech2Text:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("asr_config.yml", "asr.pb")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
            self,
            asr_train_config: Union[Path, str] = None,
            asr_model_file: Union[Path, str] = None,
            cmvn_file: Union[Path, str] = None,
            lm_train_config: Union[Path, str] = None,
            lm_file: Union[Path, str] = None,
            token_type: str = None,
            bpemodel: str = None,
            device: str = "cpu",
            maxlenratio: float = 0.0,
            minlenratio: float = 0.0,
            batch_size: int = 1,
            dtype: str = "float32",
            beam_size: int = 20,
            ctc_weight: float = 0.5,
            lm_weight: float = 1.0,
            ngram_weight: float = 0.9,
            penalty: float = 0.0,
            nbest: int = 1,
            streaming: bool = False,
            frontend_conf: dict = None,
            **kwargs,
    ):
        assert check_argument_types()

        # 1. Build ASR model
        scorers = {}
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, cmvn_file, device
        )
        frontend = None
        if asr_train_args.frontend is not None and asr_train_args.frontend_conf is not None:
            if asr_train_args.frontend == 'wav_frontend':
                frontend = WavFrontend(cmvn_file=cmvn_file, **asr_train_args.frontend_conf)
            else:
                from funasr.tasks.asr import frontend_choices
                frontend_class = frontend_choices.get_class(asr_train_args.frontend)
                frontend = frontend_class(**asr_train_args.frontend_conf).eval()

        logging.info("asr_model: {}".format(asr_model))
        logging.info("asr_train_args: {}".format(asr_train_args))
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        decoder = asr_model.decoder

        ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
        token_list = asr_model.token_list
        scorers.update(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, None, device
            )
            scorers["lm"] = lm.lm

        # 3. Build ngram model
        # ngram is not supported now
        ngram = None
        scorers["ngram"] = ngram

        # 4. Build BeamSearch object
        # transducer is not supported now
        beam_search_transducer = None
        from funasr.modules.beam_search.beam_search import BeamSearch

        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            ngram=ngram_weight,
            length_bonus=penalty,
        )
        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        )

        # 5. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.beam_search_transducer = beam_search_transducer
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.frontend = frontend

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray] = None
    ) -> List[
        Tuple[
            Optional[str],
            List[str],
            List[int],
            Union[Hypothesis],
        ]
    ]:
        """Inference

        Args:
            speech: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        if self.frontend is not None:
            feats, feats_len = self.frontend.forward(speech, speech_lengths)
            feats = to_device(feats, device=self.device)
            feats_len = feats_len.int()
            self.asr_model.frontend = None
        else:
            feats = speech
            feats_len = speech_lengths
        lfr_factor = max(1, (feats.size()[-1] // 80) - 1)
        batch = {"speech": feats, "speech_lengths": feats_len}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, _ = self.asr_model.encode(**batch)
        if isinstance(enc, tuple):
            enc = enc[0]
        assert len(enc) == 1, len(enc)

        # c. Passed the encoder result and the beam search
        nbest_hyps = self.beam_search(
            x=enc[0], maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
        )

        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, (Hypothesis)), type(hyp)

            # remove sos/eos and get results
            last_pos = -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[1:last_pos]
            else:
                token_int = hyp.yseq[1:last_pos].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        assert check_return_type(results)
        return results


class Speech2TextParaformer:
    """Speech2Text class

    Examples:
            >>> import soundfile
            >>> speech2text = Speech2TextParaformer("asr_config.yml", "asr.pb")
            >>> audio, rate = soundfile.read("speech.wav")
            >>> speech2text(audio)
            [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
            self,
            asr_train_config: Union[Path, str] = None,
            asr_model_file: Union[Path, str] = None,
            cmvn_file: Union[Path, str] = None,
            lm_train_config: Union[Path, str] = None,
            lm_file: Union[Path, str] = None,
            token_type: str = None,
            bpemodel: str = None,
            device: str = "cpu",
            maxlenratio: float = 0.0,
            minlenratio: float = 0.0,
            dtype: str = "float32",
            beam_size: int = 20,
            ctc_weight: float = 0.5,
            lm_weight: float = 1.0,
            ngram_weight: float = 0.9,
            penalty: float = 0.0,
            nbest: int = 1,
            frontend_conf: dict = None,
            hotword_list_or_file: str = None,
            decoding_ind: int = 0,
            **kwargs,
    ):
        assert check_argument_types()

        # 1. Build ASR model
        scorers = {}
        from funasr.tasks.asr import ASRTaskParaformer as ASRTask
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, cmvn_file, device
        )
        frontend = None
        if asr_train_args.frontend is not None and asr_train_args.frontend_conf is not None:
            frontend = WavFrontend(cmvn_file=cmvn_file, **asr_train_args.frontend_conf)

        logging.info("asr_model: {}".format(asr_model))
        logging.info("asr_train_args: {}".format(asr_train_args))
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        if asr_model.ctc != None:
            ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
            scorers.update(
                ctc=ctc
            )
        token_list = asr_model.token_list
        scorers.update(
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            scorers["lm"] = lm.lm

        # 3. Build ngram model
        # ngram is not supported now
        ngram = None
        scorers["ngram"] = ngram

        # 4. Build BeamSearch object
        # transducer is not supported now
        beam_search_transducer = None
        from funasr.modules.beam_search.beam_search import BeamSearchPara as BeamSearch

        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            ngram=ngram_weight,
            length_bonus=penalty,
        )
        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        )

        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()

        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 5. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer

        # 6. [Optional] Build hotword list from str, local file or url
        self.hotword_list = None
        self.hotword_list = self.generate_hotwords_list(hotword_list_or_file)

        is_use_lm = lm_weight != 0.0 and lm_file is not None
        if (ctc_weight == 0.0 or asr_model.ctc == None) and not is_use_lm:
            beam_search = None
        self.beam_search = beam_search
        logging.info(f"Beam_search: {self.beam_search}")
        self.beam_search_transducer = beam_search_transducer
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.frontend = frontend
        self.encoder_downsampling_factor = 1
        self.decoding_ind = decoding_ind
        if asr_train_args.encoder == "data2vec_encoder" or asr_train_args.encoder_conf["input_layer"] == "conv2d":
            self.encoder_downsampling_factor = 4

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray] = None,
            begin_time: int = 0, end_time: int = None,
    ):
        """Inference

        Args:
                speech: Input speech data
        Returns:
                text, token, token_int, hyp

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        if self.frontend is not None:
            feats, feats_len = self.frontend.forward(speech, speech_lengths)
            feats = to_device(feats, device=self.device)
            feats_len = feats_len.int()
            self.asr_model.frontend = None
        else:
            feats = speech
            feats_len = speech_lengths
        lfr_factor = max(1, (feats.size()[-1] // 80) - 1)
        batch = {"speech": feats, "speech_lengths": feats_len}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, enc_len = self.asr_model.encode(**batch, ind=self.decoding_ind)
        if isinstance(enc, tuple):
            enc = enc[0]
        # assert len(enc) == 1, len(enc)
        enc_len_batch_total = torch.sum(enc_len).item() * self.encoder_downsampling_factor

        predictor_outs = self.asr_model.calc_predictor(enc, enc_len)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = predictor_outs[0], predictor_outs[1], \
                                                                        predictor_outs[2], predictor_outs[3]
        pre_token_length = pre_token_length.round().long()
        if torch.max(pre_token_length) < 1:
            return []
        if not isinstance(self.asr_model, ContextualParaformer) and not isinstance(self.asr_model,
                                                                                   NeatContextualParaformer):
            if self.hotword_list:
                logging.warning("Hotword is given but asr model is not a ContextualParaformer.")
            decoder_outs = self.asr_model.cal_decoder_with_predictor(enc, enc_len, pre_acoustic_embeds,
                                                                     pre_token_length)
            decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]
        else:
            decoder_outs = self.asr_model.cal_decoder_with_predictor(enc, enc_len, pre_acoustic_embeds,
                                                                     pre_token_length, hw_list=self.hotword_list)
            decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]

        if isinstance(self.asr_model, BiCifParaformer):
            _, _, us_alphas, us_peaks = self.asr_model.calc_predictor_timestamp(enc, enc_len,
                                                                                pre_token_length)  # test no bias cif2

        results = []
        b, n, d = decoder_out.size()
        for i in range(b):
            x = enc[i, :enc_len[i], :]
            am_scores = decoder_out[i, :pre_token_length[i], :]
            if self.beam_search is not None:
                nbest_hyps = self.beam_search(
                    x=x, am_scores=am_scores, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
                )

                nbest_hyps = nbest_hyps[: self.nbest]
            else:
                if pre_token_length[i] == 0:
                    yseq = torch.tensor(
                        [self.asr_model.sos] + [self.asr_model.eos], device=pre_acoustic_embeds.device
                    )
                    score = torch.tensor(0.0, device=pre_acoustic_embeds.device)
                else:
                    yseq = am_scores.argmax(dim=-1)
                    score = am_scores.max(dim=-1)[0]
                    score = torch.sum(score, dim=-1)
                    # pad with mask tokens to ensure compatibility with sos/eos tokens
                    yseq = torch.tensor(
                        [self.asr_model.sos] + yseq.tolist() + [self.asr_model.eos], device=yseq.device
                    )
                nbest_hyps = [Hypothesis(yseq=yseq, score=score)]
            for hyp in nbest_hyps:
                assert isinstance(hyp, (Hypothesis)), type(hyp)

                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                # remove blank symbol id, which is assumed to be 0
                token_int = list(filter(lambda x: x != 0 and x != 2, token_int))

                # Change integer-ids to tokens
                token = self.converter.ids2tokens(token_int)

                if self.tokenizer is not None:
                    text = self.tokenizer.tokens2text(token)
                else:
                    text = None
                timestamp = []
                if isinstance(self.asr_model, BiCifParaformer):
                    _, timestamp = ts_prediction_lfr6_standard(us_alphas[i][:enc_len[i] * 3],
                                                               us_peaks[i][:enc_len[i] * 3],
                                                               copy.copy(token),
                                                               vad_offset=begin_time)
                results.append((text, token, token_int, hyp, timestamp, enc_len_batch_total, lfr_factor))

        # assert check_return_type(results)
        return results

    def generate_hotwords_list(self, hotword_list_or_file):
        # for None
        if hotword_list_or_file is None:
            hotword_list = None
        # for local txt inputs
        elif os.path.exists(hotword_list_or_file) and hotword_list_or_file.endswith('.txt'):
            logging.info("Attempting to parse hotwords from local txt...")
            hotword_list = []
            hotword_str_list = []
            with codecs.open(hotword_list_or_file, 'r') as fin:
                for line in fin.readlines():
                    hw = line.strip()
                    hotword_str_list.append(hw)
                    hotword_list.append(self.converter.tokens2ids([i for i in hw]))
                hotword_list.append([self.asr_model.sos])
                hotword_str_list.append('<s>')
            logging.info("Initialized hotword list from file: {}, hotword list: {}."
                         .format(hotword_list_or_file, hotword_str_list))
        # for url, download and generate txt
        elif hotword_list_or_file.startswith('http'):
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
            with codecs.open(hotword_list_or_file, 'r') as fin:
                for line in fin.readlines():
                    hw = line.strip()
                    hotword_str_list.append(hw)
                    hotword_list.append(self.converter.tokens2ids([i for i in hw]))
                hotword_list.append([self.asr_model.sos])
                hotword_str_list.append('<s>')
            logging.info("Initialized hotword list from file: {}, hotword list: {}."
                         .format(hotword_list_or_file, hotword_str_list))
        # for text str input
        elif not hotword_list_or_file.endswith('.txt'):
            logging.info("Attempting to parse hotwords as str...")
            hotword_list = []
            hotword_str_list = []
            for hw in hotword_list_or_file.strip().split():
                hotword_str_list.append(hw)
                hotword_list.append(self.converter.tokens2ids([i for i in hw]))
            hotword_list.append([self.asr_model.sos])
            hotword_str_list.append('<s>')
            logging.info("Hotword list: {}.".format(hotword_str_list))
        else:
            hotword_list = None
        return hotword_list


class Speech2TextParaformerOnline:
    """Speech2Text class

    Examples:
            >>> import soundfile
            >>> speech2text = Speech2TextParaformerOnline("asr_config.yml", "asr.pth")
            >>> audio, rate = soundfile.read("speech.wav")
            >>> speech2text(audio)
            [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
            self,
            asr_train_config: Union[Path, str] = None,
            asr_model_file: Union[Path, str] = None,
            cmvn_file: Union[Path, str] = None,
            lm_train_config: Union[Path, str] = None,
            lm_file: Union[Path, str] = None,
            token_type: str = None,
            bpemodel: str = None,
            device: str = "cpu",
            maxlenratio: float = 0.0,
            minlenratio: float = 0.0,
            dtype: str = "float32",
            beam_size: int = 20,
            ctc_weight: float = 0.5,
            lm_weight: float = 1.0,
            ngram_weight: float = 0.9,
            penalty: float = 0.0,
            nbest: int = 1,
            frontend_conf: dict = None,
            hotword_list_or_file: str = None,
            **kwargs,
    ):
        assert check_argument_types()

        # 1. Build ASR model
        scorers = {}
        from funasr.tasks.asr import ASRTaskParaformer as ASRTask
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, cmvn_file, device
        )
        frontend = None
        if asr_train_args.frontend is not None and asr_train_args.frontend_conf is not None:
            frontend = WavFrontendOnline(cmvn_file=cmvn_file, **asr_train_args.frontend_conf)

        logging.info("asr_model: {}".format(asr_model))
        logging.info("asr_train_args: {}".format(asr_train_args))
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        if asr_model.ctc != None:
            ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
            scorers.update(
                ctc=ctc
            )
        token_list = asr_model.token_list
        scorers.update(
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            scorers["lm"] = lm.lm

        # 3. Build ngram model
        # ngram is not supported now
        ngram = None
        scorers["ngram"] = ngram

        # 4. Build BeamSearch object
        # transducer is not supported now
        beam_search_transducer = None
        from funasr.modules.beam_search.beam_search import BeamSearchPara as BeamSearch

        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            ngram=ngram_weight,
            length_bonus=penalty,
        )
        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        )

        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()

        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 5. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer

        # 6. [Optional] Build hotword list from str, local file or url

        is_use_lm = lm_weight != 0.0 and lm_file is not None
        if (ctc_weight == 0.0 or asr_model.ctc == None) and not is_use_lm:
            beam_search = None
        self.beam_search = beam_search
        logging.info(f"Beam_search: {self.beam_search}")
        self.beam_search_transducer = beam_search_transducer
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.frontend = frontend
        self.encoder_downsampling_factor = 1
        if asr_train_args.encoder == "data2vec_encoder" or asr_train_args.encoder_conf["input_layer"] == "conv2d":
            self.encoder_downsampling_factor = 4

    @torch.no_grad()
    def __call__(
            self, cache: dict, speech: Union[torch.Tensor], speech_lengths: Union[torch.Tensor] = None
    ):
        """Inference

        Args:
                speech: Input speech data
        Returns:
                text, token, token_int, hyp

        """
        assert check_argument_types()
        results = []
        cache_en = cache["encoder"]
        if speech.shape[1] < 16 * 60 and cache_en["is_final"]:
            if cache_en["start_idx"] == 0:
                return []
            cache_en["tail_chunk"] = True
            feats = cache_en["feats"]
            feats_len = torch.tensor([feats.shape[1]])
            self.asr_model.frontend = None
            self.frontend.cache_reset()
            results = self.infer(feats, feats_len, cache)
            return results
        else:
            if self.frontend is not None:
                if cache_en["start_idx"] == 0:
                    self.frontend.cache_reset()
                feats, feats_len = self.frontend.forward(speech, speech_lengths, cache_en["is_final"])
                feats = to_device(feats, device=self.device)
                feats_len = feats_len.int()
                self.asr_model.frontend = None
            else:
                feats = speech
                feats_len = speech_lengths

            if feats.shape[1] != 0:
                results = self.infer(feats, feats_len, cache)

        return results

    @torch.no_grad()
    def infer(self, feats: Union[torch.Tensor], feats_len: Union[torch.Tensor], cache: List = None):
        batch = {"speech": feats, "speech_lengths": feats_len}
        batch = to_device(batch, device=self.device)
        # b. Forward Encoder
        enc, enc_len = self.asr_model.encode_chunk(feats, feats_len, cache=cache)
        if isinstance(enc, tuple):
            enc = enc[0]
        # assert len(enc) == 1, len(enc)
        enc_len_batch_total = torch.sum(enc_len).item() * self.encoder_downsampling_factor

        predictor_outs = self.asr_model.calc_predictor_chunk(enc, cache)
        pre_acoustic_embeds, pre_token_length = predictor_outs[0], predictor_outs[1]
        if torch.max(pre_token_length) < 1:
            return []
        decoder_outs = self.asr_model.cal_decoder_with_predictor_chunk(enc, pre_acoustic_embeds, cache)
        decoder_out = decoder_outs

        results = []
        b, n, d = decoder_out.size()
        for i in range(b):
            x = enc[i, :enc_len[i], :]
            am_scores = decoder_out[i, :pre_token_length[i], :]
            if self.beam_search is not None:
                nbest_hyps = self.beam_search(
                    x=x, am_scores=am_scores, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
                )

                nbest_hyps = nbest_hyps[: self.nbest]
            else:
                yseq = am_scores.argmax(dim=-1)
                score = am_scores.max(dim=-1)[0]
                score = torch.sum(score, dim=-1)
                # pad with mask tokens to ensure compatibility with sos/eos tokens
                yseq = torch.tensor(
                    [self.asr_model.sos] + yseq.tolist() + [self.asr_model.eos], device=yseq.device
                )
                nbest_hyps = [Hypothesis(yseq=yseq, score=score)]

            for hyp in nbest_hyps:
                assert isinstance(hyp, (Hypothesis)), type(hyp)

                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                # remove blank symbol id, which is assumed to be 0
                token_int = list(filter(lambda x: x != 0 and x != 2, token_int))

                # Change integer-ids to tokens
                token = self.converter.ids2tokens(token_int)
                postprocessed_result = ""
                for item in token:
                    if item.endswith('@@'):
                        postprocessed_result += item[:-2]
                    elif re.match('^[a-zA-Z]+$', item):
                        postprocessed_result += item + " "
                    else:
                        postprocessed_result += item

                results.append(postprocessed_result)

        # assert check_return_type(results)
        return results


class Speech2TextUniASR:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2TextUniASR("asr_config.yml", "asr.pb")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
            self,
            asr_train_config: Union[Path, str] = None,
            asr_model_file: Union[Path, str] = None,
            cmvn_file: Union[Path, str] = None,
            lm_train_config: Union[Path, str] = None,
            lm_file: Union[Path, str] = None,
            token_type: str = None,
            bpemodel: str = None,
            device: str = "cpu",
            maxlenratio: float = 0.0,
            minlenratio: float = 0.0,
            dtype: str = "float32",
            beam_size: int = 20,
            ctc_weight: float = 0.5,
            lm_weight: float = 1.0,
            ngram_weight: float = 0.9,
            penalty: float = 0.0,
            nbest: int = 1,
            token_num_relax: int = 1,
            decoding_ind: int = 0,
            decoding_mode: str = "model1",
            frontend_conf: dict = None,
            **kwargs,
    ):
        assert check_argument_types()

        # 1. Build ASR model
        scorers = {}
        from funasr.tasks.asr import ASRTaskUniASR as ASRTask
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, cmvn_file, device
        )
        frontend = None
        if asr_train_args.frontend is not None and asr_train_args.frontend_conf is not None:
            frontend = WavFrontend(cmvn_file=cmvn_file, **asr_train_args.frontend_conf)

        logging.info("asr_train_args: {}".format(asr_train_args))
        asr_model.to(dtype=getattr(torch, dtype)).eval()
        if decoding_mode == "model1":
            decoder = asr_model.decoder
        else:
            decoder = asr_model.decoder2

        if asr_model.ctc != None:
            ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
            scorers.update(
                ctc=ctc
            )
        token_list = asr_model.token_list
        scorers.update(
            decoder=decoder,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            scorers["lm"] = lm.lm

        # 3. Build ngram model
        # ngram is not supported now
        ngram = None
        scorers["ngram"] = ngram

        # 4. Build BeamSearch object
        # transducer is not supported now
        beam_search_transducer = None
        from funasr.modules.beam_search.beam_search import BeamSearchScama as BeamSearch

        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            ngram=ngram_weight,
            length_bonus=penalty,
        )
        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        )

        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
        # logging.info(f"Beam_search: {beam_search}")
        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 5. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.beam_search_transducer = beam_search_transducer
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.token_num_relax = token_num_relax
        self.decoding_ind = decoding_ind
        self.decoding_mode = decoding_mode
        self.frontend = frontend

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray] = None
    ) -> List[
        Tuple[
            Optional[str],
            List[str],
            List[int],
            Union[Hypothesis],
        ]
    ]:
        """Inference

        Args:
            speech: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        if self.frontend is not None:
            feats, feats_len = self.frontend.forward(speech, speech_lengths)
            feats = to_device(feats, device=self.device)
            feats_len = feats_len.int()
            self.asr_model.frontend = None
        else:
            feats = speech
            feats_len = speech_lengths
        lfr_factor = max(1, (feats.size()[-1] // 80) - 1)
        feats_raw = feats.clone().to(self.device)
        batch = {"speech": feats, "speech_lengths": feats_len}

        # a. To device
        batch = to_device(batch, device=self.device)
        # b. Forward Encoder
        _, enc, enc_len = self.asr_model.encode(**batch, ind=self.decoding_ind)
        if isinstance(enc, tuple):
            enc = enc[0]
        assert len(enc) == 1, len(enc)
        if self.decoding_mode == "model1":
            predictor_outs = self.asr_model.calc_predictor_mask(enc, enc_len)
        else:
            enc, enc_len = self.asr_model.encode2(enc, enc_len, feats_raw, feats_len, ind=self.decoding_ind)
            predictor_outs = self.asr_model.calc_predictor_mask2(enc, enc_len)

        scama_mask = predictor_outs[4]
        pre_token_length = predictor_outs[1]
        pre_acoustic_embeds = predictor_outs[0]
        maxlen = pre_token_length.sum().item() + self.token_num_relax
        minlen = max(0, pre_token_length.sum().item() - self.token_num_relax)
        # c. Passed the encoder result and the beam search
        nbest_hyps = self.beam_search(
            x=enc[0], scama_mask=scama_mask, pre_acoustic_embeds=pre_acoustic_embeds, maxlenratio=self.maxlenratio,
            minlenratio=self.minlenratio, maxlen=int(maxlen), minlen=int(minlen),
        )

        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, (Hypothesis)), type(hyp)

            # remove sos/eos and get results
            last_pos = -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[1:last_pos]
            else:
                token_int = hyp.yseq[1:last_pos].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)
            token = list(filter(lambda x: x != "<gbg>", token))

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        assert check_return_type(results)
        return results


class Speech2TextMFCCA:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2TextMFCCA("asr_config.yml", "asr.pb")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
            self,
            asr_train_config: Union[Path, str] = None,
            asr_model_file: Union[Path, str] = None,
            cmvn_file: Union[Path, str] = None,
            lm_train_config: Union[Path, str] = None,
            lm_file: Union[Path, str] = None,
            token_type: str = None,
            bpemodel: str = None,
            device: str = "cpu",
            maxlenratio: float = 0.0,
            minlenratio: float = 0.0,
            batch_size: int = 1,
            dtype: str = "float32",
            beam_size: int = 20,
            ctc_weight: float = 0.5,
            lm_weight: float = 1.0,
            ngram_weight: float = 0.9,
            penalty: float = 0.0,
            nbest: int = 1,
            streaming: bool = False,
            **kwargs,
    ):
        assert check_argument_types()

        # 1. Build ASR model
        from funasr.tasks.asr import ASRTaskMFCCA as ASRTask
        scorers = {}
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, cmvn_file, device
        )

        logging.info("asr_model: {}".format(asr_model))
        logging.info("asr_train_args: {}".format(asr_train_args))
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        decoder = asr_model.decoder

        ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
        token_list = asr_model.token_list
        scorers.update(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            lm.to(device)
            scorers["lm"] = lm.lm
        # 3. Build ngram model
        # ngram is not supported now
        ngram = None
        scorers["ngram"] = ngram

        # 4. Build BeamSearch object
        # transducer is not supported now
        beam_search_transducer = None

        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            ngram=ngram_weight,
            length_bonus=penalty,
        )
        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        )
        # beam_search.__class__ = BatchBeamSearch
        # 5. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.beam_search_transducer = beam_search_transducer
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray] = None
    ) -> List[
        Tuple[
            Optional[str],
            List[str],
            List[int],
            Union[Hypothesis],
        ]
    ]:
        """Inference

        Args:
            speech: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()
        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)
        if (speech.dim() == 3):
            speech = torch.squeeze(speech, 2)
        # speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        speech = speech.to(getattr(torch, self.dtype))
        # lenghts: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, _ = self.asr_model.encode(**batch)

        assert len(enc) == 1, len(enc)

        # c. Passed the encoder result and the beam search
        nbest_hyps = self.beam_search(
            x=enc[0], maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
        )

        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, (Hypothesis)), type(hyp)

            # remove sos/eos and get results
            last_pos = -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[1:last_pos]
            else:
                token_int = hyp.yseq[1:last_pos].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        assert check_return_type(results)
        return results


class Speech2TextTransducer:
    """Speech2Text class for Transducer models.
    Args:
        asr_train_config: ASR model training config path.
        asr_model_file: ASR model path.
        beam_search_config: Beam search config path.
        lm_train_config: Language Model training config path.
        lm_file: Language Model config path.
        token_type: Type of token units.
        bpemodel: BPE model path.
        device: Device to use for inference.
        beam_size: Size of beam during search.
        dtype: Data type.
        lm_weight: Language model weight.
        quantize_asr_model: Whether to apply dynamic quantization to ASR model.
        quantize_modules: List of module names to apply dynamic quantization on.
        quantize_dtype: Dynamic quantization data type.
        nbest: Number of final hypothesis.
        streaming: Whether to perform chunk-by-chunk inference.
        chunk_size: Number of frames in chunk AFTER subsampling.
        left_context: Number of frames in left context AFTER subsampling.
        right_context: Number of frames in right context AFTER subsampling.
        display_partial_hypotheses: Whether to display partial hypotheses.
    """

    def __init__(
            self,
            asr_train_config: Union[Path, str] = None,
            asr_model_file: Union[Path, str] = None,
            cmvn_file: Union[Path, str] = None,
            beam_search_config: Dict[str, Any] = None,
            lm_train_config: Union[Path, str] = None,
            lm_file: Union[Path, str] = None,
            token_type: str = None,
            bpemodel: str = None,
            device: str = "cpu",
            beam_size: int = 5,
            dtype: str = "float32",
            lm_weight: float = 1.0,
            quantize_asr_model: bool = False,
            quantize_modules: List[str] = None,
            quantize_dtype: str = "qint8",
            nbest: int = 1,
            streaming: bool = False,
            simu_streaming: bool = False,
            chunk_size: int = 16,
            left_context: int = 32,
            right_context: int = 0,
            display_partial_hypotheses: bool = False,
    ) -> None:
        """Construct a Speech2Text object."""
        super().__init__()

        assert check_argument_types()
        from funasr.tasks.asr import ASRTransducerTask
        asr_model, asr_train_args = ASRTransducerTask.build_model_from_file(
            asr_train_config, asr_model_file, cmvn_file, device
        )

        frontend = None
        if asr_train_args.frontend is not None and asr_train_args.frontend_conf is not None:
            frontend = WavFrontend(cmvn_file=cmvn_file, **asr_train_args.frontend_conf)

        if quantize_asr_model:
            if quantize_modules is not None:
                if not all([q in ["LSTM", "Linear"] for q in quantize_modules]):
                    raise ValueError(
                        "Only 'Linear' and 'LSTM' modules are currently supported"
                        " by PyTorch and in --quantize_modules"
                    )

                q_config = set([getattr(torch.nn, q) for q in quantize_modules])
            else:
                q_config = {torch.nn.Linear}

            if quantize_dtype == "float16" and (V(torch.__version__) < V("1.5.0")):
                raise ValueError(
                    "float16 dtype for dynamic quantization is not supported with torch"
                    " version < 1.5.0. Switching to qint8 dtype instead."
                )
            q_dtype = getattr(torch, quantize_dtype)

            asr_model = torch.quantization.quantize_dynamic(
                asr_model, q_config, dtype=q_dtype
            ).eval()
        else:
            asr_model.to(dtype=getattr(torch, dtype)).eval()

        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            lm_scorer = lm.lm
        else:
            lm_scorer = None

        # 4. Build BeamSearch object
        if beam_search_config is None:
            beam_search_config = {}

        beam_search = BeamSearchTransducer(
            asr_model.decoder,
            asr_model.joint_network,
            beam_size,
            lm=lm_scorer,
            lm_weight=lm_weight,
            nbest=nbest,
            **beam_search_config,
        )

        token_list = asr_model.token_list

        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

        self.converter = converter
        self.tokenizer = tokenizer

        self.beam_search = beam_search
        self.streaming = streaming
        self.simu_streaming = simu_streaming
        self.chunk_size = max(chunk_size, 0)
        self.left_context = left_context
        self.right_context = max(right_context, 0)

        if not streaming or chunk_size == 0:
            self.streaming = False
            self.asr_model.encoder.dynamic_chunk_training = False

        if not simu_streaming or chunk_size == 0:
            self.simu_streaming = False
            self.asr_model.encoder.dynamic_chunk_training = False

        self.frontend = frontend
        self.window_size = self.chunk_size + self.right_context

        if self.streaming:
            self._ctx = self.asr_model.encoder.get_encoder_input_size(
                self.window_size
            )

            self.last_chunk_length = (
                    self.asr_model.encoder.embed.min_frame_length + self.right_context + 1
            )
            self.reset_inference_cache()

    def reset_inference_cache(self) -> None:
        """Reset Speech2Text parameters."""
        self.frontend_cache = None

        self.asr_model.encoder.reset_streaming_cache(
            self.left_context, device=self.device
        )
        self.beam_search.reset_inference_cache()

        self.num_processed_frames = torch.tensor([[0]], device=self.device)

    @torch.no_grad()
    def streaming_decode(
            self,
            speech: Union[torch.Tensor, np.ndarray],
            is_final: bool = True,
    ) -> List[HypothesisTransducer]:
        """Speech2Text streaming call.
        Args:
            speech: Chunk of speech data. (S)
            is_final: Whether speech corresponds to the final chunk of data.
        Returns:
            nbest_hypothesis: N-best hypothesis.
        """
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)
        if is_final:
            if self.streaming and speech.size(0) < self.last_chunk_length:
                pad = torch.zeros(
                    self.last_chunk_length - speech.size(0), speech.size(1), dtype=speech.dtype
                )
                speech = torch.cat([speech, pad],
                                   dim=0)  # feats, feats_length = self.apply_frontend(speech, is_final=is_final)

        feats = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        feats_lengths = feats.new_full([1], dtype=torch.long, fill_value=feats.size(1))

        if self.asr_model.normalize is not None:
            feats, feats_lengths = self.asr_model.normalize(feats, feats_lengths)

        feats = to_device(feats, device=self.device)
        feats_lengths = to_device(feats_lengths, device=self.device)
        enc_out = self.asr_model.encoder.chunk_forward(
            feats,
            feats_lengths,
            self.num_processed_frames,
            chunk_size=self.chunk_size,
            left_context=self.left_context,
            right_context=self.right_context,
        )
        nbest_hyps = self.beam_search(enc_out[0], is_final=is_final)

        self.num_processed_frames += self.chunk_size

        if is_final:
            self.reset_inference_cache()

        return nbest_hyps

    @torch.no_grad()
    def simu_streaming_decode(self, speech: Union[torch.Tensor, np.ndarray]) -> List[HypothesisTransducer]:
        """Speech2Text call.
        Args:
            speech: Speech data. (S)
        Returns:
            nbest_hypothesis: N-best hypothesis.
        """
        assert check_argument_types()

        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        if self.frontend is not None:
            speech = torch.unsqueeze(speech, axis=0)
            speech_lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            feats = speech.unsqueeze(0).to(getattr(torch, self.dtype))
            feats_lengths = feats.new_full([1], dtype=torch.long, fill_value=feats.size(1))

        if self.asr_model.normalize is not None:
            feats, feats_lengths = self.asr_model.normalize(feats, feats_lengths)

        feats = to_device(feats, device=self.device)
        feats_lengths = to_device(feats_lengths, device=self.device)
        enc_out = self.asr_model.encoder.simu_chunk_forward(feats, feats_lengths, self.chunk_size, self.left_context,
                                                            self.right_context)
        nbest_hyps = self.beam_search(enc_out[0])

        return nbest_hyps

    @torch.no_grad()
    def __call__(self, speech: Union[torch.Tensor, np.ndarray]) -> List[HypothesisTransducer]:
        """Speech2Text call.
        Args:
            speech: Speech data. (S)
        Returns:
            nbest_hypothesis: N-best hypothesis.
        """
        assert check_argument_types()

        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        if self.frontend is not None:
            speech = torch.unsqueeze(speech, axis=0)
            speech_lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            feats = speech.unsqueeze(0).to(getattr(torch, self.dtype))
            feats_lengths = feats.new_full([1], dtype=torch.long, fill_value=feats.size(1))

        feats = to_device(feats, device=self.device)
        feats_lengths = to_device(feats_lengths, device=self.device)

        enc_out, _, _ = self.asr_model.encoder(feats, feats_lengths)

        nbest_hyps = self.beam_search(enc_out[0])

        return nbest_hyps

    def hypotheses_to_results(self, nbest_hyps: List[HypothesisTransducer]) -> List[Any]:
        """Build partial or final results from the hypotheses.
        Args:
            nbest_hyps: N-best hypothesis.
        Returns:
            results: Results containing different representation for the hypothesis.
        """
        results = []

        for hyp in nbest_hyps:
            token_int = list(filter(lambda x: x != 0, hyp.yseq))

            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

            assert check_return_type(results)

        return results

    @staticmethod
    def from_pretrained(
            model_tag: Optional[str] = None,
            **kwargs: Optional[Any],
    ) -> Speech2Text:
        """Build Speech2Text instance from the pretrained model.
        Args:
            model_tag: Model tag of the pretrained models.
        Return:
            : Speech2Text instance.
        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return Speech2TextTransducer(**kwargs)


class Speech2TextSAASR:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2TextSAASR("asr_config.yml", "asr.pb")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
            self,
            asr_train_config: Union[Path, str] = None,
            asr_model_file: Union[Path, str] = None,
            cmvn_file: Union[Path, str] = None,
            lm_train_config: Union[Path, str] = None,
            lm_file: Union[Path, str] = None,
            token_type: str = None,
            bpemodel: str = None,
            device: str = "cpu",
            maxlenratio: float = 0.0,
            minlenratio: float = 0.0,
            batch_size: int = 1,
            dtype: str = "float32",
            beam_size: int = 20,
            ctc_weight: float = 0.5,
            lm_weight: float = 1.0,
            ngram_weight: float = 0.9,
            penalty: float = 0.0,
            nbest: int = 1,
            streaming: bool = False,
            frontend_conf: dict = None,
            **kwargs,
    ):
        assert check_argument_types()

        # 1. Build ASR model
        from funasr.tasks.sa_asr import ASRTask
        scorers = {}
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, cmvn_file, device
        )
        frontend = None
        if asr_train_args.frontend is not None and asr_train_args.frontend_conf is not None:
            if asr_train_args.frontend == 'wav_frontend':
                frontend = WavFrontend(cmvn_file=cmvn_file, **asr_train_args.frontend_conf)
            else:
                frontend_class = frontend_choices.get_class(asr_train_args.frontend)
                frontend = frontend_class(**asr_train_args.frontend_conf).eval()

        logging.info("asr_model: {}".format(asr_model))
        logging.info("asr_train_args: {}".format(asr_train_args))
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        decoder = asr_model.decoder

        ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
        token_list = asr_model.token_list
        scorers.update(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, None, device
            )
            scorers["lm"] = lm.lm

        # 3. Build ngram model
        # ngram is not supported now
        ngram = None
        scorers["ngram"] = ngram

        # 4. Build BeamSearch object
        # transducer is not supported now
        beam_search_transducer = None
        from funasr.modules.beam_search.beam_search_sa_asr import BeamSearch

        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            ngram=ngram_weight,
            length_bonus=penalty,
        )
        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        )

        # 5. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.beam_search_transducer = beam_search_transducer
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.frontend = frontend

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray],
            profile: Union[torch.Tensor, np.ndarray], profile_lengths: Union[torch.Tensor, np.ndarray]
    ) -> List[
        Tuple[
            Optional[str],
            Optional[str],
            List[str],
            List[int],
            Union[HypothesisSAASR],
        ]
    ]:
        """Inference

        Args:
            speech: Input speech data
        Returns:
            text, text_id, token, token_int, hyp

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        if isinstance(profile, np.ndarray):
            profile = torch.tensor(profile)

        if self.frontend is not None:
            feats, feats_len = self.frontend.forward(speech, speech_lengths)
            feats = to_device(feats, device=self.device)
            feats_len = feats_len.int()
            self.asr_model.frontend = None
        else:
            feats = speech
            feats_len = speech_lengths
        lfr_factor = max(1, (feats.size()[-1] // 80) - 1)
        batch = {"speech": feats, "speech_lengths": feats_len}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        asr_enc, _, spk_enc = self.asr_model.encode(**batch)
        if isinstance(asr_enc, tuple):
            asr_enc = asr_enc[0]
        if isinstance(spk_enc, tuple):
            spk_enc = spk_enc[0]
        assert len(asr_enc) == 1, len(asr_enc)
        assert len(spk_enc) == 1, len(spk_enc)

        # c. Passed the encoder result and the beam search
        nbest_hyps = self.beam_search(
            asr_enc[0], spk_enc[0], profile[0], maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
        )

        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, (HypothesisSAASR)), type(hyp)

            # remove sos/eos and get results
            last_pos = -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[1: last_pos]
            else:
                token_int = hyp.yseq[1: last_pos].tolist()

            spk_weigths = torch.stack(hyp.spk_weigths, dim=0)

            token_ori = self.converter.ids2tokens(token_int)
            text_ori = self.tokenizer.tokens2text(token_ori)

            text_ori_spklist = text_ori.split('$')
            cur_index = 0
            spk_choose = []
            for i in range(len(text_ori_spklist)):
                text_ori_split = text_ori_spklist[i]
                n = len(text_ori_split)
                spk_weights_local = spk_weigths[cur_index: cur_index + n]
                cur_index = cur_index + n + 1
                spk_weights_local = spk_weights_local.mean(dim=0)
                spk_choose_local = spk_weights_local.argmax(-1)
                spk_choose.append(spk_choose_local.item() + 1)

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None

            text_spklist = text.split('$')
            assert len(spk_choose) == len(text_spklist)

            spk_list = []
            for i in range(len(text_spklist)):
                text_split = text_spklist[i]
                n = len(text_split)
                spk_list.append(str(spk_choose[i]) * n)

            text_id = '$'.join(spk_list)

            assert len(text) == len(text_id)

            results.append((text, text_id, token, token_int, hyp))

        assert check_return_type(results)
        return results
