#!/usr/bin/env python3
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import Dict
from typing import Any
from typing import List
import math
import numpy as np
import torch
from typeguard import check_argument_types

from funasr.fileio.datadir_writer import DatadirWriter
from funasr.modules.beam_search.beam_search import BeamSearchPara as BeamSearch
from funasr.modules.beam_search.beam_search import Hypothesis
from funasr.modules.scorers.ctc import CTCPrefixScorer
from funasr.modules.scorers.length_bonus import LengthBonus
from funasr.modules.subsampling import TooShortUttError
from funasr.tasks.asr import ASRTaskParaformer as ASRTask
from funasr.tasks.lm import LMTask
from funasr.text.build_tokenizer import build_tokenizer
from funasr.text.token_id_converter import TokenIDConverter
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import config_argparse
from funasr.utils.cli_utils import get_commandline_args
from funasr.utils.types import str2bool
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_none
from funasr.utils import asr_utils, wav_utils, postprocess_utils
from funasr.models.frontend.wav_frontend import WavFrontend
from funasr.tasks.vad import VADTask
from funasr.utils.timestamp_tools import time_stamp_lfr6
from funasr.tasks.punctuation import PunctuationTask
from funasr.torch_utils.forward_adaptor import ForwardAdaptor
from funasr.datasets.preprocessor import CommonPreprocessor
from funasr.punctuation.text_preprocessor import split_words, split_to_mini_sentence

header_colors = '\033[95m'
end_colors = '\033[0m'

global_asr_language: str = 'zh-cn'
global_sample_rate: Union[int, Dict[Any, int]] = {
    'audio_fs': 16000,
    'model_fs': 16000
}

class Speech2Text:
    """Speech2Text class

    Examples:
            >>> import soundfile
            >>> speech2text = Speech2Text("asr_config.yml", "asr.pth")
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
            **kwargs,
    ):
        assert check_argument_types()

        # 1. Build ASR model
        scorers = {}
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, cmvn_file=cmvn_file, device=device
        )
        frontend = None
        if asr_model.frontend is not None and asr_train_args.frontend_conf is not None:
            frontend = WavFrontend(cmvn_file=cmvn_file, **asr_train_args.frontend_conf)

        # logging.info("asr_model: {}".format(asr_model))
        # logging.info("asr_train_args: {}".format(asr_train_args))
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
        token_list = asr_model.token_list
        scorers.update(
            ctc=ctc,
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
        is_use_lm = lm_weight != 0.0 and lm_file is not None
        if ctc_weight == 0.0 and not is_use_lm:
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
        if asr_train_args.encoder_conf["input_layer"] == "conv2d":
            self.encoder_downsampling_factor = 4
        
            

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray] = None, begin_time: int = 0, end_time: int = None, 
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
            # feats, feats_len = self.frontend.forward(speech, speech_lengths)
            # fbanks, fbanks_len = self.frontend.forward_fbank(speech, speech_lengths)
            feats, feats_len = self.frontend.forward_lfr_cmvn(speech, speech_lengths)
            feats = to_device(feats, device=self.device)
            feats_len = feats_len.int()
            self.asr_model.frontend = None
        else:
            feats = speech
            feats_len = speech_lengths
        lfr_factor = max(1, (feats.size()[-1]//80)-1)
        batch = {"speech": feats, "speech_lengths": feats_len}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, enc_len = self.asr_model.encode(**batch)
        if isinstance(enc, tuple):
            enc = enc[0]
        # assert len(enc) == 1, len(enc)
        enc_len_batch_total = torch.sum(enc_len).item() * self.encoder_downsampling_factor

        predictor_outs = self.asr_model.calc_predictor(enc, enc_len)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = predictor_outs[0], predictor_outs[1], predictor_outs[2], predictor_outs[3]
        pre_token_length = pre_token_length.round().long()
        decoder_outs = self.asr_model.cal_decoder_with_predictor(enc, enc_len, pre_acoustic_embeds, pre_token_length)
        decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]

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
    
                if self.tokenizer is not None:
                    text = self.tokenizer.tokens2text(token)
                else:
                    text = None

                time_stamp = time_stamp_lfr6(alphas[i:i+1,], enc_len[i:i+1,], token, begin_time, end_time)
    
                results.append((text, token, token_int, time_stamp, enc_len_batch_total, lfr_factor))

        # assert check_return_type(results)
        return results

class Speech2VadSegment:
    """Speech2VadSegment class

    Examples:
        >>> import soundfile
        >>> speech2segment = Speech2VadSegment("vad_config.yml", "vad.pt")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2segment(audio)
        [[10, 230], [245, 450], ...]

    """

    def __init__(
            self,
            vad_infer_config: Union[Path, str] = None,
            vad_model_file: Union[Path, str] = None,
            vad_cmvn_file: Union[Path, str] = None,
            device: str = "cpu",
            batch_size: int = 1,
            dtype: str = "float32",
            **kwargs,
    ):
        assert check_argument_types()

        # 1. Build vad model
        vad_model, vad_infer_args = VADTask.build_model_from_file(
            vad_infer_config, vad_model_file, device
        )
        frontend = None
        if vad_infer_args.frontend is not None:
            frontend = WavFrontend(cmvn_file=vad_cmvn_file, **vad_infer_args.frontend_conf)

        # logging.info("vad_model: {}".format(vad_model))
        # logging.info("vad_infer_args: {}".format(vad_infer_args))
        vad_model.to(dtype=getattr(torch, dtype)).eval()

        self.vad_model = vad_model
        self.vad_infer_args = vad_infer_args
        self.device = device
        self.dtype = dtype
        self.frontend = frontend

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray] = None
    ) -> List[List[int]]:
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
            self.frontend.filter_length_max = math.inf
            fbanks, fbanks_len = self.frontend.forward_fbank(speech, speech_lengths)
            feats, feats_len = self.frontend.forward_lfr_cmvn(fbanks, fbanks_len)
            fbanks = to_device(fbanks, device=self.device)
            feats = to_device(feats, device=self.device)
            feats_len = feats_len.int()
        else:
            raise Exception("Need to extract feats first, please configure frontend configuration")
        batch = {"feats": feats, "feats_lengths": feats_len, "waveform": speech}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        segments = self.vad_model(**batch)

        return fbanks, segments



def inference(
    maxlenratio: float,
    minlenratio: float,
    batch_size: int,
    beam_size: int,
    ngpu: int,
    ctc_weight: float,
    lm_weight: float,
    penalty: float,
    log_level: Union[int, str],
    data_path_and_name_and_type,
    asr_train_config: Optional[str],
    asr_model_file: Optional[str],
    cmvn_file: Optional[str] = None,
    raw_inputs: Union[np.ndarray, torch.Tensor] = None,
    lm_train_config: Optional[str] = None,
    lm_file: Optional[str] = None,
    token_type: Optional[str] = None,
    key_file: Optional[str] = None,
    word_lm_train_config: Optional[str] = None,
    bpemodel: Optional[str] = None,
    allow_variable_data_keys: bool = False,
    streaming: bool = False,
    output_dir: Optional[str] = None,
    dtype: str = "float32",
    seed: int = 0,
    ngram_weight: float = 0.9,
    nbest: int = 1,
    num_workers: int = 1,
    vad_infer_config: Optional[str] = None,
    vad_model_file: Optional[str] = None,
    vad_cmvn_file: Optional[str] = None,
    time_stamp_writer: bool = False,
    punc_infer_config: Optional[str] = None,
    punc_model_file: Optional[str] = None,
    **kwargs,
):

    inference_pipeline = inference_modelscope(
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        batch_size=batch_size,
        beam_size=beam_size,
        ngpu=ngpu,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        penalty=penalty,
        log_level=log_level,
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        cmvn_file=cmvn_file,
        raw_inputs=raw_inputs,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        key_file=key_file,
        word_lm_train_config=word_lm_train_config,
        bpemodel=bpemodel,
        allow_variable_data_keys=allow_variable_data_keys,
        streaming=streaming,
        output_dir=output_dir,
        dtype=dtype,
        seed=seed,
        ngram_weight=ngram_weight,
        nbest=nbest,
        num_workers=num_workers,
        vad_infer_config=vad_infer_config,
        vad_model_file=vad_model_file,
        vad_cmvn_file=vad_cmvn_file,
        time_stamp_writer=time_stamp_writer,
        punc_infer_config=punc_infer_config,
        punc_model_file=punc_model_file,
        **kwargs,
    )
    return inference_pipeline(data_path_and_name_and_type, raw_inputs)

def inference_modelscope(
    maxlenratio: float,
    minlenratio: float,
    batch_size: int,
    beam_size: int,
    ngpu: int,
    ctc_weight: float,
    lm_weight: float,
    penalty: float,
    log_level: Union[int, str],
    # data_path_and_name_and_type,
    asr_train_config: Optional[str],
    asr_model_file: Optional[str],
    cmvn_file: Optional[str] = None,
    lm_train_config: Optional[str] = None,
    lm_file: Optional[str] = None,
    token_type: Optional[str] = None,
    key_file: Optional[str] = None,
    word_lm_train_config: Optional[str] = None,
    bpemodel: Optional[str] = None,
    allow_variable_data_keys: bool = False,
    output_dir: Optional[str] = None,
    dtype: str = "float32",
    seed: int = 0,
    ngram_weight: float = 0.9,
    nbest: int = 1,
    num_workers: int = 1,
    vad_infer_config: Optional[str] = None,
    vad_model_file: Optional[str] = None,
    vad_cmvn_file: Optional[str] = None,
    time_stamp_writer: bool = True,
    punc_infer_config: Optional[str] = None,
    punc_model_file: Optional[str] = None,
    outputs_dict: Optional[bool] = True,
    **kwargs,
):
    assert check_argument_types()
    
    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    
    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # 1. Set random-seed
    set_all_random_seed(seed)
    
    # 2. Build speech2vadsegment
    speech2vadsegment_kwargs = dict(
        vad_infer_config=vad_infer_config,
        vad_model_file=vad_model_file,
        vad_cmvn_file=vad_cmvn_file,
        device=device,
        dtype=dtype,
    )
    # logging.info("speech2vadsegment_kwargs: {}".format(speech2vadsegment_kwargs))
    speech2vadsegment = Speech2VadSegment(**speech2vadsegment_kwargs)
    
    # 3. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        cmvn_file=cmvn_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
    )
    speech2text = Speech2Text(**speech2text_kwargs)
    
    text2punc = Text2Punc(punc_infer_config, punc_model_file, device=device, dtype=dtype)

    if output_dir is not None:
        writer = DatadirWriter(output_dir)
        ibest_writer = writer[f"1best_recog"]
        ibest_writer["token_list"][""] = " ".join(speech2text.asr_train_args.token_list)
    
    def _forward(data_path_and_name_and_type,
                 raw_inputs: Union[np.ndarray, torch.Tensor] = None,
                 output_dir_v2: Optional[str] = None,
                 ):
        # 3. Build data-iterator
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]
        loader = ASRTask.build_streaming_iterator(
            data_path_and_name_and_type,
            dtype=dtype,
            batch_size=1,
            key_file=key_file,
            num_workers=num_workers,
            preprocess_fn=VADTask.build_preprocess_fn(speech2vadsegment.vad_infer_args, False),
            collate_fn=VADTask.build_collate_fn(speech2vadsegment.vad_infer_args, False),
            allow_variable_data_keys=allow_variable_data_keys,
            inference=True,
        )
        
        forward_time_total = 0.0
        length_total = 0.0
        finish_count = 0
        file_count = 1
        lfr_factor = 6
        # 7 .Start for-loop
        asr_result_list = []
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path is not None:
            writer = DatadirWriter(output_path)
            ibest_writer = writer[f"1best_recog"]
            # ibest_writer["punc_dict"][""] = " ".join(punc_infer_config.punc_list)
            # ibest_writer["token_list"][""] = " ".join(asr_train_config.token_list)
        else:
            writer = None
        
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            # batch = {k: v for k, v in batch.items() if not k.endswith("_lengths")}
            
            logging.info("decoding, utt_id: {}".format(keys))
            # N-best list of (text, token, token_int, hyp_object)
            time_beg = time.time()
            vad_results = speech2vadsegment(**batch)
            time_end = time.time()
            fbanks, vadsegments = vad_results[0], vad_results[1]
            for i, segments in enumerate(vadsegments):
                result_segments = [["", [], [], ]]
                for j, segment_idx in enumerate(segments):
                    bed_idx, end_idx = int(segment_idx[0] / 10), int(segment_idx[1] / 10)
                    segment = fbanks[:, bed_idx:end_idx, :].to(device)
                    speech_lengths = torch.Tensor([end_idx - bed_idx]).int().to(device)
                    batch = {"speech": segment, "speech_lengths": speech_lengths, "begin_time": vadsegments[i][j][0],
                             "end_time": vadsegments[i][j][1]}
                    results = speech2text(**batch)
                    if len(results) < 1:
                        hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                        results = [[" ", ["<space>"], [2], 0, 1, 6]] * nbest
                    time_end = time.time()
                    forward_time = time_end - time_beg
                    lfr_factor = results[0][-1]
                    length = results[0][-2]
                    forward_time_total += forward_time
                    length_total += length
                    logging.info(
                        "decoding, feature length: {}, forward_time: {:.4f}, rtf: {:.4f}".
                        format(length, forward_time, 100 * forward_time / (length * lfr_factor)))
                    result_cur = [results[0][:-2]]
                    if j == 0:
                        result_segments = result_cur
                    else:
                        result_segments = [[result_segments[0][i] + result_cur[0][i] for i in range(len(result_cur[0]))]]
                
                key = keys[0]
                result = result_segments[0]
                text, token, token_int = result[0], result[1], result[2]
                time_stamp = None if len(result) < 4 else result[3]
                
                # Create a directory: outdir/{n}best_recog
                if writer is not None:
                    # Write the result to each file
                    ibest_writer["token"][key] = " ".join(token)
                    ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                    ibest_writer["vad"][key] = "{}".format(vadsegments)
                
                if text is not None:
                    postprocessed_result = postprocess_utils.sentence_postprocess(token, time_stamp)
                    if len(postprocessed_result) == 3:
                        text_postprocessed, time_stamp_postprocessed, word_lists = postprocessed_result[0], \
                                                                                   postprocessed_result[1], \
                                                                                   postprocessed_result[2]
                        text_postprocessed_punc, punc_id_list = text2punc(word_lists, 20)
                        text_postprocessed_punc_time_stamp = "predictions: {}  time_stamp: {}".format(
                            text_postprocessed_punc, time_stamp_postprocessed)
                    else:
                        text_postprocessed = ""
                        time_stamp_postprocessed = ""
                        word_lists = ""
                        text_postprocessed_punc_time_stamp = ""
                        punc_id_list = ""
                        text_postprocessed_punc = ""

                    item = {'key': key, 'value': text_postprocessed_punc_time_stamp, 'text': text_postprocessed,
                            'time_stamp': time_stamp_postprocessed, 'punc': punc_id_list, 'token': token}
                    if outputs_dict:
                        item = {'text_punc': text_postprocessed_punc, 'text': text_postprocessed,
                                'punc_id': punc_id_list, 'token': token, 'time_stamp': time_stamp_postprocessed}
                        item = {'key': key, 'value': item}
                    asr_result_list.append(item)
                    finish_count += 1
                    # asr_utils.print_progress(finish_count / file_count)
                    if writer is not None:
                        ibest_writer["text"][key] = text_postprocessed
                        ibest_writer["punc_id"][key] = "{}".format(punc_id_list)
                        ibest_writer["text_with_punc"][key] = text_postprocessed_punc_time_stamp
                        if time_stamp_postprocessed is not None:
                            ibest_writer["time_stamp"][key] = "{}".format(time_stamp_postprocessed)
                
                logging.info("decoding, utt: {}, predictions: {}, time_stamp: {}".format(key, text_postprocessed_punc,
                                                                                         time_stamp_postprocessed))
        
        logging.info("decoding, feature length total: {}, forward_time total: {:.4f}, rtf avg: {:.4f}".
                     format(length_total, forward_time_total, 100 * forward_time_total / (length_total * lfr_factor+1e-6)))
        return asr_result_list
    return _forward

def Text2Punc(
    train_config: Optional[str],
    model_file: Optional[str],
    device: str = "cpu",
    dtype: str = "float32",
):
   
    # 2. Build Model
    model, train_args = PunctuationTask.build_model_from_file(
        train_config, model_file, device)
    # Wrape model to make model.nll() data-parallel
    wrapped_model = ForwardAdaptor(model, "inference")
    wrapped_model.to(dtype=getattr(torch, dtype)).to(device=device).eval()
    # logging.info(f"Model:\n{model}")
    punc_list = train_args.punc_list
    period = 0
    for i in range(len(punc_list)):
        if punc_list[i] == ",":
            punc_list[i] = "，"
        elif punc_list[i] == "?":
            punc_list[i] = "？"
        elif punc_list[i] == "。":
            period = i
    preprocessor = CommonPreprocessor(
        train=False,
        token_type="word",
        token_list=train_args.token_list,
        bpemodel=train_args.bpemodel,
        text_cleaner=train_args.cleaner,
        g2p_type=train_args.g2p,
        text_name="text",
        non_linguistic_symbols=train_args.non_linguistic_symbols,
    )

    print("start decoding!!!")
    
    def _forward(words, split_size = 20):
        cache_sent = []
        mini_sentences = split_to_mini_sentence(words, split_size)
        new_mini_sentence = ""
        new_mini_sentence_punc = []
        cache_pop_trigger_limit = 200
        for mini_sentence_i in range(len(mini_sentences)):
            mini_sentence = mini_sentences[mini_sentence_i]
            mini_sentence = cache_sent + mini_sentence
            data = {"text": " ".join(mini_sentence)}
            batch = preprocessor(data=data, uid="12938712838719")
            batch["text_lengths"] = torch.from_numpy(np.array([len(batch["text"])], dtype='int32'))
            batch["text"] = torch.from_numpy(batch["text"])
            # Extend one dimension to fake a batch dim.
            batch["text"] = torch.unsqueeze(batch["text"], 0)
            batch = to_device(batch, device)
            y, _ = wrapped_model(**batch)
            _, indices = y.view(-1, y.shape[-1]).topk(1, dim=1)
            punctuations = indices
            if indices.size()[0] != 1:
                punctuations = torch.squeeze(indices)
            assert punctuations.size()[0] == len(mini_sentence)

            # Search for the last Period/QuestionMark as cache
            if mini_sentence_i < len(mini_sentences) - 1:
                sentenceEnd = -1
                last_comma_index = -1
                for i in range(len(punctuations) - 2, 1, -1):
                    if punc_list[punctuations[i]] == "。" or punc_list[punctuations[i]] == "？":
                        sentenceEnd = i
                        break
                    if last_comma_index < 0 and punc_list[punctuations[i]] == "，":
                        last_comma_index = i

                if sentenceEnd < 0 and len(mini_sentence) > cache_pop_trigger_limit and last_comma_index >= 0:
                    # The sentence it too long, cut off at a comma.
                    sentenceEnd = last_comma_index
                    punctuations[sentenceEnd] = period
                cache_sent = mini_sentence[sentenceEnd + 1:]
                mini_sentence = mini_sentence[0:sentenceEnd + 1]
                punctuations = punctuations[0:sentenceEnd + 1]

            # if len(punctuations) == 0:
            #    continue

            punctuations_np = punctuations.cpu().numpy()
            new_mini_sentence_punc += [int(x) for x in punctuations_np]
            words_with_punc = []
            for i in range(len(mini_sentence)):
                if i > 0:
                    if len(mini_sentence[i][0].encode()) == 1 and len(mini_sentence[i - 1][0].encode()) == 1:
                        mini_sentence[i] = " " + mini_sentence[i]
                words_with_punc.append(mini_sentence[i])
                if punc_list[punctuations[i]] != "_":
                    words_with_punc.append(punc_list[punctuations[i]])
            new_mini_sentence += "".join(words_with_punc)

        return new_mini_sentence, new_mini_sentence_punc
    return _forward

def get_parser():
    parser = config_argparse.ArgumentParser(
        description="ASR Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=False,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--asr_train_config",
        type=str,
        help="ASR training configuration",
    )
    group.add_argument(
        "--asr_model_file",
        type=str,
        help="ASR model parameter file",
    )
    group.add_argument(
        "--cmvn_file",
        type=str,
        help="Global cmvn file",
    )
    group.add_argument(
        "--lm_train_config",
        type=str,
        help="LM training configuration",
    )
    group.add_argument(
        "--lm_file",
        type=str,
        help="LM parameter file",
    )
    group.add_argument(
        "--word_lm_train_config",
        type=str,
        help="Word LM training configuration",
    )
    group.add_argument(
        "--word_lm_file",
        type=str,
        help="Word LM parameter file",
    )
    group.add_argument(
        "--ngram_file",
        type=str,
        help="N-gram parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
             "*_file will be overwritten",
    )

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    group.add_argument("--beam_size", type=int, default=20, help="Beam size")
    group.add_argument("--penalty", type=float, default=0.0, help="Insertion penalty")
    group.add_argument(
        "--maxlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain max output length. "
             "If maxlenratio=0.0 (default), it uses a end-detect "
             "function "
             "to automatically find maximum hypothesis lengths."
             "If maxlenratio<0.0, its absolute value is interpreted"
             "as a constant max output length",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length",
    )
    group.add_argument(
        "--ctc_weight",
        type=float,
        default=0.5,
        help="CTC weight in joint decoding",
    )
    group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
    group.add_argument("--ngram_weight", type=float, default=0.9, help="ngram weight")
    group.add_argument("--streaming", type=str2bool, default=False)
    group.add_argument("--time_stamp_writer", type=str2bool, default=False)

    group.add_argument(
        "--frontend_conf",
        default=None,
        help="",
    )
    group.add_argument("--raw_inputs", type=list, default=None)
    # example=[{'key':'EdevDEWdIYQ_0021','file':'/mnt/data/jiangyu.xzy/test_data/speech_io/SPEECHIO_ASR_ZH00007_zhibodaihuo/wav/EdevDEWdIYQ_0021.wav'}])

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ASR model. "
             "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
             "If not given, refers from the training args",
    )
    group.add_argument(
        "--vad_infer_config",
        type=str,
        help="VAD infer configuration",
    )
    group.add_argument(
        "--vad_model_file",
        type=str,
        help="VAD model parameter file",
    )
    group.add_argument(
        "--vad_cmvn_file",
        type=str,
        help="vad, Global cmvn file",
    )
    group.add_argument(
        "--punc_infer_config",
        type=str,
        help="VAD infer configuration",
    )
    group.add_argument(
        "--punc_model_file",
        type=str,
        help="VAD model parameter file",
    )
    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
