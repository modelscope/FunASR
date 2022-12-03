#!/usr/bin/env python3
# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import sys
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import Dict

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from funasr.fileio.datadir_writer import DatadirWriter
from funasr.modules.beam_search.batch_beam_search import BatchBeamSearch
from funasr.modules.beam_search.batch_beam_search_online_sim import BatchBeamSearchOnlineSim
from funasr.modules.beam_search.beam_search import BeamSearch
from funasr.modules.beam_search.beam_search import Hypothesis
from funasr.modules.scorers.ctc import CTCPrefixScorer
from funasr.modules.scorers.length_bonus import LengthBonus
from funasr.modules.scorers.scorer_interface import BatchScorerInterface
from funasr.modules.subsampling import TooShortUttError
from funasr.tasks.asr import ASRTask
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

from modelscope.utils.logger import get_logger

logger = get_logger()

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
            asr_train_config, asr_model_file, device
        )
        if asr_model.frontend is None and frontend_conf is not None:
            frontend = WavFrontend(**frontend_conf)
            asr_model.frontend = frontend
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

        # TODO(karita): make all scorers batchfied
        if batch_size == 1:
            non_batch = [
                k
                for k, v in beam_search.full_scorers.items()
                if not isinstance(v, BatchScorerInterface)
            ]
            if len(non_batch) == 0:
                if streaming:
                    beam_search.__class__ = BatchBeamSearchOnlineSim
                    beam_search.set_streaming_config(asr_train_config)
                    logging.info(
                        "BatchBeamSearchOnlineSim implementation is selected."
                    )
                else:
                    beam_search.__class__ = BatchBeamSearch
                    logging.info("BatchBeamSearch implementation is selected.")
            else:
                logging.warning(
                    f"As non-batch scorers {non_batch} are found, "
                    f"fall back to non-batch implementation."
                )

            beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
            for scorer in scorers.values():
                if isinstance(scorer, torch.nn.Module):
                    scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
            logging.info(f"Beam_search: {beam_search}")
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

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray]
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

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        lfr_factor = max(1, (speech.size()[-1] // 80) - 1)
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

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


def inference(
        maxlenratio: float,
        minlenratio: float,
        batch_size: int,
        dtype: str,
        beam_size: int,
        ngpu: int,
        seed: int,
        ctc_weight: float,
        lm_weight: float,
        ngram_weight: float,
        penalty: float,
        nbest: int,
        num_workers: int,
        log_level: Union[int, str],
        data_path_and_name_and_type: list,
        audio_lists: Union[List[Any], bytes],
        key_file: Optional[str],
        asr_train_config: Optional[str],
        asr_model_file: Optional[str],
        lm_train_config: Optional[str],
        lm_file: Optional[str],
        word_lm_train_config: Optional[str],
        token_type: Optional[str],
        bpemodel: Optional[str],
        output_dir: Optional[str],
        allow_variable_data_keys: bool,
        streaming: bool,
        frontend_conf: dict = None,
        fs: Union[dict, int] = 16000,
        **kwargs,
) -> List[Any]:
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"
    features_type: str = data_path_and_name_and_type[1]
    hop_length: int = 160
    sr: int = 16000
    if isinstance(fs, int):
        sr = fs
    else:
        if 'model_fs' in fs and fs['model_fs'] is not None:
            sr = fs['model_fs']
    if features_type != 'sound':
        frontend_conf = None
    if frontend_conf is not None:
        if 'hop_length' in frontend_conf:
            hop_length = frontend_conf['hop_length']

    finish_count = 0
    file_count = 1
    if isinstance(audio_lists, bytes):
        file_count = 1
    else:
        file_count = len(audio_lists)
    if len(data_path_and_name_and_type) >= 3 and frontend_conf is not None:
        mvn_file = data_path_and_name_and_type[2]
        mvn_data = wav_utils.extract_CMVN_featrures(mvn_file)
        frontend_conf['mvn_data'] = mvn_data
    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
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
        streaming=streaming,
        frontend_conf=frontend_conf,
    )
    speech2text = Speech2Text(**speech2text_kwargs)
    data_path_and_name_and_type_new = [
        audio_lists, data_path_and_name_and_type[0], data_path_and_name_and_type[1]
    ]
    # 3. Build data-iterator
    loader = ASRTask.build_streaming_iterator_modelscope(
        data_path_and_name_and_type_new,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ASRTask.build_preprocess_fn(speech2text.asr_train_args, False),
        collate_fn=ASRTask.build_collate_fn(speech2text.asr_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
        sample_rate=fs
    )

    # 7 .Start for-loop
    # FIXME(kamo): The output format should be discussed about
    asr_result_list = []
    for keys, batch in loader:
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert len(keys) == _bs, f"{len(keys)} != {_bs}"
        batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

        # N-best list of (text, token, token_int, hyp_object)
        try:
            results = speech2text(**batch)
        except TooShortUttError as e:
            logging.warning(f"Utterance {keys} {e}")
            hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
            results = [[" ", ["<space>"], [2], hyp]] * nbest

        # Only supporting batch_size==1
        key = keys[0]
        for n, (text, token, token_int, hyp) in zip(range(1, nbest + 1), results):
            if text is not None:
                text_postprocessed = postprocess_utils.sentence_postprocess(token)
                item = {'key': key, 'value': text_postprocessed}
                asr_result_list.append(item)
        finish_count += 1
        asr_utils.print_progress(finish_count / file_count)

    return asr_result_list



def set_parameters(language: str = None,
                   sample_rate: Union[int, Dict[Any, int]] = None):
    if language is not None:
        global global_asr_language
        global_asr_language = language
    if sample_rate is not None:
        global global_sample_rate
        global_sample_rate = sample_rate


def asr_inference(maxlenratio: float,
                  minlenratio: float,
                  beam_size: int,
                  ngpu: int,
                  ctc_weight: float,
                  lm_weight: float,
                  penalty: float,
                  name_and_type: list,
                  audio_lists: Union[List[Any], bytes],
                  asr_train_config: Optional[str],
                  asr_model_file: Optional[str],
                  nbest: int = 1,
                  num_workers: int = 1,
                  log_level: Union[int, str] = 'INFO',
                  batch_size: int = 1,
                  dtype: str = 'float32',
                  seed: int = 0,
                  key_file: Optional[str] = None,
                  lm_train_config: Optional[str] = None,
                  lm_file: Optional[str] = None,
                  word_lm_train_config: Optional[str] = None,
                  word_lm_file: Optional[str] = None,
                  ngram_file: Optional[str] = None,
                  ngram_weight: float = 0.9,
                  model_tag: Optional[str] = None,
                  token_type: Optional[str] = None,
                  bpemodel: Optional[str] = None,
                  allow_variable_data_keys: bool = False,
                  transducer_conf: Optional[dict] = None,
                  streaming: bool = False,
                  frontend_conf: dict = None,
                  fs: Union[dict, int] = None,
                  lang: Optional[str] = None,
                  outputdir: Optional[str] = None):
    if lang is not None:
        global global_asr_language
        global_asr_language = lang
    if fs is not None:
        global global_sample_rate
        global_sample_rate = fs

    # force use CPU if data type is bytes
    if isinstance(audio_lists, bytes):
        num_workers = 0
        ngpu = 0

    return inference(output_dir=outputdir,
                     maxlenratio=maxlenratio,
                     minlenratio=minlenratio,
                     batch_size=batch_size,
                     dtype=dtype,
                     beam_size=beam_size,
                     ngpu=ngpu,
                     seed=seed,
                     ctc_weight=ctc_weight,
                     lm_weight=lm_weight,
                     ngram_weight=ngram_weight,
                     penalty=penalty,
                     nbest=nbest,
                     num_workers=num_workers,
                     log_level=log_level,
                     data_path_and_name_and_type=name_and_type,
                     audio_lists=audio_lists,
                     key_file=key_file,
                     asr_train_config=asr_train_config,
                     asr_model_file=asr_model_file,
                     lm_train_config=lm_train_config,
                     lm_file=lm_file,
                     word_lm_train_config=word_lm_train_config,
                     word_lm_file=word_lm_file,
                     ngram_file=ngram_file,
                     model_tag=model_tag,
                     token_type=token_type,
                     bpemodel=bpemodel,
                     allow_variable_data_keys=allow_variable_data_keys,
                     transducer_conf=transducer_conf,
                     streaming=streaming,
                     frontend_conf=frontend_conf)


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
    parser.add_argument(
        "--gpuid_list",
        type=str,
        default="",
        help="The visible gpus",
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
        required=True,
        action="append",
    )
    group.add_argument("--audio_lists", type=list,
                       default=[{'key':'EdevDEWdIYQ_0021',
                                 'file':'/mnt/data/jiangyu.xzy/test_data/speech_io/SPEECHIO_ASR_ZH00007_zhibodaihuo/wav/EdevDEWdIYQ_0021.wav'}])
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
