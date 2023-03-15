#!/usr/bin/env python3

""" Inference class definition for Transducer models."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from packaging.version import parse as V
from typeguard import check_argument_types, check_return_type

from funasr.models_transducer.beam_search_transducer import (
    BeamSearchTransducer,
    Hypothesis,
)
from funasr.models_transducer.utils import TooShortUttError
from funasr.fileio.datadir_writer import DatadirWriter
from funasr.tasks.asr_transducer import ASRTransducerTask
from funasr.tasks.lm import LMTask
from funasr.text.build_tokenizer import build_tokenizer
from funasr.text.token_id_converter import TokenIDConverter
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import config_argparse
from funasr.utils.types import str2bool, str2triple_str, str_or_none
from funasr.utils.cli_utils import get_commandline_args
from funasr.models.frontend.wav_frontend import WavFrontend

class Speech2Text:
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
        self.left_context = max(left_context, 0)
        self.right_context = max(right_context, 0)

        if not streaming or chunk_size == 0:
            self.streaming = False
            self.asr_model.encoder.dynamic_chunk_training = False
        
        if not simu_streaming or chunk_size == 0:
            self.simu_streaming = False
            self.asr_model.encoder.dynamic_chunk_training = False

        self.frontend = frontend
        self.window_size = self.chunk_size + self.right_context
        
        self._ctx = self.asr_model.encoder.get_encoder_input_size(
            self.window_size
        )
       
        #self.last_chunk_length = (
        #    self.asr_model.encoder.embed.min_frame_length + self.right_context + 1
        #) * self.hop_length

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
    ) -> List[Hypothesis]:
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
                    self.last_chunk_length - speech.size(0), speech.size(1),  dtype=speech.dtype
                )
                speech = torch.cat([speech, pad], dim=0) #feats, feats_length = self.apply_frontend(speech, is_final=is_final)

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
    def simu_streaming_decode(self, speech: Union[torch.Tensor, np.ndarray]) -> List[Hypothesis]:
        """Speech2Text call.
        Args:
            speech: Speech data. (S)
        Returns:
            nbest_hypothesis: N-best hypothesis.
        """
        assert check_argument_types()

        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)
        
        feats = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        feats_lengths = feats.new_full([1], dtype=torch.long, fill_value=feats.size(1))

        if self.asr_model.normalize is not None:
            feats, feats_lengths = self.asr_model.normalize(feats, feats_lengths)

        feats = to_device(feats, device=self.device)
        feats_lengths = to_device(feats_lengths, device=self.device)
        enc_out = self.asr_model.encoder.simu_chunk_forward(feats, feats_lengths, self.chunk_size, self.left_context, self.right_context)
        nbest_hyps = self.beam_search(enc_out[0])

        return nbest_hyps

    @torch.no_grad()
    def __call__(self, speech: Union[torch.Tensor, np.ndarray]) -> List[Hypothesis]:
        """Speech2Text call.
        Args:
            speech: Speech data. (S)
        Returns:
            nbest_hypothesis: N-best hypothesis.
        """
        assert check_argument_types()

        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)
        
        feats = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        feats_lengths = feats.new_full([1], dtype=torch.long, fill_value=feats.size(1))

        if self.asr_model.normalize is not None:
            feats, feats_lengths = self.asr_model.normalize(feats, feats_lengths)

        feats = to_device(feats, device=self.device)
        feats_lengths = to_device(feats_lengths, device=self.device)

        enc_out, _ = self.asr_model.encoder(feats, feats_lengths)

        nbest_hyps = self.beam_search(enc_out[0])

        return nbest_hyps

    def hypotheses_to_results(self, nbest_hyps: List[Hypothesis]) -> List[Any]:
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

        return Speech2Text(**kwargs)


def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    beam_size: int,
    ngpu: int,
    seed: int,
    lm_weight: float,
    nbest: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    asr_train_config: Optional[str],
    asr_model_file: Optional[str],
    cmvn_file: Optional[str],
    beam_search_config: Optional[dict],
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    key_file: Optional[str],
    allow_variable_data_keys: bool,
    quantize_asr_model: Optional[bool],
    quantize_modules: Optional[List[str]],
    quantize_dtype: Optional[str],
    streaming: Optional[bool],
    simu_streaming: Optional[bool],
    chunk_size: Optional[int],
    left_context: Optional[int],
    right_context: Optional[int],
    display_partial_hypotheses: bool,
    **kwargs,
) -> None:
    """Transducer model inference.
    Args:
        output_dir: Output directory path.
        batch_size: Batch decoding size.
        dtype: Data type.
        beam_size: Beam size.
        ngpu: Number of GPUs.
        seed: Random number generator seed.
        lm_weight: Weight of language model.
        nbest: Number of final hypothesis.
        num_workers: Number of workers.
        log_level: Level of verbose for logs.
        data_path_and_name_and_type:
        asr_train_config: ASR model training config path.
        asr_model_file: ASR model path.
        beam_search_config: Beam search config path.
        lm_train_config: Language Model training config path.
        lm_file: Language Model path.
        model_tag: Model tag.
        token_type: Type of token units.
        bpemodel: BPE model path.
        key_file: File key.
        allow_variable_data_keys: Whether to allow variable data keys.
        quantize_asr_model: Whether to apply dynamic quantization to ASR model.
        quantize_modules: List of module names to apply dynamic quantization on.
        quantize_dtype: Dynamic quantization data type.
        streaming: Whether to perform chunk-by-chunk inference.
        chunk_size: Number of frames in chunk AFTER subsampling.
        left_context: Number of frames in left context AFTER subsampling.
        right_context: Number of frames in right context AFTER subsampling.
        display_partial_hypotheses: Whether to display partial hypotheses.
    """
    assert check_argument_types()

    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
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
    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        cmvn_file=cmvn_file,
        beam_search_config=beam_search_config,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        dtype=dtype,
        beam_size=beam_size,
        lm_weight=lm_weight,
        nbest=nbest,
        quantize_asr_model=quantize_asr_model,
        quantize_modules=quantize_modules,
        quantize_dtype=quantize_dtype,
        streaming=streaming,
        simu_streaming=simu_streaming,
        chunk_size=chunk_size,
        left_context=left_context,
        right_context=right_context,
    )
    speech2text = Speech2Text.from_pretrained(
        model_tag=model_tag,
        **speech2text_kwargs,
    )

    # 3. Build data-iterator
    loader = ASRTransducerTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ASRTransducerTask.build_preprocess_fn(
            speech2text.asr_train_args, False
        ),
        collate_fn=ASRTransducerTask.build_collate_fn(
            speech2text.asr_train_args, False
        ),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 4 .Start for-loop
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys

            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}
            assert len(batch.keys()) == 1

            try:
                if speech2text.streaming:
                    speech = batch["speech"]

                    _steps = len(speech) // speech2text._ctx
                    _end = 0
                    for i in range(_steps):
                        _end = (i + 1) * speech2text._ctx

                        speech2text.streaming_decode(
                            speech[i * speech2text._ctx : _end], is_final=False
                        )

                    final_hyps = speech2text.streaming_decode(
                        speech[_end : len(speech)], is_final=True
                    )
                elif speech2text.simu_streaming:
                    final_hyps = speech2text.simu_streaming_decode(**batch)
                else:
                    final_hyps = speech2text(**batch)

                results = speech2text.hypotheses_to_results(final_hyps)
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                hyp = Hypothesis(score=0.0, yseq=[], dec_state=None)
                results = [[" ", ["<space>"], [2], hyp]] * nbest

            key = keys[0]
            for n, (text, token, token_int, hyp) in zip(range(1, nbest + 1), results):
                ibest_writer = writer[f"{n}best_recog"]

                ibest_writer["token"][key] = " ".join(token)
                ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                ibest_writer["score"][key] = str(hyp.score)

                if text is not None:
                    ibest_writer["text"][key] = text


def get_parser():
    """Get Transducer model inference parser."""

    parser = config_argparse.ArgumentParser(
        description="ASR Transducer Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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
        required=True,
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
    group.add_argument("--beam_size", type=int, default=5, help="Beam size")
    group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
    group.add_argument(
        "--beam_search_config",
        default={},
        help="The keyword arguments for transducer beam search.",
    )

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

    group = parser.add_argument_group("Dynamic quantization related")
    parser.add_argument(
        "--quantize_asr_model",
        type=bool,
        default=False,
        help="Apply dynamic quantization to ASR model.",
    )
    parser.add_argument(
        "--quantize_modules",
        nargs="*",
        default=None,
        help="""Module names to apply dynamic quantization on.
        The module names are provided as a list, where each name is separated
        by a comma (e.g.: --quantize-config=[Linear,LSTM,GRU]).
        Each specified name should be an attribute of 'torch.nn', e.g.:
        torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU, ...""",
    )
    parser.add_argument(
        "--quantize_dtype",
        type=str,
        default="qint8",
        choices=["float16", "qint8"],
        help="Dtype for dynamic quantization.",
    )

    group = parser.add_argument_group("Streaming related")
    parser.add_argument(
        "--streaming",
        type=bool,
        default=False,
        help="Whether to perform chunk-by-chunk inference.",
    )
    parser.add_argument(
        "--simu_streaming",
        type=bool,
        default=False,
        help="Whether to simulate chunk-by-chunk inference.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=16,
        help="Number of frames in chunk AFTER subsampling.",
    )
    parser.add_argument(
        "--left_context",
        type=int,
        default=32,
        help="Number of frames in left context of the chunk AFTER subsampling.",
    )
    parser.add_argument(
        "--right_context",
        type=int,
        default=0,
        help="Number of frames in right context of the chunk AFTER subsampling.",
    )
    parser.add_argument(
        "--display_partial_hypotheses",
        type=bool,
        default=False,
        help="Whether to display partial hypotheses during chunk-by-chunk inference.",
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

