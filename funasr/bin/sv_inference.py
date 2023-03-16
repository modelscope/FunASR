#!/usr/bin/env python3
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
from kaldiio import WriteHelper
from typeguard import check_argument_types
from typeguard import check_return_type

from funasr.utils.cli_utils import get_commandline_args
from funasr.tasks.sv import SVTask
from funasr.tasks.asr import ASRTask
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import config_argparse
from funasr.utils.types import str2bool
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_none
from funasr.utils.misc import statistic_model_parameters

class Speech2Xvector:
    """Speech2Xvector class

    Examples:
        >>> import soundfile
        >>> speech2xvector = Speech2Xvector("sv_config.yml", "sv.pb")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2xvector(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
            self,
            sv_train_config: Union[Path, str] = None,
            sv_model_file: Union[Path, str] = None,
            device: str = "cpu",
            batch_size: int = 1,
            dtype: str = "float32",
            streaming: bool = False,
            embedding_node: str = "resnet1_dense",
    ):
        assert check_argument_types()

        # TODO: 1. Build SV model
        sv_model, sv_train_args = SVTask.build_model_from_file(
            config_file=sv_train_config,
            model_file=sv_model_file,
            device=device
        )
        logging.info("sv_model: {}".format(sv_model))
        logging.info("model parameter number: {}".format(statistic_model_parameters(sv_model)))
        logging.info("sv_train_args: {}".format(sv_train_args))
        sv_model.to(dtype=getattr(torch, dtype)).eval()

        self.sv_model = sv_model
        self.sv_train_args = sv_train_args
        self.device = device
        self.dtype = dtype
        self.embedding_node = embedding_node

    @torch.no_grad()
    def calculate_embedding(self, speech: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, ilens = self.sv_model.encode(**batch)

        # c. Forward Pooling
        pooling = self.sv_model.pooling_layer(enc)

        # d. Forward Decoder
        outputs, embeddings = self.sv_model.decoder(pooling)

        if self.embedding_node not in embeddings:
            raise ValueError("Required embedding node {} not in {}".format(
                self.embedding_node, embeddings.keys()))

        return embeddings[self.embedding_node]

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray],
            ref_speech: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        """Inference

        Args:
            speech: Input speech data
            ref_speech: Reference speech to compare
        Returns:
            embedding, ref_embedding, similarity_score

        """
        assert check_argument_types()
        self.sv_model.eval()
        embedding = self.calculate_embedding(speech)
        ref_emb, score = None, None
        if ref_speech is not None:
            ref_emb = self.calculate_embedding(ref_speech)
            score = torch.cosine_similarity(embedding, ref_emb)

        results = (embedding, ref_emb, score)
        assert check_return_type(results)
        return results

    @staticmethod
    def from_pretrained(
            model_tag: Optional[str] = None,
            **kwargs: Optional[Any],
    ):
        """Build Speech2Xvector instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2Xvector: Speech2Xvector instance.

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

        return Speech2Xvector(**kwargs)


def inference_modelscope(
        output_dir: Optional[str] = None,
        batch_size: int = 1,
        dtype: str = "float32",
        ngpu: int = 1,
        seed: int = 0,
        num_workers: int = 0,
        log_level: Union[int, str] = "INFO",
        key_file: Optional[str] = None,
        sv_train_config: Optional[str] = "sv.yaml",
        sv_model_file: Optional[str] =  "sv.pb",
        model_tag: Optional[str] = None,
        allow_variable_data_keys: bool = True,
        streaming: bool = False,
        embedding_node: str = "resnet1_dense",
        sv_threshold: float = 0.9465,
        param_dict: Optional[dict] = None,
        **kwargs,
):
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    logging.info("param_dict: {}".format(param_dict))

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2xvector
    speech2xvector_kwargs = dict(
        sv_train_config=sv_train_config,
        sv_model_file=sv_model_file,
        device=device,
        dtype=dtype,
        streaming=streaming,
        embedding_node=embedding_node
    )
    logging.info("speech2xvector_kwargs: {}".format(speech2xvector_kwargs))
    speech2xvector = Speech2Xvector.from_pretrained(
        model_tag=model_tag,
        **speech2xvector_kwargs,
    )
    speech2xvector.sv_model.eval()

    def _forward(
            data_path_and_name_and_type: Sequence[Tuple[str, str, str]] = None,
            raw_inputs: Union[np.ndarray, torch.Tensor] = None,
            output_dir_v2: Optional[str] = None,
            param_dict: Optional[dict] = None,
    ):
        logging.info("param_dict: {}".format(param_dict))
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]

        # 3. Build data-iterator
        loader = ASRTask.build_streaming_iterator(
            data_path_and_name_and_type,
            dtype=dtype,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
            preprocess_fn=None,
            collate_fn=None,
            allow_variable_data_keys=allow_variable_data_keys,
            inference=True,
        )

        # 7 .Start for-loop
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        embd_writer, ref_embd_writer, score_writer = None, None, None
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            embd_writer = WriteHelper("ark,scp:{}/xvector.ark,{}/xvector.scp".format(output_path, output_path))
        sv_result_list = []
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            embedding, ref_embedding, score = speech2xvector(**batch)
            # Only supporting batch_size==1
            key = keys[0]
            normalized_score = 0.0
            if score is not None:
                score = score.item()
                normalized_score = max(score - sv_threshold, 0.0) / (1.0 - sv_threshold) * 100.0
                item = {"key": key, "value": normalized_score}
            else:
                item = {"key": key, "value": embedding.squeeze(0).cpu().numpy()}
            sv_result_list.append(item)
            if output_path is not None:
                embd_writer(key, embedding[0].cpu().numpy())
                if ref_embedding is not None:
                    if ref_embd_writer is None:
                        ref_embd_writer = WriteHelper(
                            "ark,scp:{}/ref_xvector.ark,{}/ref_xvector.scp".format(output_path, output_path)
                        )
                        score_writer = open(os.path.join(output_path, "score.txt"), "w")
                    ref_embd_writer(key, ref_embedding[0].cpu().numpy())
                    score_writer.write("{} {:.6f}\n".format(key, normalized_score))

        if output_path is not None:
            embd_writer.close()
            if ref_embd_writer is not None:
                ref_embd_writer.close()
                score_writer.close()

        return sv_result_list

    return _forward


def inference(
        output_dir: Optional[str],
        batch_size: int,
        dtype: str,
        ngpu: int,
        seed: int,
        num_workers: int,
        log_level: Union[int, str],
        data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
        key_file: Optional[str],
        sv_train_config: Optional[str],
        sv_model_file: Optional[str],
        model_tag: Optional[str],
        allow_variable_data_keys: bool = True,
        streaming: bool = False,
        embedding_node: str = "resnet1_dense",
        sv_threshold: float = 0.9465,
        **kwargs,
):
    inference_pipeline = inference_modelscope(
        output_dir=output_dir,
        batch_size=batch_size,
        dtype=dtype,
        ngpu=ngpu,
        seed=seed,
        num_workers=num_workers,
        log_level=log_level,
        key_file=key_file,
        sv_train_config=sv_train_config,
        sv_model_file=sv_model_file,
        model_tag=model_tag,
        allow_variable_data_keys=allow_variable_data_keys,
        streaming=streaming,
        embedding_node=embedding_node,
        sv_threshold=sv_threshold,
        **kwargs,
    )

    return inference_pipeline(data_path_and_name_and_type, raw_inputs=None)


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Speaker verification/x-vector extraction",
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

    parser.add_argument("--output_dir", type=str, required=False)
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
        required=False,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--sv_train_config",
        type=str,
        help="SV training configuration",
    )
    group.add_argument(
        "--sv_model_file",
        type=str,
        help="SV model parameter file",
    )
    group.add_argument(
        "--sv_threshold",
        type=float,
        default=0.9465,
        help="The threshold for verification"
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
             "*_file will be overwritten",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    parser.add_argument("--streaming", type=str2bool, default=False)
    parser.add_argument("--embedding_node", type=str, default="resnet1_dense")

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    logging.info("args: {}".format(kwargs))
    if args.output_dir is None:
        jobid, n_gpu = 1, 1
        gpuid = args.gpuid_list.split(",")[jobid-1]
    else:
        jobid = int(args.output_dir.split(".")[-1])
        n_gpu = len(args.gpuid_list.split(","))
        gpuid = args.gpuid_list.split(",")[(jobid - 1) % n_gpu]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    results_list = inference(**kwargs)
    for results in results_list:
        print("{} {}".format(results["key"], results["value"]))


if __name__ == "__main__":
    main()
