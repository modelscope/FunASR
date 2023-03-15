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
from typeguard import check_argument_types

from funasr.models.frontend.wav_frontend import WavFrontendMel23
from funasr.tasks.diar import EENDOLADiarTask
from funasr.torch_utils.device_funcs import to_device
from funasr.utils import config_argparse
from funasr.utils.cli_utils import get_commandline_args
from funasr.utils.types import str2bool
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_none


class Speech2Diarization:
    """Speech2Diarlization class

    Examples:
        >>> import soundfile
        >>> import numpy as np
        >>> speech2diar = Speech2Diarization("diar_sond_config.yml", "diar_sond.pth")
        >>> profile = np.load("profiles.npy")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2diar(audio, profile)
        {"spk1": [(int, int), ...], ...}

    """

    def __init__(
            self,
            diar_train_config: Union[Path, str] = None,
            diar_model_file: Union[Path, str] = None,
            device: str = "cpu",
            dtype: str = "float32",
    ):
        assert check_argument_types()

        # 1. Build Diarization model
        diar_model, diar_train_args = EENDOLADiarTask.build_model_from_file(
            config_file=diar_train_config,
            model_file=diar_model_file,
            device=device
        )
        frontend = None
        if diar_train_args.frontend is not None and diar_train_args.frontend_conf is not None:
            frontend = WavFrontendMel23(**diar_train_args.frontend_conf)

        # set up seed for eda
        np.random.seed(diar_train_args.seed)
        torch.manual_seed(diar_train_args.seed)
        torch.cuda.manual_seed(diar_train_args.seed)
        os.environ['PYTORCH_SEED'] = str(diar_train_args.seed)
        logging.info("diar_model: {}".format(diar_model))
        logging.info("diar_train_args: {}".format(diar_train_args))
        diar_model.to(dtype=getattr(torch, dtype)).eval()

        self.diar_model = diar_model
        self.diar_train_args = diar_train_args
        self.device = device
        self.dtype = dtype
        self.frontend = frontend

    @torch.no_grad()
    def __call__(
            self,
            speech: Union[torch.Tensor, np.ndarray],
            speech_lengths: Union[torch.Tensor, np.ndarray] = None
    ):
        """Inference

        Args:
            speech: Input speech data
        Returns:
            diarization results

        """
        assert check_argument_types()
        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        if self.frontend is not None:
            feats, feats_len = self.frontend.forward(speech, speech_lengths)
            feats = to_device(feats, device=self.device)
            feats_len = feats_len.int()
            self.diar_model.frontend = None
        else:
            feats = speech
            feats_len = speech_lengths
        batch = {"speech": feats, "speech_lengths": feats_len}
        batch = to_device(batch, device=self.device)
        results = self.diar_model.estimate_sequential(**batch)

        return results

    @staticmethod
    def from_pretrained(
            model_tag: Optional[str] = None,
            **kwargs: Optional[Any],
    ):
        """Build Speech2Diarization instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2Diarization: Speech2Diarization instance.

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

        return Speech2Diarization(**kwargs)


def inference_modelscope(
        diar_train_config: str,
        diar_model_file: str,
        output_dir: Optional[str] = None,
        batch_size: int = 1,
        dtype: str = "float32",
        ngpu: int = 0,
        num_workers: int = 0,
        log_level: Union[int, str] = "INFO",
        key_file: Optional[str] = None,
        model_tag: Optional[str] = None,
        allow_variable_data_keys: bool = True,
        streaming: bool = False,
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

    # 1. Build speech2diar
    speech2diar_kwargs = dict(
        diar_train_config=diar_train_config,
        diar_model_file=diar_model_file,
        device=device,
        dtype=dtype,
    )
    logging.info("speech2diarization_kwargs: {}".format(speech2diar_kwargs))
    speech2diar = Speech2Diarization.from_pretrained(
        model_tag=model_tag,
        **speech2diar_kwargs,
    )
    speech2diar.diar_model.eval()

    def output_results_str(results: dict, uttid: str):
        rst = []
        mid = uttid.rsplit("-", 1)[0]
        for key in results:
            results[key] = [(x[0] / 100, x[1] / 100) for x in results[key]]
        template = "SPEAKER {} 0 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>"
        for spk, segs in results.items():
            rst.extend([template.format(mid, st, ed, spk) for st, ed in segs])

        return "\n".join(rst)

    def _forward(
            data_path_and_name_and_type: Sequence[Tuple[str, str, str]] = None,
            raw_inputs: List[List[Union[np.ndarray, torch.Tensor, str, bytes]]] = None,
            output_dir_v2: Optional[str] = None,
            param_dict: Optional[dict] = None,
    ):
        # 2. Build data-iterator
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]
        loader = EENDOLADiarTask.build_streaming_iterator(
            data_path_and_name_and_type,
            dtype=dtype,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
            preprocess_fn=EENDOLADiarTask.build_preprocess_fn(speech2diar.diar_train_args, False),
            collate_fn=EENDOLADiarTask.build_collate_fn(speech2diar.diar_train_args, False),
            allow_variable_data_keys=allow_variable_data_keys,
            inference=True,
        )

        # 3. Start for-loop
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            output_writer = open("{}/result.txt".format(output_path), "w")
        result_list = []
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            # batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            results = speech2diar(**batch)
            # Only supporting batch_size==1
            key, value = keys[0], output_results_str(results, keys[0])
            item = {"key": key, "value": value}
            result_list.append(item)
            if output_path is not None:
                output_writer.write(value)
                output_writer.flush()

        if output_path is not None:
            output_writer.close()

        return result_list

    return _forward


def inference(
        data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
        diar_train_config: Optional[str],
        diar_model_file: Optional[str],
        output_dir: Optional[str] = None,
        batch_size: int = 1,
        dtype: str = "float32",
        ngpu: int = 0,
        seed: int = 0,
        num_workers: int = 1,
        log_level: Union[int, str] = "INFO",
        key_file: Optional[str] = None,
        model_tag: Optional[str] = None,
        allow_variable_data_keys: bool = True,
        streaming: bool = False,
        smooth_size: int = 83,
        dur_threshold: int = 10,
        out_format: str = "vad",
        **kwargs,
):
    inference_pipeline = inference_modelscope(
        diar_train_config=diar_train_config,
        diar_model_file=diar_model_file,
        output_dir=output_dir,
        batch_size=batch_size,
        dtype=dtype,
        ngpu=ngpu,
        seed=seed,
        num_workers=num_workers,
        log_level=log_level,
        key_file=key_file,
        model_tag=model_tag,
        allow_variable_data_keys=allow_variable_data_keys,
        streaming=streaming,
        smooth_size=smooth_size,
        dur_threshold=dur_threshold,
        out_format=out_format,
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
        "--diar_train_config",
        type=str,
        help="diarization training configuration",
    )
    group.add_argument(
        "--diar_model_file",
        type=str,
        help="diarization model parameter file",
    )
    group.add_argument(
        "--dur_threshold",
        type=int,
        default=10,
        help="The threshold for short segments in number frames"
    )
    parser.add_argument(
        "--smooth_size",
        type=int,
        default=83,
        help="The smoothing window length in number frames"
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
        gpuid = args.gpuid_list.split(",")[jobid - 1]
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
