# !/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)


import argparse
import logging
import os
import sys
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import soundfile
import torch
from scipy.signal import medfilt
from typeguard import check_argument_types

from funasr.bin.diar_infer import Speech2DiarizationSOND, Speech2DiarizationEEND
from funasr.datasets.iterable_dataset import load_bytes
from funasr.build_utils.build_streaming_iterator import build_streaming_iterator
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import config_argparse
from funasr.utils.cli_utils import get_commandline_args
from funasr.utils.types import str2bool
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_none


def inference_sond(
        diar_train_config: str,
        diar_model_file: str,
        output_dir: Optional[str] = None,
        batch_size: int = 1,
        dtype: str = "float32",
        ngpu: int = 0,
        seed: int = 0,
        num_workers: int = 0,
        log_level: Union[int, str] = "INFO",
        key_file: Optional[str] = None,
        model_tag: Optional[str] = None,
        allow_variable_data_keys: bool = True,
        streaming: bool = False,
        smooth_size: int = 83,
        dur_threshold: int = 10,
        out_format: str = "vad",
        param_dict: Optional[dict] = None,
        mode: str = "sond",
        **kwargs,
):
    assert check_argument_types()
    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)
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

    # 2a. Build speech2xvec [Optional]
    if mode == "sond_demo" and param_dict is not None and "extract_profile" in param_dict and param_dict[
        "extract_profile"]:
        assert "sv_train_config" in param_dict, "sv_train_config must be provided param_dict."
        assert "sv_model_file" in param_dict, "sv_model_file must be provided in param_dict."
        sv_train_config = param_dict["sv_train_config"]
        sv_model_file = param_dict["sv_model_file"]
        if "model_dir" in param_dict:
            sv_train_config = os.path.join(param_dict["model_dir"], sv_train_config)
            sv_model_file = os.path.join(param_dict["model_dir"], sv_model_file)
        from funasr.bin.sv_infer import Speech2Xvector
        speech2xvector_kwargs = dict(
            sv_train_config=sv_train_config,
            sv_model_file=sv_model_file,
            device=device,
            dtype=dtype,
            streaming=streaming,
            embedding_node="resnet1_dense"
        )
        logging.info("speech2xvector_kwargs: {}".format(speech2xvector_kwargs))
        speech2xvector = Speech2Xvector.from_pretrained(
            model_tag=model_tag,
            **speech2xvector_kwargs,
        )
        speech2xvector.sv_model.eval()

    # 2b. Build speech2diar
    speech2diar_kwargs = dict(
        diar_train_config=diar_train_config,
        diar_model_file=diar_model_file,
        device=device,
        dtype=dtype,
        streaming=streaming,
        smooth_size=smooth_size,
        dur_threshold=dur_threshold,
    )
    logging.info("speech2diarization_kwargs: {}".format(speech2diar_kwargs))
    speech2diar = Speech2DiarizationSOND.from_pretrained(
        model_tag=model_tag,
        **speech2diar_kwargs,
    )
    speech2diar.diar_model.eval()

    def output_results_str(results: dict, uttid: str):
        rst = []
        mid = uttid.rsplit("-", 1)[0]
        for key in results:
            results[key] = [(x[0] / 100, x[1] / 100) for x in results[key]]
        if out_format == "vad":
            for spk, segs in results.items():
                rst.append("{} {}".format(spk, segs))
        else:
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
        logging.info("param_dict: {}".format(param_dict))
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, (list, tuple)):
                if not isinstance(raw_inputs[0], List):
                    raw_inputs = [raw_inputs]

                assert all([len(example) >= 2 for example in raw_inputs]), \
                    "The length of test case in raw_inputs must larger than 1 (>=2)."

                def prepare_dataset():
                    for idx, example in enumerate(raw_inputs):
                        # read waveform file
                        example = [load_bytes(x) if isinstance(x, bytes) else x
                                   for x in example]
                        example = [soundfile.read(x)[0] if isinstance(x, str) else x
                                   for x in example]
                        # convert torch tensor to numpy array
                        example = [x.numpy() if isinstance(example[0], torch.Tensor) else x
                                   for x in example]
                        speech = example[0]
                        logging.info("Extracting profiles for {} waveforms".format(len(example) - 1))
                        profile = [speech2xvector.calculate_embedding(x) for x in example[1:]]
                        profile = torch.cat(profile, dim=0)
                        yield ["test{}".format(idx)], {"speech": [speech], "profile": [profile]}

                loader = prepare_dataset()
            else:
                raise TypeError("raw_inputs must be a list or tuple in [speech, profile1, profile2, ...] ")
        else:
            # 3. Build data-iterator
            loader = build_streaming_iterator(
                task_name="diar",
                preprocess_args=None,
                data_path_and_name_and_type=data_path_and_name_and_type,
                dtype=dtype,
                batch_size=batch_size,
                key_file=key_file,
                num_workers=num_workers,
                use_collate_fn=False,
            )

        # 7. Start for-loop
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            output_writer = open("{}/result.txt".format(output_path), "w")
            pse_label_writer = open("{}/labels.txt".format(output_path), "w")
        logging.info("Start to diarize...")
        result_list = []
        for idx, (keys, batch) in enumerate(loader):
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            results, pse_labels = speech2diar(**batch)
            # Only supporting batch_size==1
            key, value = keys[0], output_results_str(results, keys[0])
            item = {"key": key, "value": value}
            result_list.append(item)
            if output_path is not None:
                output_writer.write(value)
                output_writer.flush()
                pse_label_writer.write("{} {}\n".format(key, " ".join(pse_labels)))
                pse_label_writer.flush()

            if idx % 100 == 0:
                logging.info("Processing {:5d}: {}".format(idx, key))

        if output_path is not None:
            output_writer.close()
            pse_label_writer.close()

        return result_list

    return _forward


def inference_eend(
        diar_train_config: str,
        diar_model_file: str,
        output_dir: Optional[str] = None,
        batch_size: int = 1,
        dtype: str = "float32",
        ngpu: int = 1,
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
    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)
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
    speech2diar = Speech2DiarizationEEND.from_pretrained(
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
            data_path_and_name_and_type = [raw_inputs[0], "speech", "sound"]
        loader = build_streaming_iterator(
            task_name="diar",
            preprocess_args=None,
            data_path_and_name_and_type=data_path_and_name_and_type,
            dtype=dtype,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
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

            # post process
            a = results[0][0].cpu().numpy()
            a = medfilt(a, (11, 1))
            rst = []
            for spkid, frames in enumerate(a.T):
                frames = np.pad(frames, (1, 1), 'constant')
                changes, = np.where(np.diff(frames, axis=0) != 0)
                fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
                for s, e in zip(changes[::2], changes[1::2]):
                    st = s / 10.
                    dur = (e - s) / 10.
                    rst.append(fmt.format(keys[0], st, dur, "{}_{}".format(keys[0], str(spkid))))

            # Only supporting batch_size==1
            value = "\n".join(rst)
            item = {"key": keys[0], "value": value}
            result_list.append(item)
            if output_path is not None:
                output_writer.write(value)
                output_writer.flush()

        if output_path is not None:
            output_writer.close()

        return result_list

    return _forward


def inference_launch(mode, **kwargs):
    if mode == "sond":
        return inference_sond(mode=mode, **kwargs)
    elif mode == "sond_demo":
        param_dict = {
            "extract_profile": True,
            "sv_train_config": "sv.yaml",
            "sv_model_file": "sv.pb",
        }
        if "param_dict" in kwargs and kwargs["param_dict"] is not None:
            for key in param_dict:
                if key not in kwargs["param_dict"]:
                    kwargs["param_dict"][key] = param_dict[key]
        else:
            kwargs["param_dict"] = param_dict
        return inference_sond(mode=mode, **kwargs)
    elif mode == "eend-ola":
        return inference_eend(mode=mode, **kwargs)
    else:
        logging.info("Unknown decoding mode: {}".format(mode))
        return None


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Speaker Verification",
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
        "--njob",
        type=int,
        default=1,
        help="The number of jobs for each gpu",
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
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=True)

    group = parser.add_argument_group("The model configuration related")
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
        "--diar_train_config",
        type=str,
        help="ASR training configuration",
    )
    group.add_argument(
        "--diar_model_file",
        type=str,
        help="ASR model parameter file",
    )
    group.add_argument(
        "--cmvn_file",
        type=str,
        help="Global CMVN file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
             "*_file will be overwritten",
    )

    group = parser.add_argument_group("The inference configuration related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument(
        "--diar_smooth_size",
        type=int,
        default=121,
        help="The smoothing size for post-processing"
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    parser.add_argument(
        "--mode",
        type=str,
        default="sond",
        help="The decoding mode",
    )
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)

    # set logging messages
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    logging.info("Decoding args: {}".format(kwargs))

    # gpu setting
    if args.ngpu > 0:
        jobid = int(args.output_dir.split(".")[-1])
        gpuid = args.gpuid_list.split(",")[(jobid - 1) // args.njob]
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

    inference_pipeline = inference_launch(**kwargs)
    return inference_pipeline(kwargs["data_path_and_name_and_type"])


if __name__ == "__main__":
    main()
