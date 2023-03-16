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

from collections import OrderedDict
import numpy as np
import soundfile
import torch
from torch.nn import functional as F
from typeguard import check_argument_types
from typeguard import check_return_type

from funasr.utils.cli_utils import get_commandline_args
from funasr.tasks.diar import DiarTask
from funasr.tasks.asr import ASRTask
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import config_argparse
from funasr.utils.types import str2bool
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_none
from scipy.ndimage import median_filter
from funasr.utils.misc import statistic_model_parameters
from funasr.datasets.iterable_dataset import load_bytes


class Speech2Diarization:
    """Speech2Xvector class

    Examples:
        >>> import soundfile
        >>> import numpy as np
        >>> speech2diar = Speech2Diarization("diar_sond_config.yml", "diar_sond.pb")
        >>> profile = np.load("profiles.npy")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2diar(audio, profile)
        {"spk1": [(int, int), ...], ...}

    """

    def __init__(
            self,
            diar_train_config: Union[Path, str] = None,
            diar_model_file: Union[Path, str] = None,
            device: Union[str, torch.device] = "cpu",
            batch_size: int = 1,
            dtype: str = "float32",
            streaming: bool = False,
            smooth_size: int = 83,
            dur_threshold: float = 10,
    ):
        assert check_argument_types()

        # TODO: 1. Build Diarization model
        diar_model, diar_train_args = DiarTask.build_model_from_file(
            config_file=diar_train_config,
            model_file=diar_model_file,
            device=device
        )
        logging.info("diar_model: {}".format(diar_model))
        logging.info("model parameter number: {}".format(statistic_model_parameters(diar_model)))
        logging.info("diar_train_args: {}".format(diar_train_args))
        diar_model.to(dtype=getattr(torch, dtype)).eval()

        self.diar_model = diar_model
        self.diar_train_args = diar_train_args
        self.token_list = diar_train_args.token_list
        self.smooth_size = smooth_size
        self.dur_threshold = dur_threshold
        self.device = device
        self.dtype = dtype

    def smooth_multi_labels(self, multi_label):
        multi_label = median_filter(multi_label, (self.smooth_size, 1), mode="constant", cval=0.0).astype(int)
        return multi_label

    @staticmethod
    def calc_spk_turns(label_arr, spk_list):
        turn_list = []
        length = label_arr.shape[0]
        n_spk = label_arr.shape[1]
        for k in range(n_spk):
            if spk_list[k] == "None":
                continue
            in_utt = False
            start = 0
            for i in range(length):
                if label_arr[i, k] == 1 and in_utt is False:
                    start = i
                    in_utt = True
                if label_arr[i, k] == 0 and in_utt is True:
                    turn_list.append([spk_list[k], start, i - start])
                    in_utt = False
            if in_utt:
                turn_list.append([spk_list[k], start, length - start])
        return turn_list

    @staticmethod
    def seq2arr(seq, vec_dim=8):
        def int2vec(x, vec_dim=8, dtype=np.int):
            b = ('{:0' + str(vec_dim) + 'b}').format(x)
            # little-endian order: lower bit first
            return (np.array(list(b)[::-1]) == '1').astype(dtype)

        # process oov
        seq = np.array([int(x) for x in seq])
        new_seq = []
        for i, x in enumerate(seq):
            if x < 2 ** vec_dim:
                new_seq.append(x)
            else:
                idx_list = np.where(seq < 2 ** vec_dim)[0]
                idx = np.abs(idx_list - i).argmin()
                new_seq.append(seq[idx_list[idx]])
        return np.row_stack([int2vec(x, vec_dim) for x in new_seq])

    def post_processing(self, raw_logits: torch.Tensor, spk_num: int, output_format: str = "speaker_turn"):
        logits_idx = raw_logits.argmax(-1)  # B, T, vocab_size -> B, T
        # upsampling outputs to match inputs
        ut = logits_idx.shape[1] * self.diar_model.encoder.time_ds_ratio
        logits_idx = F.upsample(
            logits_idx.unsqueeze(1).float(),
            size=(ut, ),
            mode="nearest",
        ).squeeze(1).long()
        logits_idx = logits_idx[0].tolist()
        pse_labels = [self.token_list[x] for x in logits_idx]
        if output_format == "pse_labels":
            return pse_labels, None

        multi_labels = self.seq2arr(pse_labels, spk_num)[:, :spk_num]  # remove padding speakers
        multi_labels = self.smooth_multi_labels(multi_labels)
        if output_format == "binary_labels":
            return multi_labels, None

        spk_list = ["spk{}".format(i + 1) for i in range(spk_num)]
        spk_turns = self.calc_spk_turns(multi_labels, spk_list)
        results = OrderedDict()
        for spk, st, dur in spk_turns:
            if spk not in results:
                results[spk] = []
            if dur > self.dur_threshold:
                results[spk].append((st, st+dur))

        # sort segments in start time ascending
        for spk in results:
            results[spk] = sorted(results[spk], key=lambda x: x[0])

        return results, pse_labels

    @torch.no_grad()
    def __call__(
            self,
            speech: Union[torch.Tensor, np.ndarray],
            profile: Union[torch.Tensor, np.ndarray],
            output_format: str = "speaker_turn"
    ):
        """Inference

        Args:
            speech: Input speech data
            profile: Speaker profiles
        Returns:
            diarization results for each speaker

        """
        assert check_argument_types()
        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)
        if isinstance(profile, np.ndarray):
            profile = torch.tensor(profile)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        profile = profile.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        speech_lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        profile_lengths = profile.new_full([1], dtype=torch.long, fill_value=profile.size(1))
        batch = {"speech": speech, "speech_lengths": speech_lengths,
                 "profile": profile, "profile_lengths": profile_lengths}
        # a. To device
        batch = to_device(batch, device=self.device)

        logits = self.diar_model.prediction_forward(**batch)
        results, pse_labels = self.post_processing(logits, profile.shape[1], output_format)

        return results, pse_labels

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

        return Speech2Diarization(**kwargs)


def inference_modelscope(
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
    if mode == "sond_demo" and param_dict is not None and "extract_profile" in param_dict and param_dict["extract_profile"]:
        assert "sv_train_config" in param_dict, "sv_train_config must be provided param_dict."
        assert "sv_model_file" in param_dict, "sv_model_file must be provided in param_dict."
        sv_train_config = param_dict["sv_train_config"]
        sv_model_file = param_dict["sv_model_file"]
        if "model_dir" in param_dict:
            sv_train_config = os.path.join(param_dict["model_dir"], sv_train_config)
            sv_model_file = os.path.join(param_dict["model_dir"], sv_model_file)
        from funasr.bin.sv_inference import Speech2Xvector
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
    speech2diar = Speech2Diarization.from_pretrained(
        model_tag=model_tag,
        **speech2diar_kwargs,
    )
    speech2diar.diar_model.eval()

    def output_results_str(results: dict, uttid: str):
        rst = []
        mid = uttid.rsplit("-", 1)[0]
        for key in results:
            results[key] = [(x[0]/100, x[1]/100) for x in results[key]]
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
                        logging.info("Extracting profiles for {} waveforms".format(len(example)-1))
                        profile = [speech2xvector.calculate_embedding(x) for x in example[1:]]
                        profile = torch.cat(profile, dim=0)
                        yield ["test{}".format(idx)], {"speech": [speech], "profile": [profile]}

                loader = prepare_dataset()
            else:
                raise TypeError("raw_inputs must be a list or tuple in [speech, profile1, profile2, ...] ")
        else:
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
