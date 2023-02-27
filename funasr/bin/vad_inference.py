import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import Dict

import math
import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from funasr.fileio.datadir_writer import DatadirWriter
from funasr.modules.scorers.scorer_interface import BatchScorerInterface
from funasr.modules.subsampling import TooShortUttError
from funasr.tasks.vad import VADTask
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import config_argparse
from funasr.utils.cli_utils import get_commandline_args
from funasr.utils.types import str2bool
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_none
from funasr.utils import asr_utils, wav_utils, postprocess_utils
from funasr.models.frontend.wav_frontend import WavFrontend

header_colors = '\033[95m'
end_colors = '\033[0m'

global_asr_language: str = 'zh-cn'
global_sample_rate: Union[int, Dict[Any, int]] = {
    'audio_fs': 16000,
    'model_fs': 16000
}


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

        logging.info("vad_model: {}".format(vad_model))
        logging.info("vad_infer_args: {}".format(vad_infer_args))
        vad_model.to(dtype=getattr(torch, dtype)).eval()

        self.vad_model = vad_model
        self.vad_infer_args = vad_infer_args
        self.device = device
        self.dtype = dtype
        self.frontend = frontend
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray] = None
    ) -> Tuple[List[List[int]], Dict[str, torch.Tensor]]:
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

        # b. Forward Encoder streaming
        t_offset = 0
        step = min(feats_len.max(), 6000)
        segments = [[]] * self.batch_size
        for t_offset in range(0, feats_len, min(step, feats_len - t_offset)):
            if t_offset + step >= feats_len - 1:
                step = feats_len - t_offset
                is_final = True
            else:
                is_final = False
            batch = {
                "feats": feats[:, t_offset:t_offset + step, :],
                "waveform": speech[:, t_offset * 160:min(speech.shape[-1], (t_offset + step - 1) * 160 + 400)],
                "is_final": is_final
            }
            # a. To device
            batch = to_device(batch, device=self.device)
            segments_part = self.vad_model(**batch)
            if segments_part:
                for batch_num in range(0, self.batch_size):
                    segments[batch_num] += segments_part[batch_num]
        return fbanks, segments


def inference(
        batch_size: int,
        ngpu: int,
        log_level: Union[int, str],
        data_path_and_name_and_type,
        vad_infer_config: Optional[str],
        vad_model_file: Optional[str],
        vad_cmvn_file: Optional[str] = None,
        raw_inputs: Union[np.ndarray, torch.Tensor] = None,
        key_file: Optional[str] = None,
        allow_variable_data_keys: bool = False,
        output_dir: Optional[str] = None,
        dtype: str = "float32",
        seed: int = 0,
        num_workers: int = 1,
        **kwargs,
):
    inference_pipeline = inference_modelscope(
        batch_size=batch_size,
        ngpu=ngpu,
        log_level=log_level,
        vad_infer_config=vad_infer_config,
        vad_model_file=vad_model_file,
        vad_cmvn_file=vad_cmvn_file,
        key_file=key_file,
        allow_variable_data_keys=allow_variable_data_keys,
        output_dir=output_dir,
        dtype=dtype,
        seed=seed,
        num_workers=num_workers,
        **kwargs,
    )
    return inference_pipeline(data_path_and_name_and_type, raw_inputs)


def inference_modelscope(
        batch_size: int,
        ngpu: int,
        log_level: Union[int, str],
        # data_path_and_name_and_type,
        vad_infer_config: Optional[str],
        vad_model_file: Optional[str],
        vad_cmvn_file: Optional[str] = None,
        # raw_inputs: Union[np.ndarray, torch.Tensor] = None,
        key_file: Optional[str] = None,
        allow_variable_data_keys: bool = False,
        output_dir: Optional[str] = None,
        dtype: str = "float32",
        seed: int = 0,
        num_workers: int = 1,
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
    logging.info("speech2vadsegment_kwargs: {}".format(speech2vadsegment_kwargs))
    speech2vadsegment = Speech2VadSegment(**speech2vadsegment_kwargs)

    def _forward(
            data_path_and_name_and_type,
            raw_inputs: Union[np.ndarray, torch.Tensor] = None,
            output_dir_v2: Optional[str] = None,
            fs: dict = None,
            param_dict: dict = None
    ):
        # 3. Build data-iterator
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]
        loader = VADTask.build_streaming_iterator(
            data_path_and_name_and_type,
            dtype=dtype,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
            preprocess_fn=VADTask.build_preprocess_fn(speech2vadsegment.vad_infer_args, False),
            collate_fn=VADTask.build_collate_fn(speech2vadsegment.vad_infer_args, False),
            allow_variable_data_keys=allow_variable_data_keys,
            inference=True,
        )

        finish_count = 0
        file_count = 1
        # 7 .Start for-loop
        # FIXME(kamo): The output format should be discussed about
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path is not None:
            writer = DatadirWriter(output_path)
            ibest_writer = writer[f"1best_recog"]
        else:
            writer = None
            ibest_writer = None

        vad_results = []
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"

            # do vad segment
            _, results = speech2vadsegment(**batch)
            for i, _ in enumerate(keys):
                results[i] = json.dumps(results[i])
                item = {'key': keys[i], 'value': results[i]}
                vad_results.append(item)
                if writer is not None:
                    results[i] = json.loads(results[i])
                    ibest_writer["text"][keys[i]] = "{}".format(results[i])

        return vad_results

    return _forward


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="VAD Decoding",
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
    group.add_argument("--raw_inputs", type=list, default=None)
    # example=[{'key':'EdevDEWdIYQ_0021','file':'/mnt/data/jiangyu.xzy/test_data/speech_io/SPEECHIO_ASR_ZH00007_zhibodaihuo/wav/EdevDEWdIYQ_0021.wav'}])
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

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
        "--vad_cmvn_file",
        type=str,
        help="Global cmvn file",
    )

    group = parser.add_argument_group("infer related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
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
