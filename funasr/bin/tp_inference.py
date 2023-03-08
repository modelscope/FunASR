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
from funasr.tasks.asr import ASRTaskAligner as ASRTask
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import config_argparse
from funasr.utils.cli_utils import get_commandline_args
from funasr.utils.types import str2bool
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_none
from funasr.utils import asr_utils, wav_utils, postprocess_utils
from funasr.models.frontend.wav_frontend import WavFrontend
from funasr.text.token_id_converter import TokenIDConverter

header_colors = '\033[95m'
end_colors = '\033[0m'

global_asr_language: str = 'zh-cn'
global_sample_rate: Union[int, Dict[Any, int]] = {
    'audio_fs': 16000,
    'model_fs': 16000
}

def time_stamp_lfr6_advance(us_alphas, us_cif_peak, char_list):
    START_END_THRESHOLD = 5
    MAX_TOKEN_DURATION = 12
    TIME_RATE = 10.0 * 6 / 1000 / 3  #  3 times upsampled
    if len(us_cif_peak.shape) == 2:
        alphas, cif_peak = us_alphas[0], us_cif_peak[0]  # support inference batch_size=1 only
    else:
        alphas, cif_peak = us_alphas, us_cif_peak
    num_frames = cif_peak.shape[0]
    if char_list[-1] == '</s>':
        char_list = char_list[:-1]
    # char_list = [i for i in text]
    timestamp_list = []
    new_char_list = []
    # for bicif model trained with large data, cif2 actually fires when a character starts
    # so treat the frames between two peaks as the duration of the former token
    fire_place = torch.where(cif_peak>1.0-1e-4)[0].cpu().numpy() - 3.2  # total offset
    num_peak = len(fire_place)
    assert num_peak == len(char_list) + 1 # number of peaks is supposed to be number of tokens + 1
    # begin silence
    if fire_place[0] > START_END_THRESHOLD:
        # char_list.insert(0, '<sil>')
        timestamp_list.append([0.0, fire_place[0]*TIME_RATE])
        new_char_list.append('<sil>')
    # tokens timestamp
    for i in range(len(fire_place)-1):
        new_char_list.append(char_list[i])
        if MAX_TOKEN_DURATION < 0 or fire_place[i+1] - fire_place[i] < MAX_TOKEN_DURATION:
            timestamp_list.append([fire_place[i]*TIME_RATE, fire_place[i+1]*TIME_RATE])
        else:
            # cut the duration to token and sil of the 0-weight frames last long
            _split = fire_place[i] + MAX_TOKEN_DURATION
            timestamp_list.append([fire_place[i]*TIME_RATE, _split*TIME_RATE])
            timestamp_list.append([_split*TIME_RATE, fire_place[i+1]*TIME_RATE])
            new_char_list.append('<sil>')
    # tail token and end silence
    # new_char_list.append(char_list[-1])
    if num_frames - fire_place[-1] > START_END_THRESHOLD:
        _end = (num_frames + fire_place[-1]) * 0.5
        # _end = fire_place[-1] 
        timestamp_list[-1][1] = _end*TIME_RATE
        timestamp_list.append([_end*TIME_RATE, num_frames*TIME_RATE])
        new_char_list.append("<sil>")
    else:
        timestamp_list[-1][1] = num_frames*TIME_RATE
    assert len(new_char_list) == len(timestamp_list)
    res_str = ""
    for char, timestamp in zip(new_char_list, timestamp_list):
        res_str += "{} {} {};".format(char, str(timestamp[0]+0.0005)[:5], str(timestamp[1]+0.0005)[:5])
    res = []
    for char, timestamp in zip(char_list, timestamp_list):
        if char != '<sil>':
            res.append([int(timestamp[0] * 1000), int(timestamp[1] * 1000)])
    return res_str, res


class SpeechText2Timestamp:
    def __init__(
        self,
        timestamp_infer_config: Union[Path, str] = None,
        timestamp_model_file: Union[Path, str] = None,
        timestamp_cmvn_file: Union[Path, str] = None,
        device: str = "cpu",
        dtype: str = "float32",
        **kwargs,
    ):
        assert check_argument_types()
        # 1. Build ASR model
        tp_model, tp_train_args = ASRTask.build_model_from_file(
            timestamp_infer_config, timestamp_model_file, device
        )
        frontend = None
        if tp_train_args.frontend is not None:
            frontend = WavFrontend(cmvn_file=timestamp_cmvn_file, **tp_train_args.frontend_conf)
        
        logging.info("tp_model: {}".format(tp_model))
        logging.info("tp_train_args: {}".format(tp_train_args))
        tp_model.to(dtype=getattr(torch, dtype)).eval()

        logging.info(f"Decoding device={device}, dtype={dtype}")


        self.tp_model = tp_model
        self.tp_train_args = tp_train_args

        token_list = self.tp_model.token_list
        self.converter = TokenIDConverter(token_list=token_list)

        self.device = device
        self.dtype = dtype
        self.frontend = frontend
        self.encoder_downsampling_factor = 1
        if tp_train_args.encoder_conf["input_layer"] == "conv2d":
            self.encoder_downsampling_factor = 4
    
    @torch.no_grad()
    def __call__(
        self, 
        speech: Union[torch.Tensor, np.ndarray], 
        speech_lengths: Union[torch.Tensor, np.ndarray] = None, 
        text_lengths: Union[torch.Tensor, np.ndarray] = None
    ):
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        if self.frontend is not None:
            feats, feats_len = self.frontend.forward(speech, speech_lengths)
            feats = to_device(feats, device=self.device)
            feats_len = feats_len.int()
        else:
            feats = speech
            feats_len = speech_lengths

        # lfr_factor = max(1, (feats.size()[-1]//80)-1)
        batch = {"speech": feats, "speech_lengths": feats_len}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, enc_len = self.tp_model.encode(**batch)
        if isinstance(enc, tuple):
            enc = enc[0]

        # c. Forward Predictor
        _, _, us_alphas, us_cif_peak = self.tp_model.calc_predictor_timestamp(enc, enc_len, text_lengths.to(self.device)+1)
        return us_alphas, us_cif_peak


def inference(
        batch_size: int,
        ngpu: int,
        log_level: Union[int, str],
        data_path_and_name_and_type,
        timestamp_infer_config: Optional[str],
        timestamp_model_file: Optional[str],
        timestamp_cmvn_file: Optional[str] = None,
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
        timestamp_infer_config=timestamp_infer_config,
        timestamp_model_file=timestamp_model_file,
        timestamp_cmvn_file=timestamp_cmvn_file,
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
        timestamp_infer_config: Optional[str],
        timestamp_model_file: Optional[str],
        timestamp_cmvn_file: Optional[str] = None,
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
    speechtext2timestamp_kwargs = dict(
        timestamp_infer_config=timestamp_infer_config,
        timestamp_model_file=timestamp_model_file,
        timestamp_cmvn_file=timestamp_cmvn_file,
        device=device,
        dtype=dtype,
    )
    logging.info("speechtext2timestamp_kwargs: {}".format(speechtext2timestamp_kwargs))
    speechtext2timestamp = SpeechText2Timestamp(**speechtext2timestamp_kwargs)
    
    def _forward(
            data_path_and_name_and_type,
            raw_inputs: Union[np.ndarray, torch.Tensor] = None,
            output_dir_v2: Optional[str] = None,
            fs: dict = None,
            param_dict: dict = None,
            **kwargs
    ):
        # 3. Build data-iterator
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]
        
        loader = ASRTask.build_streaming_iterator(
            data_path_and_name_and_type,
            dtype=dtype,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
            preprocess_fn=ASRTask.build_preprocess_fn(speechtext2timestamp.tp_train_args, False),
            collate_fn=ASRTask.build_collate_fn(speechtext2timestamp.tp_train_args, False),
            allow_variable_data_keys=allow_variable_data_keys,
            inference=True,
        )

        finish_count = 0
        file_count = 1

        tp_result_list = []
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"

            logging.info("timestamp predicting, utt_id: {}".format(keys))
            _batch = {'speech':batch['speech'], 
                      'speech_lengths':batch['speech_lengths'],
                      'text_lengths':batch['text_lengths']}
            us_alphas, us_cif_peak = speechtext2timestamp(**_batch)

            for batch_id in range(_bs):
                key = keys[batch_id]
                token = speechtext2timestamp.converter.ids2tokens(batch['text'][batch_id])
                ts_str, ts_list = time_stamp_lfr6_advance(us_alphas[batch_id], us_cif_peak[batch_id], token)
                logging.warning(ts_str)
                tp_result_list.append({'text':"".join([i for i in token if i != '<sil>']), 'timestamp': ts_list})
        return tp_result_list

    return _forward


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Timestamp Prediction Inference",
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
        default=0,
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
        "--timestamp_infer_config",
        type=str,
        help="VAD infer configuration",
    )
    group.add_argument(
        "--timestamp_model_file",
        type=str,
        help="VAD model parameter file",
    )
    group.add_argument(
        "--timestamp_cmvn_file",
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
