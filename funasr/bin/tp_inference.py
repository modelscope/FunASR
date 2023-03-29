import argparse
import logging
from optparse import Option
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

import numpy as np
import torch
from typeguard import check_argument_types

from funasr.fileio.datadir_writer import DatadirWriter
from funasr.datasets.preprocessor import LMPreprocessor
from funasr.tasks.asr import ASRTaskAligner as ASRTask
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import config_argparse
from funasr.utils.cli_utils import get_commandline_args
from funasr.utils.types import str2bool
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_none
from funasr.models.frontend.wav_frontend import WavFrontend
from funasr.text.token_id_converter import TokenIDConverter
from funasr.utils.timestamp_tools import ts_prediction_lfr6_standard


header_colors = '\033[95m'
end_colors = '\033[0m'

global_asr_language: str = 'zh-cn'
global_sample_rate: Union[int, Dict[Any, int]] = {
    'audio_fs': 16000,
    'model_fs': 16000
}


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
        if 'cuda' in device:
            tp_model = tp_model.cuda()  # force model to cuda

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
            self.tp_model.frontend = None
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
        _, _, us_alphas, us_peaks = self.tp_model.calc_predictor_timestamp(enc, enc_len, text_lengths.to(self.device)+1)
        return us_alphas, us_peaks


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
        split_with_space: bool = True,
        seg_dict_file: Optional[str] = None,
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
        split_with_space=split_with_space,
        seg_dict_file=seg_dict_file,
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
        split_with_space: bool = True,
        seg_dict_file: Optional[str] = None,
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

    preprocessor = LMPreprocessor(
        train=False,
        token_type=speechtext2timestamp.tp_train_args.token_type,
        token_list=speechtext2timestamp.tp_train_args.token_list,
        bpemodel=None,
        text_cleaner=None,
        g2p_type=None,
        text_name="text",
        non_linguistic_symbols=speechtext2timestamp.tp_train_args.non_linguistic_symbols,
        split_with_space=split_with_space,
        seg_dict_file=seg_dict_file,
    )
    
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
            preprocess_fn=preprocessor,
            collate_fn=ASRTask.build_collate_fn(speechtext2timestamp.tp_train_args, False),
            allow_variable_data_keys=allow_variable_data_keys,
            inference=True,
        )

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
                ts_str, ts_list = ts_prediction_lfr6_standard(us_alphas[batch_id], us_cif_peak[batch_id], token, force_time_shift=-3.0)
                logging.warning(ts_str)
                item = {'key': key, 'value': ts_str, 'timestamp':ts_list}
                tp_result_list.append(item)
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
    group.add_argument(
        "--seg_dict_file",
        type=str,
        default=None,
        help="The batch size for inference",
    )
    group.add_argument(
        "--split_with_space",
        type=bool,
        default=False,
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
