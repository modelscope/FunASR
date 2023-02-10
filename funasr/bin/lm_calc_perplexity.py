#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch.nn.parallel import data_parallel
from typeguard import check_argument_types

from funasr.utils.cli_utils import get_commandline_args
from funasr.fileio.datadir_writer import DatadirWriter
from funasr.tasks.lm import LMTask
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.forward_adaptor import ForwardAdaptor
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import config_argparse
from funasr.utils.types import float_or_none
from funasr.utils.types import str2bool
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_none


def calc_perplexity(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    log_base: Optional[float],
    allow_variable_data_keys: bool,
):
    assert check_argument_types()
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

    # 2. Build LM
    model, train_args = LMTask.build_model_from_file(config_file=train_config, model_file=model_file, device=device)
    # Wrape model to make model.nll() data-parallel
    wrapped_model = ForwardAdaptor(model, "nll")
    wrapped_model.to(dtype=getattr(torch, dtype)).eval()
    logging.info(f"Model:\n{model}")

    # 3. Build data-iterator
    loader = LMTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=LMTask.build_preprocess_fn(train_args, False),
        collate_fn=LMTask.build_collate_fn(train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 4. Start for-loop
    with DatadirWriter(output_dir) as writer:
        total_nll = 0.0
        total_ntokens = 0
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"

            with torch.no_grad():
                batch = to_device(batch, device)
                if ngpu <= 1:
                    # NOTE(kamo): data_parallel also should work with ngpu=1,
                    # but for debuggability it's better to keep this block.
                    nll, lengths = wrapped_model(**batch)
                else:
                    nll, lengths = data_parallel(
                        wrapped_model, (), range(ngpu), module_kwargs=batch
                    )

            assert _bs == len(nll) == len(lengths), (_bs, len(nll), len(lengths))
            # nll: (B, L) -> (B,)
            nll = nll.detach().cpu().numpy().sum(1)
            # lengths: (B,)
            lengths = lengths.detach().cpu().numpy()
            total_nll += nll.sum()
            total_ntokens += lengths.sum()

            for key, _nll, ntoken in zip(keys, nll, lengths):
                if log_base is None:
                    utt_ppl = np.exp(_nll / ntoken)
                else:
                    utt_ppl = log_base ** (_nll / ntoken / np.log(log_base))

                # Write PPL of each utts for debugging or analysis
                writer["utt2nll"][key] = str(-_nll)
                writer["utt2ppl"][key] = str(utt_ppl)
                writer["utt2ntokens"][key] = str(ntoken)

        if log_base is None:
            ppl = np.exp(total_nll / total_ntokens)
        else:
            ppl = log_base ** (total_nll / total_ntokens / np.log(log_base))

        with (Path(output_dir) / "ppl").open("w", encoding="utf-8") as f:
            f.write(f"{ppl}\n")
        with (Path(output_dir) / "base").open("w", encoding="utf-8") as f:
            if log_base is None:
                _log_base = np.e
            else:
                _log_base = log_base
            f.write(f"{_log_base}\n")
        logging.info(f"PPL={ppl}")


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Calc perplexity",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    parser.add_argument(
        "--log_base",
        type=float_or_none,
        default=None,
        help="The base of logarithm for Perplexity. "
        "If None, napier's constant is used.",
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
    group.add_argument("--train_config", type=str)
    group.add_argument("--model_file", type=str)

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    calc_perplexity(**kwargs)


if __name__ == "__main__":
    main()
