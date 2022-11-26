from collections import defaultdict
import logging
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch.nn.parallel import data_parallel
from torch.utils.data import DataLoader
from typeguard import check_argument_types

from funasr.fileio.datadir_writer import DatadirWriter
from funasr.fileio.npy_scp import NpyScpWriter
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.forward_adaptor import ForwardAdaptor
from funasr.train.abs_espnet_model import AbsESPnetModel


@torch.no_grad()
def collect_stats(
    model: AbsESPnetModel,
    train_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
    valid_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
    output_dir: Path,
    ngpu: Optional[int],
    log_interval: Optional[int],
    write_collected_feats: bool,
) -> None:
    """Perform on collect_stats mode.

    Running for deriving the shape information from data
    and gathering statistics.
    This method is used before executing train().

    """
    assert check_argument_types()

    npy_scp_writers = {}
    for itr, mode in zip([train_iter, valid_iter], ["train", "valid"]):
        if log_interval is None:
            try:
                log_interval = max(len(itr) // 20, 10)
            except TypeError:
                log_interval = 100

        sum_dict = defaultdict(lambda: 0)
        sq_dict = defaultdict(lambda: 0)
        count_dict = defaultdict(lambda: 0)

        with DatadirWriter(output_dir / mode) as datadir_writer:
            for iiter, (keys, batch) in enumerate(itr, 1):
                batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

                # 1. Write shape file
                for name in batch:
                    if name.endswith("_lengths"):
                        continue
                    for i, (key, data) in enumerate(zip(keys, batch[name])):
                        if f"{name}_lengths" in batch:
                            lg = int(batch[f"{name}_lengths"][i])
                            data = data[:lg]
                        datadir_writer[f"{name}_shape"][key] = ",".join(
                            map(str, data.shape)
                        )

                # 2. Extract feats
                if ngpu <= 1:
                    data = model.collect_feats(**batch)
                else:
                    # Note that data_parallel can parallelize only "forward()"
                    data = data_parallel(
                        ForwardAdaptor(model, "collect_feats"),
                        (),
                        range(ngpu),
                        module_kwargs=batch,
                    )

                # 3. Calculate sum and square sum
                for key, v in data.items():
                    for i, (uttid, seq) in enumerate(zip(keys, v.cpu().numpy())):
                        # Truncate zero-padding region
                        if f"{key}_lengths" in data:
                            length = data[f"{key}_lengths"][i]
                            # seq: (Length, Dim, ...)
                            seq = seq[:length]
                        else:
                            # seq: (Dim, ...) -> (1, Dim, ...)
                            seq = seq[None]
                        # Accumulate value, its square, and count
                        sum_dict[key] += seq.sum(0)
                        sq_dict[key] += (seq**2).sum(0)
                        count_dict[key] += len(seq)

                        # 4. [Option] Write derived features as npy format file.
                        if write_collected_feats:
                            # Instantiate NpyScpWriter for the first iteration
                            if (key, mode) not in npy_scp_writers:
                                p = output_dir / mode / "collect_feats"
                                npy_scp_writers[(key, mode)] = NpyScpWriter(
                                    p / f"data_{key}", p / f"{key}.scp"
                                )
                            # Save array as npy file
                            npy_scp_writers[(key, mode)][uttid] = seq

                if iiter % log_interval == 0:
                    logging.info(f"Niter: {iiter}")

        for key in sum_dict:
            np.savez(
                output_dir / mode / f"{key}_stats.npz",
                count=count_dict[key],
                sum=sum_dict[key],
                sum_square=sq_dict[key],
            )

        # batch_keys and stats_keys are used by aggregate_stats_dirs.py
        with (output_dir / mode / "batch_keys").open("w", encoding="utf-8") as f:
            f.write(
                "\n".join(filter(lambda x: not x.endswith("_lengths"), batch)) + "\n"
            )
        with (output_dir / mode / "stats_keys").open("w", encoding="utf-8") as f:
            f.write("\n".join(sum_dict) + "\n")
