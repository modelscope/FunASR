import logging
from pathlib import Path
from typing import Optional
from typing import Sequence
from typing import Union
import warnings
import os
from io import BytesIO

import torch
from typeguard import check_argument_types
from typing import Collection

from funasr.train.reporter import Reporter


@torch.no_grad()
def average_nbest_models(
    output_dir: Path,
    reporter: Reporter,
    best_model_criterion: Sequence[Sequence[str]],
    nbest: Union[Collection[int], int],
    suffix: Optional[str] = None,
    oss_bucket=None,
    pai_output_dir=None,
) -> None:
    """Generate averaged model from n-best models

    Args:
        output_dir: The directory contains the model file for each epoch
        reporter: Reporter instance
        best_model_criterion: Give criterions to decide the best model.
            e.g. [("valid", "loss", "min"), ("train", "acc", "max")]
        nbest: Number of best model files to be averaged
        suffix: A suffix added to the averaged model file name
    """
    assert check_argument_types()
    if isinstance(nbest, int):
        nbests = [nbest]
    else:
        nbests = list(nbest)
    if len(nbests) == 0:
        warnings.warn("At least 1 nbest values are required")
        nbests = [1]
    if suffix is not None:
        suffix = suffix + "."
    else:
        suffix = ""

    # 1. Get nbests: List[Tuple[str, str, List[Tuple[epoch, value]]]]
    nbest_epochs = [
        (ph, k, reporter.sort_epochs_and_values(ph, k, m)[: max(nbests)])
        for ph, k, m in best_model_criterion
        if reporter.has(ph, k)
    ]

    _loaded = {}
    for ph, cr, epoch_and_values in nbest_epochs:
        _nbests = [i for i in nbests if i <= len(epoch_and_values)]
        if len(_nbests) == 0:
            _nbests = [1]

        for n in _nbests:
            if n == 0:
                continue
            elif n == 1:
                # The averaged model is same as the best model
                e, _ = epoch_and_values[0]
                op = output_dir / f"{e}epoch.pth"
                sym_op = output_dir / f"{ph}.{cr}.ave_1best.{suffix}pth"
                if sym_op.is_symlink() or sym_op.exists():
                    sym_op.unlink()
                sym_op.symlink_to(op.name)
            else:
                op = output_dir / f"{ph}.{cr}.ave_{n}best.{suffix}pth"
                logging.info(
                    f"Averaging {n}best models: " f'criterion="{ph}.{cr}": {op}'
                )

                avg = None
                # 2.a. Averaging model
                for e, _ in epoch_and_values[:n]:
                    if e not in _loaded:
                        if oss_bucket is None:
                            _loaded[e] = torch.load(
                                output_dir / f"{e}epoch.pth",
                                map_location="cpu",
                            )
                        else:
                            buffer = BytesIO(
                                oss_bucket.get_object(os.path.join(pai_output_dir, f"{e}epoch.pth")).read())
                            _loaded[e] = torch.load(buffer)
                    states = _loaded[e]

                    if avg is None:
                        avg = states
                    else:
                        # Accumulated
                        for k in avg:
                            avg[k] = avg[k] + states[k]
                for k in avg:
                    if str(avg[k].dtype).startswith("torch.int"):
                        # For int type, not averaged, but only accumulated.
                        # e.g. BatchNorm.num_batches_tracked
                        # (If there are any cases that requires averaging
                        #  or the other reducing method, e.g. max/min, for integer type,
                        #  please report.)
                        pass
                    else:
                        avg[k] = avg[k] / n

                # 2.b. Save the ave model and create a symlink
                if oss_bucket is None:
                    torch.save(avg, op)
                else:
                    buffer = BytesIO()
                    torch.save(avg, buffer)
                    oss_bucket.put_object(os.path.join(pai_output_dir, f"{ph}.{cr}.ave_{n}best.{suffix}pth"),
                                          buffer.getvalue())

        # 3. *.*.ave.pth is a symlink to the max ave model
        if oss_bucket is None:
            op = output_dir / f"{ph}.{cr}.ave_{max(_nbests)}best.{suffix}pth"
            sym_op = output_dir / f"{ph}.{cr}.ave.{suffix}pth"
            if sym_op.is_symlink() or sym_op.exists():
                sym_op.unlink()
            sym_op.symlink_to(op.name)
