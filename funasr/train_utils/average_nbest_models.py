import logging
from pathlib import Path
from typing import Optional
from typing import Sequence
from typing import Union
import warnings
import os
from io import BytesIO

import torch
from typing import Collection
import os
import torch
import re
from collections import OrderedDict
from functools import cmp_to_key


def _get_checkpoint_paths(output_dir: str, last_n: int = 5, use_deepspeed=False, **kwargs):
    """
    Get the paths of the last 'last_n' checkpoints by parsing filenames
    in the output directory.
    """
    try:
        if not use_deepspeed:
            checkpoint = torch.load(os.path.join(output_dir, "model.pt"), map_location="cpu")
        else:
            checkpoint = torch.load(
                os.path.join(output_dir, "model.pt", "mp_rank_00_model_states.pt"),
                map_location="cpu",
            )
        avg_keep_nbest_models_type = checkpoint["avg_keep_nbest_models_type"]
        val_step_or_eoch = checkpoint[f"val_{avg_keep_nbest_models_type}_step_or_eoch"]
        sorted_items = sorted(val_step_or_eoch.items(), key=lambda x: x[1], reverse=True)
        sorted_items = (
            sorted_items[:last_n] if avg_keep_nbest_models_type == "acc" else sorted_items[-last_n:]
        )
        checkpoint_paths = []
        for key, value in sorted_items[:last_n]:
            if not use_deepspeed:
                ckpt = os.path.join(output_dir, key)
            else:
                ckpt = os.path.join(output_dir, key, "mp_rank_00_model_states.pt")
            checkpoint_paths.append(ckpt)

    except:
        print(f"{checkpoint} does not exist, avg the lastet checkpoint.")
        # List all files in the output directory
        files = os.listdir(output_dir)
        # Filter out checkpoint files and extract epoch numbers
        checkpoint_files = [f for f in files if f.startswith("model.pt.e")]
        # Sort files by epoch number in descending order
        checkpoint_files.sort(key=lambda x: int(re.search(r"(\d+)", x).group()), reverse=True)
        # Get the last 'last_n' checkpoint paths
        checkpoint_paths = [os.path.join(output_dir, f) for f in checkpoint_files[:last_n]]
    return checkpoint_paths


@torch.no_grad()
def average_checkpoints(output_dir: str, last_n: int = 5, **kwargs):
    """
    Average the last 'last_n' checkpoints' model state_dicts.
    If a tensor is of type torch.int, perform sum instead of average.
    """
    checkpoint_paths = _get_checkpoint_paths(output_dir, last_n, **kwargs)
    print(f"average_checkpoints: {checkpoint_paths}")
    state_dicts = []

    # Load state_dicts from checkpoints
    for path in checkpoint_paths:
        if os.path.isfile(path):
            state_dicts.append(torch.load(path, map_location="cpu")["state_dict"])
        else:
            print(f"Checkpoint file {path} not found.")

    # Check if we have any state_dicts to average
    if len(state_dicts) < 1:
        print("No checkpoints found for averaging.")
        return

    # Average or sum weights
    avg_state_dict = OrderedDict()
    for key in state_dicts[0].keys():
        tensors = [state_dict[key].cpu() for state_dict in state_dicts]
        # Check the type of the tensor
        if str(tensors[0].dtype).startswith("torch.int"):
            # Perform sum for integer tensors
            summed_tensor = sum(tensors)
            avg_state_dict[key] = summed_tensor
        else:
            # Perform average for other types of tensors
            stacked_tensors = torch.stack(tensors)
            avg_state_dict[key] = torch.mean(stacked_tensors, dim=0)
    checkpoint_outpath = os.path.join(output_dir, f"model.pt.avg{last_n}")
    torch.save({"state_dict": avg_state_dict}, checkpoint_outpath)
    return checkpoint_outpath
