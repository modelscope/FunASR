from typing import Any
from typing import Dict
from typing import Union
from io import BytesIO

import logging
import torch
import torch.nn
import torch.optim
import pdb


def filter_state_dict(
    dst_state: Dict[str, Union[float, torch.Tensor]],
    src_state: Dict[str, Union[float, torch.Tensor]],
):
    """Filter name, size mismatch instances between dicts.

    Args:
            dst_state: reference state dict for filtering
            src_state: target state dict for filtering

    """
    match_state = {}
    for key, value in src_state.items():
        if key in dst_state and (dst_state[key].size() == src_state[key].size()):
            match_state[key] = value
        else:
            if key not in dst_state:
                logging.warning(
                    f"Filter out {key} from pretrained dict"
                    + " because of name not found in target dict"
                )
            else:
                logging.warning(
                    f"Filter out {key} from pretrained dict"
                    + " because of size mismatch"
                    + f"({dst_state[key].size()}-{src_state[key].size()})"
                )
    return match_state


def load_pretrained_model(
    path: str,
    model: torch.nn.Module,
    ignore_init_mismatch: bool = True,
    map_location: str = "cpu",
    oss_bucket=None,
    scope_map=[],
    excludes=None,
    **kwargs,
):
    """Load a model state and set it to the model.

    Args:
            init_param: <file_path>:<src_key>:<dst_key>:<exclude_Keys>

    Examples:

    """

    obj = model
    dst_state = obj.state_dict()

    print(f"ckpt: {path}")

    if oss_bucket is None:
        src_state = torch.load(path, map_location=map_location)
    else:
        buffer = BytesIO(oss_bucket.get_object(path).read())
        src_state = torch.load(buffer, map_location=map_location)

    src_state = src_state["state_dict"] if "state_dict" in src_state else src_state
    src_state = src_state["model_state_dict"] if "model_state_dict" in src_state else src_state
    src_state = src_state["model"] if "model" in src_state else src_state

    if isinstance(scope_map, str):
        scope_map = scope_map.split(",")
    scope_map += ["module.", "None"]

    for k in dst_state.keys():

        k_src = k

        if scope_map is not None:
            src_prefix = ""
            dst_prefix = ""
            for i in range(0, len(scope_map), 2):
                src_prefix = scope_map[i] if scope_map[i].lower() != "none" else ""
                dst_prefix = scope_map[i + 1] if scope_map[i + 1].lower() != "none" else ""

                if dst_prefix == "" and (src_prefix + k) in src_state.keys():
                    k_src = src_prefix + k
                    if not k_src.startswith("module."):
                        print(f"init param, map: {k} from {k_src} in ckpt")
                elif (
                    k.startswith(dst_prefix)
                    and k.replace(dst_prefix, src_prefix, 1) in src_state.keys()
                ):
                    k_src = k.replace(dst_prefix, src_prefix, 1)
                    if not k_src.startswith("module."):
                        print(f"init param, map: {k} from {k_src} in ckpt")

        if k_src in src_state.keys():
            if ignore_init_mismatch and dst_state[k].shape != src_state[k_src].shape:
                print(
                    f"ignore_init_mismatch:{ignore_init_mismatch}, dst: {k, dst_state[k].shape}, src: {k_src, src_state[k_src].shape}"
                )
            else:
                dst_state[k] = src_state[k_src]

        else:
            print(f"Warning, miss key in ckpt: {k}, mapped: {k_src}")

    flag = obj.load_state_dict(dst_state, strict=True)
    # print(flag)
