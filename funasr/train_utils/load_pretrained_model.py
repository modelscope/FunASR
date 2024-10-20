from typing import Any
from typing import Dict
from typing import Union
from io import BytesIO

import logging
import torch
import torch.nn
import torch.optim
import pdb
import copy


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

    logging.info(f"ckpt: {path}")

    if oss_bucket is None:
        ori_state = torch.load(path, map_location=map_location)
    else:
        buffer = BytesIO(oss_bucket.get_object(path).read())
        ori_state = torch.load(buffer, map_location=map_location)

    src_state = copy.deepcopy(ori_state)
    src_state = src_state["state_dict"] if "state_dict" in src_state else src_state
    src_state = src_state["model_state_dict"] if "model_state_dict" in src_state else src_state
    src_state = src_state["model"] if "model" in src_state else src_state

    if isinstance(scope_map, str):
        scope_map = scope_map.split(",")
    scope_map += ["module.", "None"]
    logging.info(f"scope_map: {scope_map}")

    if excludes is not None:
        if isinstance(excludes, str):
            excludes = excludes.split(",")

    logging.info(f"excludes: {excludes}")

    for k in dst_state.keys():
        excludes_flag = False
        if excludes is not None:
            for k_ex in excludes:
                if k.startswith(k_ex):
                    logging.info(f"key: {k} matching: {k_ex}, excluded")
                    excludes_flag = True
                    break
        if excludes_flag:
            continue

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
                        logging.info(f"init param, map: {k} from {k_src} in ckpt")
                elif (
                    k.startswith(dst_prefix)
                    and k.replace(dst_prefix, src_prefix, 1) in src_state.keys()
                ):
                    k_src = k.replace(dst_prefix, src_prefix, 1)
                    if not k_src.startswith("module."):
                        logging.info(f"init param, map: {k} from {k_src} in ckpt")

        if k_src in src_state.keys():
            if ignore_init_mismatch and dst_state[k].shape != src_state[k_src].shape:
                logging.info(
                    f"ignore_init_mismatch:{ignore_init_mismatch}, dst: {k, dst_state[k].shape}, src: {k_src, src_state[k_src].shape}"
                )
            else:
                dst_state[k] = src_state[k_src]
        else:
            print(f"Warning, miss key in ckpt: {k}, {path}")

    flag = obj.load_state_dict(dst_state, strict=True)
    logging.info(f"Loading ckpt: {path}, status: {flag}")
