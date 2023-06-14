import argparse
import logging
import os
from pathlib import Path
from typing import Union

import torch
import yaml
from typeguard import check_argument_types

from funasr.build_utils.build_model import build_model
from funasr.models.base_model import FunASRModel


def build_model_from_file(
        config_file: Union[Path, str] = None,
        model_file: Union[Path, str] = None,
        cmvn_file: Union[Path, str] = None,
        device: str = "cpu",
        mode: str = "paraformer",
):
    """Build model from the files.

    This method is used for inference or fine-tuning.

    Args:
        config_file: The yaml file saved when training.
        model_file: The model file saved when training.
        device: Device type, "cpu", "cuda", or "cuda:N".

    """
    assert check_argument_types()
    if config_file is None:
        assert model_file is not None, (
            "The argument 'model_file' must be provided "
            "if the argument 'config_file' is not specified."
        )
        config_file = Path(model_file).parent / "config.yaml"
    else:
        config_file = Path(config_file)

    with config_file.open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    if cmvn_file is not None:
        args["cmvn_file"] = cmvn_file
    args = argparse.Namespace(**args)
    model = build_model(args)
    if not isinstance(model, FunASRModel):
        raise RuntimeError(
            f"model must inherit {FunASRModel.__name__}, but got {type(model)}"
        )
    model.to(device)
    model_dict = dict()
    model_name_pth = None
    if model_file is not None:
        logging.info("model_file is {}".format(model_file))
        if device == "cuda":
            device = f"cuda:{torch.cuda.current_device()}"
        model_dir = os.path.dirname(model_file)
        model_name = os.path.basename(model_file)
        if "model.ckpt-" in model_name or ".bin" in model_name:
            model_name_pth = os.path.join(model_dir, model_name.replace('.bin',
                                                                        '.pb')) if ".bin" in model_name else os.path.join(
                model_dir, "{}.pb".format(model_name))
            if os.path.exists(model_name_pth):
                logging.info("model_file is load from pth: {}".format(model_name_pth))
                model_dict = torch.load(model_name_pth, map_location=device)
            else:
                model_dict = convert_tf2torch(model, model_file, mode)
            model.load_state_dict(model_dict)
        else:
            model_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(model_dict)
    if model_name_pth is not None and not os.path.exists(model_name_pth):
        torch.save(model_dict, model_name_pth)
        logging.info("model_file is saved to pth: {}".format(model_name_pth))

    return model, args


def convert_tf2torch(
        model,
        ckpt,
        mode,
):
    assert mode == "paraformer" or mode == "uniasr"
    logging.info("start convert tf model to torch model")
    from funasr.modules.streaming_utils.load_fr_tf import load_tf_dict
    var_dict_tf = load_tf_dict(ckpt)
    var_dict_torch = model.state_dict()
    var_dict_torch_update = dict()
    if mode == "uniasr":
        # encoder
        var_dict_torch_update_local = model.encoder.convert_tf2torch(var_dict_tf, var_dict_torch)
        var_dict_torch_update.update(var_dict_torch_update_local)
        # predictor
        var_dict_torch_update_local = model.predictor.convert_tf2torch(var_dict_tf, var_dict_torch)
        var_dict_torch_update.update(var_dict_torch_update_local)
        # decoder
        var_dict_torch_update_local = model.decoder.convert_tf2torch(var_dict_tf, var_dict_torch)
        var_dict_torch_update.update(var_dict_torch_update_local)
        # encoder2
        var_dict_torch_update_local = model.encoder2.convert_tf2torch(var_dict_tf, var_dict_torch)
        var_dict_torch_update.update(var_dict_torch_update_local)
        # predictor2
        var_dict_torch_update_local = model.predictor2.convert_tf2torch(var_dict_tf, var_dict_torch)
        var_dict_torch_update.update(var_dict_torch_update_local)
        # decoder2
        var_dict_torch_update_local = model.decoder2.convert_tf2torch(var_dict_tf, var_dict_torch)
        var_dict_torch_update.update(var_dict_torch_update_local)
        # stride_conv
        var_dict_torch_update_local = model.stride_conv.convert_tf2torch(var_dict_tf, var_dict_torch)
        var_dict_torch_update.update(var_dict_torch_update_local)
    else:
        # encoder
        var_dict_torch_update_local = model.encoder.convert_tf2torch(var_dict_tf, var_dict_torch)
        var_dict_torch_update.update(var_dict_torch_update_local)
        # predictor
        var_dict_torch_update_local = model.predictor.convert_tf2torch(var_dict_tf, var_dict_torch)
        var_dict_torch_update.update(var_dict_torch_update_local)
        # decoder
        var_dict_torch_update_local = model.decoder.convert_tf2torch(var_dict_tf, var_dict_torch)
        var_dict_torch_update.update(var_dict_torch_update_local)
        # bias_encoder
        var_dict_torch_update_local = model.clas_convert_tf2torch(var_dict_tf, var_dict_torch)
        var_dict_torch_update.update(var_dict_torch_update_local)

    return var_dict_torch_update
