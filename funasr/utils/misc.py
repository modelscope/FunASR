import os
import io
import shutil
import logging
from collections import OrderedDict
import numpy as np
from omegaconf import DictConfig, OmegaConf


def statistic_model_parameters(model, prefix=None):
    """Statistic model parameters.
    
        Args:
            model: Model instance or model name.
            prefix: TODO.
        """
    var_dict = model.state_dict()
    numel = 0
    for i, key in enumerate(
        sorted(list([x for x in var_dict.keys() if "num_batches_tracked" not in x]))
    ):
        if prefix is None or key.startswith(prefix):
            numel += var_dict[key].numel()
    return numel


def int2vec(x, vec_dim=8, dtype=np.int32):
    """Int2vec.
    
        Args:
            x: TODO.
            vec_dim: Size/dimension parameter.
            dtype: TODO.
        """
    b = ("{:0" + str(vec_dim) + "b}").format(x)
    # little-endian order: lower bit first
    return (np.array(list(b)[::-1]) == "1").astype(dtype)


def seq2arr(seq, vec_dim=8):
    """Seq2arr.
    
        Args:
            seq: TODO.
            vec_dim: Size/dimension parameter.
        """
    return np.row_stack([int2vec(int(x), vec_dim) for x in seq])


def load_scp_as_dict(scp_path, value_type="str", kv_sep=" "):
    """Load scp as dict.
    
        Args:
            scp_path: TODO.
            value_type: TODO.
            kv_sep: TODO.
        """
    with io.open(scp_path, "r", encoding="utf-8") as f:
        ret_dict = OrderedDict()
        for one_line in f.readlines():
            one_line = one_line.strip()
            pos = one_line.find(kv_sep)
            key, value = one_line[:pos], one_line[pos + 1 :]
            if value_type == "list":
                value = value.split(" ")
            ret_dict[key] = value
        return ret_dict


def load_scp_as_list(scp_path, value_type="str", kv_sep=" "):
    """Load scp as list.
    
        Args:
            scp_path: TODO.
            value_type: TODO.
            kv_sep: TODO.
        """
    with io.open(scp_path, "r", encoding="utf8") as f:
        ret_dict = []
        for one_line in f.readlines():
            one_line = one_line.strip()
            pos = one_line.find(kv_sep)
            key, value = one_line[:pos], one_line[pos + 1 :]
            if value_type == "list":
                value = value.split(" ")
            ret_dict.append((key, value))
        return ret_dict


def deep_update(original, update):

    """Recursively merge update dict into original dict (in-place).

    For nested dicts, merges recursively. For other types, overwrites.

    Args:
        original (dict): Target dict to be updated in-place.
        update (dict): Source dict with new values.
    """
    for key, value in update.items():
        if isinstance(value, dict) and key in original:
            if len(value) == 0:
                original[key] = value
            deep_update(original[key], value)
        else:
            original[key] = value


def prepare_model_dir(**kwargs):

    """Prepare model dir.
    
        Args:
            **kwargs: Additional keyword arguments.
        """
    os.makedirs(kwargs.get("output_dir", "./"), exist_ok=True)

    yaml_file = os.path.join(kwargs.get("output_dir", "./"), "config.yaml")
    OmegaConf.save(config=kwargs, f=yaml_file)
    logging.info(f"kwargs: {kwargs}")
    logging.info("config.yaml is saved to: %s", yaml_file)

    model_path = kwargs.get("model_path", None)
    if model_path is not None:
        config_json = os.path.join(model_path, "configuration.json")
        if os.path.exists(config_json):
            shutil.copy(
                config_json, os.path.join(kwargs.get("output_dir", "./"), "configuration.json")
            )


def extract_filename_without_extension(file_path):
    """
    从给定的文件路径中提取文件名（不包含路径和扩展名）
    :param file_path: 完整的文件路径
    :return: 文件名（不含路径和扩展名）
    """
    # 首先，使用os.path.basename获取路径中的文件名部分（含扩展名）
    filename_with_extension = os.path.basename(file_path)
    # 然后，使用os.path.splitext分离文件名和扩展名
    filename, extension = os.path.splitext(filename_with_extension)
    # 返回不包含扩展名的文件名
    return filename


def smart_remove(path):
    """Intelligently removes files, empty directories, and non-empty directories recursively."""
    # Check if the provided path exists
    if not os.path.exists(path):
        print(f"{path} does not exist.")
        return

    # If the path is a file, delete it
    if os.path.isfile(path):
        os.remove(path)
        print(f"File {path} has been deleted.")
    # If the path is a directory
    elif os.path.isdir(path):
        try:
            # Attempt to remove an empty directory
            os.rmdir(path)
            print(f"Empty directory {path} has been deleted.")
        except OSError:
            # If the directory is not empty, remove it along with all its contents
            shutil.rmtree(path)
            print(f"Non-empty directory {path} has been recursively deleted.")
