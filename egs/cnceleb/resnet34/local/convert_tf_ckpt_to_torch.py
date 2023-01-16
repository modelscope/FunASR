import numpy as np
import logging
from typing import Dict
import sys
import torch


def load_ckpt(checkpoint_path: str) -> Dict[str, np.ndarray]:
    from tensorflow.python import pywrap_tensorflow

    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    var_dict = dict()
    for var_name in sorted(var_to_shape_map):
        if "optimizer" in var_name:
            continue
        tensor = reader.get_tensor(var_name)
        print("in ckpt: {}, {}".format(var_name, tensor.shape))  # print(tensor)
        var_dict[var_name] = tensor

    return var_dict


def convert_parameter_name_for_asv_resnet34(
        var_dict: Dict[str, np.ndarray],
        old_prefix: str = "EAND/speech_encoder",
        new_prefix: str = "encoder",
        train_steps: int = 0
) -> Dict[str, np.ndarray]:
    new_dict = dict()
    model_size = 0
    for name, tensor in var_dict.items():
        if not name.startswith(old_prefix):
            if name == "softmax/output/kernel":
                new_name = "decoder.output_dense.weight"
                tensor = np.transpose(tensor, [1, 0])
                new_dict[new_name] = torch.Tensor(tensor)
            continue
        new_name = name.replace(old_prefix, new_prefix)
        new_name = new_name.replace("/", ".")
        if "resnet1" in new_name or "resnet2" in new_name:
            new_name = new_name.replace("encoder", "decoder")
        module_name, para_name = new_name.rsplit(".", 1)
        # process for batch normalization
        if "bn" in module_name:
            new_name = new_name.replace("gamma", "weight")
            new_name = new_name.replace("beta", "bias")
            new_name = new_name.replace("moving_mean", "running_mean")
            new_name = new_name.replace("moving_variance", "running_var")

            new_dict[new_name] = torch.Tensor(tensor)
            new_dict[module_name + ".num_batches_tracked"] = torch.Tensor(train_steps)

        # process for dense layers
        elif "dense" in module_name:
            new_name = new_name.replace("kernel", "weight")
            if para_name == "kernel":
                if len(tensor.shape) == 2:
                    tensor = np.transpose(tensor, [1, 0])
                elif len(tensor.shape) == 3:
                    tensor = np.transpose(tensor, [2, 1, 0])
                # for dense0
                elif len(tensor.shape) == 4:
                    tensor = np.transpose(tensor, [3, 2, 0, 1])

            new_dict[new_name] = torch.Tensor(tensor)

        # process for conv layers
        elif "conv" in module_name:
            new_name = new_name.replace("kernel", "weight")
            if para_name == "kernel":
                tensor = np.transpose(tensor, [3, 2, 0, 1])

            new_dict[new_name] = torch.Tensor(tensor)

        print("{} -> {}".format(name, new_name))
        model_size += new_dict[new_name].numel()
    print("Model size: {}".format(model_size))
    return new_dict


if __name__ == '__main__':
    checkpoint_path = sys.argv[1]
    pkl_path = sys.argv[2]
    tf_dict = load_ckpt(checkpoint_path)
    torch_dict = convert_parameter_name_for_asv_resnet34(
        tf_dict,
        train_steps=300000,
    )
    torch.save(
        torch_dict, pkl_path
    )
