import yaml
import argparse

def update_dct(fin_configs, root):
    if root == {}:
        return {}
    for root_key, root_value  in root.items():
        if not isinstance(root[root_key],dict):
            fin_configs[root_key] = root[root_key]
        else:
            result = update_dct(fin_configs[root_key], root[root_key])
            fin_configs[root_key] = result
    return fin_configs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="update configs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--modelscope_config",
                        type=str,
                        help="modelscope config file")
    parser.add_argument("--finetune_config",
                        type=str,
                        help="finetune config file")
    parser.add_argument("--output_config",
                        type=str,
                        help="output config file")
    args = parser.parse_args()

    with open(args.modelscope_config) as f:
        modelscope_configs = yaml.safe_load(f)

    with open(args.finetune_config) as f:
        finetune_configs = yaml.safe_load(f)

    # update configs, e.g., lr, batch_size, ...
    modelscope_configs = update_dct(modelscope_configs, finetune_configs)

    with open(args.output_config, "w") as f:
        yaml.dump(modelscope_configs, f, indent=4)
