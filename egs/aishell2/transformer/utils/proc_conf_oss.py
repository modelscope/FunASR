from pathlib import Path

import torch
import yaml


class NoAliasSafeDumper(yaml.SafeDumper):
    # Disable anchor/alias in yaml because looks ugly
    def ignore_aliases(self, data):
        return True


def yaml_no_alias_safe_dump(data, stream=None, **kwargs):
    """Safe-dump in yaml with no anchor/alias"""
    return yaml.dump(
        data, stream, allow_unicode=True, Dumper=NoAliasSafeDumper, **kwargs
    )


def gen_conf(file, out_dir):
    conf = torch.load(file)["config"]
    conf["oss_bucket"] = "null"
    print(conf)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml_no_alias_safe_dump(conf, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    import sys

    in_f = sys.argv[1]
    out_f = sys.argv[2]
    gen_conf(in_f, out_f)
