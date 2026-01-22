#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import argparse
import json
import os
from typing import List, Tuple

from omegaconf import OmegaConf

from funasr import AutoModel


def load_jsonl(jsonl_path: str) -> Tuple[List[str], List[str]]:
    keys = []
    targets = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("key")
            if key is None and isinstance(record.get("source"), dict):
                key = record["source"].get("key")
            keys.append(key or "")
            targets.append(record.get("target", ""))
    return keys, targets


def build_model(args: argparse.Namespace):
    kwargs = {}
    if args.config_path and args.config_name:
        cfg_path = os.path.join(args.config_path, args.config_name)
        cfg = OmegaConf.load(cfg_path)
        kwargs.update(OmegaConf.to_container(cfg, resolve=True))
    if args.model:
        kwargs["model"] = args.model
    if args.init_param:
        kwargs["init_param"] = args.init_param
    kwargs["device"] = args.device
    if args.batch_size:
        kwargs["batch_size"] = args.batch_size
    return AutoModel(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="model name or model dir")
    parser.add_argument("--config-path", type=str, default=None, help="config directory")
    parser.add_argument("--config-name", type=str, default=None, help="config filename")
    parser.add_argument("--init-param", type=str, default=None, help="model checkpoint path")
    parser.add_argument("--input-jsonl", type=str, required=True, help="input jsonl with source/target")
    parser.add_argument("--output-dir", type=str, required=True, help="output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size for inference")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    keys, targets = load_jsonl(args.input_jsonl)

    model = build_model(args)
    results = model.generate(input=args.input_jsonl, batch_size=args.batch_size)

    hyp_path = os.path.join(args.output_dir, "text.hyp")
    ref_path = os.path.join(args.output_dir, "text.ref")

    with open(hyp_path, "w", encoding="utf-8") as hyp_f, open(
        ref_path, "w", encoding="utf-8"
    ) as ref_f:
        for idx, result in enumerate(results):
            key = keys[idx] if idx < len(keys) else result.get("key", f"utt_{idx}")
            hyp = result.get("text", "")
            ref = targets[idx] if idx < len(targets) else ""
            hyp_f.write(f"{key} {hyp}\n")
            ref_f.write(f"{key} {ref}\n")

    print(f"hyp saved to: {hyp_path}")
    print(f"ref saved to: {ref_path}")


if __name__ == "__main__":
    main()
