#!/usr/bin/env python3

import argparse

import torch


def average_model(input_files, output_file):
    output_model = {}
    for ckpt_path in input_files:
        model_params = torch.load(ckpt_path, map_location="cpu")
        for key, value in model_params.items():
            if key not in output_model:
                output_model[key] = value
            else:
                output_model[key] += value
    for key in output_model.keys():
        output_model[key] /= len(input_files)
    torch.save(output_model, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file")
    parser.add_argument("input_files", nargs='+')
    args = parser.parse_args()

    average_model(args.input_files, args.output_file)