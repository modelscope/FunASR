from __future__ import print_function

import argparse
import copy
import logging
import os
from shutil import copyfile

import torch
import yaml
from typing import Union


from funasr.models.fsmn_kws.model import FsmnKWSConvert


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_args():
    parser = argparse.ArgumentParser(
        description=
        'load and convert network to each other between kaldi/pytorch format')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument(
        '--network_file',
        default='',
        required=True,
        help='input network, support kaldi.txt/pytorch.pt')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--model_name', required=True, help='save model name')
    parser.add_argument('--convert_to',
                        default='kaldi',
                        required=True,
                        help='target network type, kaldi/pytorch')

    args = parser.parse_args()
    return args


def convert_to_kaldi(
    configs,
    network_file,
    model_dir,
    model_name="convert.kaldi.txt"
):
    copyfile(network_file, os.path.join(model_dir, 'origin.torch.pt'))

    model = FsmnKWSConvert(
            vocab_size=configs['encoder_conf']['output_dim'],
            encoder='FSMNConvert',
            encoder_conf=configs['encoder_conf'],
            ctc_conf=configs['ctc_conf'],
    )
    print(model)
    num_params = count_parameters(model)
    print('the number of model params: {}'.format(num_params))

    states= torch.load(network_file, map_location='cpu')
    model.load_state_dict(states["state_dict"])

    kaldi_text = os.path.join(model_dir, model_name)
    with open(kaldi_text, 'w', encoding='utf8') as fout:
        nnet_desp = model.to_kaldi_net()
        fout.write(nnet_desp)
    fout.close()


def convert_to_pytorch(
    configs,
    network_file,
    model_dir,
    model_name="convert.torch.pt"
):
    model = FsmnKWSConvert(
            vocab_size=configs['encoder_conf']['output_dim'],
            frontend=None,
            specaug=None,
            normalize=None,
            encoder='FSMNConvert',
            encoder_conf=configs['encoder_conf'],
            ctc_conf=configs['ctc_conf'],
    )

    num_params = count_parameters(model)
    print('the number of model params: {}'.format(num_params))

    copyfile(network_file, os.path.join(model_dir, 'origin.kaldi.txt'))
    model.to_pytorch_net(network_file)

    save_model_path = os.path.join(model_dir, model_name)
    torch.save({"model": model.state_dict()}, save_model_path)

    print('convert torch format back to kaldi')
    kaldi_text = os.path.join(model_dir, 'convert.kaldi.txt')
    with open(kaldi_text, 'w', encoding='utf8') as fout:
        nnet_desp = model.to_kaldi_net()
        fout.write(nnet_desp)
    fout.close()

    print('Done!')


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    print(args)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    if args.convert_to == 'pytorch':
        print('convert kaldi net to pytorch...')
        convert_to_pytorch(
            configs,
            args.network_file,
            args.model_dir,
            args.model_name
        )
    elif args.convert_to == 'kaldi':
        print('convert pytorch net to kaldi...')
        convert_to_kaldi(
            configs,
            args.network_file,
            args.model_dir,
            args.model_name
        )
    else:
        print('unsupported target network type: {}'.format(args.convert_to))


if __name__ == '__main__':
    main()
