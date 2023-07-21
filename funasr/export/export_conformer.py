import json
from typing import Union, Dict
from pathlib import Path

import os
import logging
import torch

from funasr.export.models import get_model
import numpy as np
import random
from funasr.utils.types import str2bool, str2triple_str
# torch_version = float(".".join(torch.__version__.split(".")[:2]))
# assert torch_version > 1.9

class ModelExport:
    def __init__(
        self,
        cache_dir: Union[Path, str] = None,
        onnx: bool = True,
        device: str = "cpu",
        quant: bool = True,
        fallback_num: int = 0,
        audio_in: str = None,
        calib_num: int = 200,
        model_revision: str = None,
    ):
        self.set_all_random_seed(0)

        self.cache_dir = cache_dir
        self.export_config = dict(
            feats_dim=560,
            onnx=False,
        )
        
        self.onnx = onnx
        self.device = device
        self.quant = quant
        self.fallback_num = fallback_num
        self.frontend = None
        self.audio_in = audio_in
        self.calib_num = calib_num
        self.model_revision = model_revision

    def _export(
        self,
        model,
        model_dir: str = None,
        verbose: bool = False,
    ):

        export_dir = model_dir
        os.makedirs(export_dir, exist_ok=True)

        self.export_config["model_name"] = "model"
        model = get_model(
            model,
            self.export_config,
        )
        model.eval()

        if self.onnx:
            self._export_onnx(model, verbose, export_dir)

        print("output dir: {}".format(export_dir))

    def _export_onnx(self, model, verbose, path):
        model._export_onnx(verbose, path)

    def set_all_random_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    def parse_audio_in(self, audio_in):
        
        wav_list, name_list = [], []
        if audio_in.endswith(".scp"):
            f = open(audio_in, 'r')
            lines = f.readlines()[:self.calib_num]
            for line in lines:
                name, path = line.strip().split()
                name_list.append(name)
                wav_list.append(path)
        else:
            wav_list = [audio_in,]
            name_list = ["test",]
        return wav_list, name_list
    
    def load_feats(self, audio_in: str = None):
        import torchaudio

        wav_list, name_list = self.parse_audio_in(audio_in)
        feats = []
        feats_len = []
        for line in wav_list:
            path = line.strip()
            waveform, sampling_rate = torchaudio.load(path)
            if sampling_rate != self.frontend.fs:
                waveform = torchaudio.transforms.Resample(orig_freq=sampling_rate,
                                                          new_freq=self.frontend.fs)(waveform)
            fbank, fbank_len = self.frontend(waveform, [waveform.size(1)])
            feats.append(fbank)
            feats_len.append(fbank_len)
        return feats, feats_len

    def export(self,
               mode: str = None,
               ):

        if mode.startswith('conformer'):
            from funasr.tasks.asr import ASRTask
            config = os.path.join(model_dir, 'config.yaml')
            model_file = os.path.join(model_dir, 'model.pb')
            cmvn_file = os.path.join(model_dir, 'am.mvn')
            model, asr_train_args = ASRTask.build_model_from_file(
                config, model_file, cmvn_file, 'cpu'
            )
            self.frontend = model.frontend
            self.export_config["feats_dim"] = 560

        self._export(model, self.cache_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--model-name', type=str, action="append", required=True, default=[])
    parser.add_argument('--export-dir', type=str, required=True)
    parser.add_argument('--type', type=str, default='onnx', help='["onnx", "torch"]')
    parser.add_argument('--device', type=str, default='cpu', help='["cpu", "cuda"]')
    parser.add_argument('--quantize', type=str2bool, default=False, help='export quantized model')
    parser.add_argument('--fallback-num', type=int, default=0, help='amp fallback number')
    parser.add_argument('--audio_in', type=str, default=None, help='["wav", "wav.scp"]')
    parser.add_argument('--calib_num', type=int, default=200, help='calib max num')
    parser.add_argument('--model_revision', type=str, default=None, help='model_revision')
    args = parser.parse_args()

    export_model = ModelExport(
        cache_dir=args.export_dir,
        onnx=args.type == 'onnx',
        device=args.device,
        quant=args.quantize,
        fallback_num=args.fallback_num,
        audio_in=args.audio_in,
        calib_num=args.calib_num,
        model_revision=args.model_revision,
    )
    for model_name in args.model_name:
        print("export model: {}".format(model_name))
        export_model.export(model_name)
