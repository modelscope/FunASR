import json
from typing import Union, Dict
from pathlib import Path
from typeguard import check_argument_types

import os
import logging
import torch

from funasr.bin.asr_inference_paraformer import Speech2Text
from funasr.export.models import get_model
import numpy as np
import random

class ASRModelExportParaformer:
    def __init__(self, cache_dir: Union[Path, str] = None, onnx: bool = True):
        assert check_argument_types()
        self.set_all_random_seed(0)
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "export"

        self.cache_dir = Path(cache_dir)
        self.export_config = dict(
            feats_dim=560,
            onnx=False,
        )
        print("output dir: {}".format(self.cache_dir))
        self.onnx = onnx
        

    def _export(
        self,
        model: Speech2Text,
        tag_name: str = None,
        verbose: bool = False,
    ):

        export_dir = self.cache_dir / tag_name.replace(' ', '-')
        os.makedirs(export_dir, exist_ok=True)

        # export encoder1
        self.export_config["model_name"] = "model"
        model = get_model(
            model,
            self.export_config,
        )
        model.eval()
        # self._export_onnx(model, verbose, export_dir)
        if self.onnx:
            self._export_onnx(model, verbose, export_dir)
        else:
            self._export_torchscripts(model, verbose, export_dir)

        print("output dir: {}".format(export_dir))


    def _export_torchscripts(self, model, verbose, path, enc_size=None):
        if enc_size:
            dummy_input = model.get_dummy_inputs(enc_size)
        else:
            dummy_input = model.get_dummy_inputs_txt()

        # model_script = torch.jit.script(model)
        model_script = torch.jit.trace(model, dummy_input)
        model_script.save(os.path.join(path, f'{model.model_name}.torchscripts'))

    def set_all_random_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
    def export(self,
               tag_name: str = 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
               mode: str = 'paraformer',
               ):
        
        model_dir = tag_name
        if model_dir.startswith('damo/'):
            from modelscope.hub.snapshot_download import snapshot_download
            model_dir = snapshot_download(model_dir, cache_dir=self.cache_dir)
        asr_train_config = os.path.join(model_dir, 'config.yaml')
        asr_model_file = os.path.join(model_dir, 'model.pb')
        cmvn_file = os.path.join(model_dir, 'am.mvn')
        json_file = os.path.join(model_dir, 'configuration.json')
        if mode is None:
            import json
            with open(json_file, 'r') as f:
                config_data = json.load(f)
                mode = config_data['model']['model_config']['mode']
        if mode.startswith('paraformer'):
            from funasr.tasks.asr import ASRTaskParaformer as ASRTask
        elif mode.startswith('uniasr'):
            from funasr.tasks.asr import ASRTaskUniASR as ASRTask
            
        model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, cmvn_file, 'cpu'
        )
        self._export(model, tag_name)
            

    def _export_onnx(self, model, verbose, path, enc_size=None):
        if enc_size:
            dummy_input = model.get_dummy_inputs(enc_size)
        else:
            dummy_input = model.get_dummy_inputs()

        # model_script = torch.jit.script(model)
        model_script = model #torch.jit.trace(model)

        torch.onnx.export(
            model_script,
            dummy_input,
            os.path.join(path, f'{model.model_name}.onnx'),
            verbose=verbose,
            opset_version=12,
            input_names=model.get_input_names(),
            output_names=model.get_output_names(),
            dynamic_axes=model.get_dynamic_axes()
        )


class ASRModelExport:
    def __init__(self, cache_dir: Union[Path, str] = None, onnx: bool = True):
        assert check_argument_types()
        self.set_all_random_seed(0)
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "export"
        
        self.cache_dir = Path(cache_dir)
        self.export_config = dict(
            feats_dim=560,
            onnx=False,
        )
        print("output dir: {}".format(self.cache_dir))
        self.onnx = onnx
    
    def _export(
        self,
        model: Speech2Text,
        tag_name: str = None,
        verbose: bool = False,
    ):
        
        export_dir = self.cache_dir / tag_name.replace(' ', '-')
        os.makedirs(export_dir, exist_ok=True)
        
        # export encoder1
        self.export_config["model_name"] = "model"
        model = get_model(
            model,
            self.export_config,
        )
        model.eval()
        # self._export_onnx(model, verbose, export_dir)
        if self.onnx:
            self._export_onnx(model, verbose, export_dir)
        else:
            self._export_torchscripts(model, verbose, export_dir)
        
        print("output dir: {}".format(export_dir))
    
    def _export_torchscripts(self, model, verbose, path, enc_size=None):
        if enc_size:
            dummy_input = model.get_dummy_inputs(enc_size)
        else:
            dummy_input = model.get_dummy_inputs_txt()
        
        # model_script = torch.jit.script(model)
        model_script = torch.jit.trace(model, dummy_input)
        model_script.save(os.path.join(path, f'{model.model_name}.torchscripts'))
    
    def set_all_random_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
    
    def export(self,
               tag_name: str = 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
               mode: str = 'paraformer',
               ):
        
        model_dir = tag_name
        if model_dir.startswith('damo/'):
            from modelscope.hub.snapshot_download import snapshot_download
            model_dir = snapshot_download(model_dir, cache_dir=self.cache_dir)
        asr_train_config = os.path.join(model_dir, 'config.yaml')
        asr_model_file = os.path.join(model_dir, 'model.pb')
        cmvn_file = os.path.join(model_dir, 'am.mvn')
        json_file = os.path.join(model_dir, 'configuration.json')
        if mode is None:
            import json
            with open(json_file, 'r') as f:
                config_data = json.load(f)
                mode = config_data['model']['model_config']['mode']
        if mode.startswith('paraformer'):
            from funasr.tasks.asr import ASRTaskParaformer as ASRTask
        elif mode.startswith('uniasr'):
            from funasr.tasks.asr import ASRTaskUniASR as ASRTask
        
        model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, cmvn_file, 'cpu'
        )
        self._export(model, tag_name)
    
    def _export_onnx(self, model, verbose, path, enc_size=None):
        if enc_size:
            dummy_input = model.get_dummy_inputs(enc_size)
        else:
            dummy_input = model.get_dummy_inputs()
        
        # model_script = torch.jit.script(model)
        model_script = model  # torch.jit.trace(model)
        
        torch.onnx.export(
            model_script,
            dummy_input,
            os.path.join(path, f'{model.model_name}.onnx'),
            verbose=verbose,
            opset_version=12,
            input_names=model.get_input_names(),
            output_names=model.get_output_names(),
            dynamic_axes=model.get_dynamic_axes()
        )


if __name__ == '__main__':
    import sys
    
    model_path = sys.argv[1]
    output_dir = sys.argv[2]
    onnx = sys.argv[3]
    onnx = onnx.lower()
    onnx = onnx == 'true'
    # model_path = 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
    # output_dir = "../export"
    export_model = ASRModelExportParaformer(cache_dir=output_dir, onnx=onnx)
    export_model.export(model_path)
    # export_model.export('/root/cache/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')