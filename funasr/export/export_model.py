from typing import Union, Dict
from pathlib import Path
from typeguard import check_argument_types

import os
import logging
import torch

from funasr.bin.asr_inference_paraformer import Speech2Text
from funasr.export.models import get_model



class ASRModelExportParaformer:
    def __init__(self, cache_dir: Union[Path, str] = None, onnx: bool = True):
        assert check_argument_types()
        if cache_dir is None:
            cache_dir = Path.home() / "cache" / "export"

        self.cache_dir = Path(cache_dir)
        self.export_config = dict(
            feats_dim=560,
            onnx=onnx,
        )
        logging.info("output dir: {}".format(self.cache_dir))
        self.onnx = onnx

    def export(
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
        if self.onnx:
            self._export_onnx(model, verbose, export_dir)

        logging.info("output dir: {}".format(export_dir))


    def export_from_modelscope(
        self,
        tag_name: str = 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    ):
        
        from funasr.tasks.asr import ASRTaskParaformer as ASRTask
        from modelscope.hub.snapshot_download import snapshot_download

        model_dir = snapshot_download(tag_name, cache_dir=self.cache_dir)
        asr_train_config = os.path.join(model_dir, 'config.yaml')
        asr_model_file = os.path.join(model_dir, 'model.pb')
        cmvn_file = os.path.join(model_dir, 'am.mvn')
        model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, cmvn_file, 'cpu'
        )
        self.export(model, tag_name)



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

if __name__ == '__main__':
    output_dir = "../export"
    export_model = ASRModelExportParaformer(cache_dir=output_dir, onnx=True)
    export_model.export_from_modelscope('damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')