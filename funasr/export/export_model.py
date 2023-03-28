import json
from typing import Union, Dict
from pathlib import Path
from typeguard import check_argument_types

import os
import logging
import torch

from funasr.export.models import get_model
import numpy as np
import random
from funasr.utils.types import str2bool
# torch_version = float(".".join(torch.__version__.split(".")[:2]))
# assert torch_version > 1.9

class ModelExport:
    def __init__(
        self,
        cache_dir: Union[Path, str] = None,
        onnx: bool = True,
        quant: bool = True,
        fallback_num: int = 0,
        audio_in: str = None,
        calib_num: int = 200,
    ):
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
        self.quant = quant
        self.fallback_num = fallback_num
        self.frontend = None
        self.audio_in = audio_in
        self.calib_num = calib_num
        

    def _export(
        self,
        model,
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


    def _torch_quantize(self, model):
        def _run_calibration_data(m):
            # using dummy inputs for a example
            if self.audio_in is not None:
                feats, feats_len = self.load_feats(self.audio_in)
                for i, (feat, len) in enumerate(zip(feats, feats_len)):
                    with torch.no_grad():
                        m(feat, len)
            else:
                dummy_input = model.get_dummy_inputs()
                m(*dummy_input)
            

        from torch_quant.module import ModuleFilter
        from torch_quant.quantizer import Backend, Quantizer
        from funasr.export.models.modules.decoder_layer import DecoderLayerSANM
        from funasr.export.models.modules.encoder_layer import EncoderLayerSANM
        module_filter = ModuleFilter(include_classes=[EncoderLayerSANM, DecoderLayerSANM])
        module_filter.exclude_op_types = [torch.nn.Conv1d]
        quantizer = Quantizer(
            module_filter=module_filter,
            backend=Backend.FBGEMM,
        )
        model.eval()
        calib_model = quantizer.calib(model)
        _run_calibration_data(calib_model)
        if self.fallback_num > 0:
            # perform automatic mixed precision quantization
            amp_model = quantizer.amp(model)
            _run_calibration_data(amp_model)
            quantizer.fallback(amp_model, num=self.fallback_num)
            print('Fallback layers:')
            print('\n'.join(quantizer.module_filter.exclude_names))
        quant_model = quantizer.quantize(model)
        return quant_model


    def _export_torchscripts(self, model, verbose, path, enc_size=None):
        if enc_size:
            dummy_input = model.get_dummy_inputs(enc_size)
        else:
            dummy_input = model.get_dummy_inputs()

        # model_script = torch.jit.script(model)
        model_script = torch.jit.trace(model, dummy_input)
        model_script.save(os.path.join(path, f'{model.model_name}.torchscripts'))

        if self.quant:
            quant_model = self._torch_quantize(model)
            model_script = torch.jit.trace(quant_model, dummy_input)
            model_script.save(os.path.join(path, f'{model.model_name}_quant.torchscripts'))


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
               tag_name: str = 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
               mode: str = None,
               ):
        
        model_dir = tag_name
        if model_dir.startswith('damo'):
            from modelscope.hub.snapshot_download import snapshot_download
            model_dir = snapshot_download(model_dir, cache_dir=self.cache_dir)

        if mode is None:
            import json
            json_file = os.path.join(model_dir, 'configuration.json')
            with open(json_file, 'r') as f:
                config_data = json.load(f)
                mode = config_data['model']['model_config']['mode']
        if mode.startswith('paraformer'):
            from funasr.tasks.asr import ASRTaskParaformer as ASRTask
            config = os.path.join(model_dir, 'config.yaml')
            model_file = os.path.join(model_dir, 'model.pb')
            cmvn_file = os.path.join(model_dir, 'am.mvn')
            model, asr_train_args = ASRTask.build_model_from_file(
                config, model_file, cmvn_file, 'cpu'
            )
            self.frontend = model.frontend
        elif mode.startswith('offline'):
            from funasr.tasks.vad import VADTask
            config = os.path.join(model_dir, 'vad.yaml')
            model_file = os.path.join(model_dir, 'vad.pb')
            cmvn_file = os.path.join(model_dir, 'vad.mvn')
            
            model, vad_infer_args = VADTask.build_model_from_file(
                config, model_file, 'cpu'
            )
            self.export_config["feats_dim"] = 400
        self._export(model, tag_name)
            

    def _export_onnx(self, model, verbose, path, enc_size=None):
        if enc_size:
            dummy_input = model.get_dummy_inputs(enc_size)
        else:
            dummy_input = model.get_dummy_inputs()

        # model_script = torch.jit.script(model)
        model_script = model #torch.jit.trace(model)
        model_path = os.path.join(path, f'{model.model_name}.onnx')

        torch.onnx.export(
            model_script,
            dummy_input,
            model_path,
            verbose=verbose,
            opset_version=14,
            input_names=model.get_input_names(),
            output_names=model.get_output_names(),
            dynamic_axes=model.get_dynamic_axes()
        )

        if self.quant:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            import onnx
            quant_model_path = os.path.join(path, f'{model.model_name}_quant.onnx')
            onnx_model = onnx.load(model_path)
            nodes = [n.name for n in onnx_model.graph.node]
            nodes_to_exclude = [m for m in nodes if 'output' in m]
            quantize_dynamic(
                model_input=model_path,
                model_output=quant_model_path,
                op_types_to_quantize=['MatMul'],
                per_channel=True,
                reduce_range=False,
                weight_type=QuantType.QUInt8,
                nodes_to_exclude=nodes_to_exclude,
            )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--export-dir', type=str, required=True)
    parser.add_argument('--type', type=str, default='onnx', help='["onnx", "torch"]')
    parser.add_argument('--quantize', type=str2bool, default=False, help='export quantized model')
    parser.add_argument('--fallback-num', type=int, default=0, help='amp fallback number')
    parser.add_argument('--audio_in', type=str, default=None, help='["wav", "wav.scp"]')
    parser.add_argument('--calib_num', type=int, default=200, help='calib max num')
    args = parser.parse_args()

    export_model = ModelExport(
        cache_dir=args.export_dir,
        onnx=args.type == 'onnx',
        quant=args.quantize,
        fallback_num=args.fallback_num,
        audio_in=args.audio_in,
        calib_num=args.calib_num,
    )
    export_model.export(args.model_name)
