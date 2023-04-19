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
        device: str = "cpu",
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
        self.device = device
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

        if self.device == 'cuda':
            model = model.cuda()
            dummy_input = tuple([i.cuda() for i in dummy_input])

        # """
        import os
        fp16 = float(os.environ.get('FP16', 0.0))
        if fp16:
            # import pdb; pdb.set_trace()
            tmp_models = [model.encoder.model.encoders0, model.encoder.model.encoders]
            for tmp_model in tmp_models:
                state_dict = tmp_model.state_dict()
                for key, value in state_dict.items():
                    if '.feed_forward.w_2.' in key:
                        state_dict[key] = value / fp16
                tmp_model.load_state_dict(state_dict)
        # """

        """
        import pdb; pdb.set_trace()
        for i in range(100):
            import time
            tic = time.time()
            _ = model(*dummy_input)
            print('model: {:.4f}'.format(time.time() - tic))
        """

        """
        import pdb; pdb.set_trace()
        _inputs = dummy_input
        _inputs = tuple([tuple([j.cuda() for j in i]) if isinstance(i, tuple) else i.cuda() for i in _inputs])
        out = model.encoder(*_inputs)
        model.encoder.half()
        _half_inputs = tuple([tuple([j.half() for j in i]) if isinstance(i, tuple) else i.half() for i in _inputs])
        out = model.encoder(*_half_inputs)
        """

        """
        import pdb; pdb.set_trace()
        # _inputs = torch.load('encoders0_inputs.pth')
        _inputs = torch.load('encoders_inputs.pth')
        import os
        fp16 = float(os.environ.get('FP16', 1.0))
        _inputs = (_inputs[0] / fp16, _inputs[1])
        _inputs = tuple([tuple([j.cuda() for j in i]) if isinstance(i, tuple) else i.cuda() for i in _inputs])
        # model_script = torch.jit.trace(model.encoder.model.encoders0, _inputs)
        # model_script.save('model.encoders0.fixfp16.10.pt')
        model_script = torch.jit.trace(model.encoder.model.encoders, _inputs)
        model_script.save('model.encoders.fixfp16.10.pt')
        """

        """
        import pdb; pdb.set_trace()
        _inputs = torch.load('encoders_inputs.pth')
        # _inputs = tuple([tuple([j.cuda() for j in i]) if isinstance(i, tuple) else i.cuda() for i in _inputs])
        out = _inputs
        # out = model.encoder.model.encoders(*_inputs)
        # model.encoder.model.encoders.half()
        # import os
        # fp16 = float(os.environ.get('FP16', 1.0))
        # _inputs = (_inputs[0] / fp16, _inputs[1])
        # _half_inputs = tuple([tuple([j.half() for j in i]) if isinstance(i, tuple) else i.half() for i in _inputs])
        # out = model.encoder.model.encoders(*_half_inputs)
        # out = _half_inputs
        for i in range(49):
            if i == 36:
                import pdb; pdb.set_trace()
                print(i)
            with torch.no_grad():
                out = model.encoder.model.encoders[i](*out)
            print('{} {} {}'.format(i, torch.min(out[0]), torch.max(out[0])))
            # print(out[0])
        import pdb; pdb.set_trace()
        """

        """
        import pdb; pdb.set_trace()
        _inputs = torch.load('encoders31_inputs.pth')
        _inputs = tuple([tuple([j.cuda() for j in i]) if isinstance(i, tuple) else i.cuda() for i in _inputs])
        enc32 = model.encoder.model.encoders[31]
        out = enc32(*_inputs)
        # _half_inputs = tuple([tuple([j.half() for j in i]) if isinstance(i, tuple) else i.half() for i in _inputs])
        # enc32.half()
        # out = enc32(*_half_inputs)
        """

        """
        import pdb; pdb.set_trace()
        import torch_blade
        # model.encoder = torch.jit.load('inference_gpu/model.encoder.blade.pt')
        # model.decoder = torch.jit.load('inference_gpu/model.decoder.blade.pt')
        model.encoder = torch.jit.load('inference_gpu/model.encoder.blade.fp16.pt')
        model.decoder = torch.jit.load('inference_gpu/model.decoder.blade.fp16.pt')
        model_script = torch.jit.trace(model, dummy_input)
        model_script.save('model.blade.opt.pt')
        """

        # import pdb; pdb.set_trace()
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
        self.frontend = model.frontend
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
    parser.add_argument('--device', type=str, default='cpu', help='["cpu", "cuda"]')
    parser.add_argument('--quantize', type=str2bool, default=False, help='export quantized model')
    parser.add_argument('--fallback-num', type=int, default=0, help='amp fallback number')
    parser.add_argument('--audio_in', type=str, default=None, help='["wav", "wav.scp"]')
    parser.add_argument('--calib_num', type=int, default=200, help='calib max num')
    args = parser.parse_args()

    export_model = ModelExport(
        cache_dir=args.export_dir,
        onnx=args.type == 'onnx',
        device=args.device,
        quant=args.quantize,
        fallback_num=args.fallback_num,
        audio_in=args.audio_in,
        calib_num=args.calib_num,
    )
    export_model.export(args.model_name)
