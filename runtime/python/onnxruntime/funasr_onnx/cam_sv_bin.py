# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os.path
from pathlib import Path
from typing import List, Union, Tuple

import copy
import librosa
import numpy as np
from .utils.utils import ONNXRuntimeError, OrtInferSession, get_logger, read_yaml
from .utils.frontend import WavFrontend, WavFrontendOnline
import torch 
import torchaudio
logging = get_logger()


class CamPlusPlus:
    """
    https://github.com/modelscope/3D-Speaker.git
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = None,
        batch_size: int = 1,
        device_id: Union[str, int] = "-1",
        quantize: bool = False,
        intra_op_num_threads: int = 4,
        cache_dir: str = None,
        **kwargs,
    ):

        if not Path(model_dir).exists():
            try:
                from modelscope.hub.snapshot_download import snapshot_download
            except:
                raise "You are exporting model from modelscope, please install modelscope and try it again. To install modelscope, you could:\n" "\npip3 install -U modelscope\n" "For the users in China, you could install with the command:\n" "\npip3 install -U modelscope -i https://mirror.sjtu.edu.cn/pypi/web/simple"
            try:
                model_dir = snapshot_download(model_dir, cache_dir=cache_dir)
            except:
                raise "model_dir must be model_name in modelscope or local path downloaded from modelscope, but is {}".format(
                    model_dir
                )

        model_file = os.path.join(model_dir, "model.onnx")
        if quantize:
            model_file = os.path.join(model_dir, "model_quant.onnx")
        if not os.path.exists(model_file):
            print(".onnx does not exist, begin to export onnx")
            try:
                from funasr import AutoModel
            except:
                raise "You are exporting onnx, please install funasr and try it again. To install funasr, you could:\n" "\npip3 install -U funasr\n" "For the users in China, you could install with the command:\n" "\npip3 install -U funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple"

            model = AutoModel(model=model_dir)
            model_dir = model.export(type="onnx", quantize=quantize, **kwargs)
        config_file = os.path.join(model_dir, "config.yaml")
        # cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)       
        self.frontend = WavFrontend(cmvn_file=None,window="povey",dither=0, **config["frontend_conf"])
        print("cam model_file={}".format(model_file))
        self.ort_infer = OrtInferSession(
            model_file, device_id, intra_op_num_threads=intra_op_num_threads
        )
        self.batch_size = batch_size   

    def __call__(self, audio_in:Union[str, np.ndarray], **kwargs) -> List: 
        waveforms=self.load_data(audio_in)  
        feats = self.extract_feat(waveforms[0], mean_nor=True)       
        feats = np.expand_dims(feats, axis=0)
        feats = np.expand_dims(feats, axis=0)       
        output = self.infer(feats)    
        return output   
    
    
    def load_wav(self,wav_file, obj_fs=16000):
        wav, fs = torchaudio.load(wav_file)
        if fs != obj_fs:
            print(f'[WARNING]: The sample rate of {wav_file} is not {obj_fs}, resample it.')
            wav, fs = torchaudio.sox_effects.apply_effects_tensor(
                wav, fs, effects=[['rate', str(obj_fs)]]
            )
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
        return wav

    def load_data(self, wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List:
        def load_wav(path: str) -> np.ndarray:
            waveform, _ = librosa.load(path, sr=fs)         
            return waveform

        if isinstance(wav_content, np.ndarray):
            return [wav_content]

        if isinstance(wav_content, str):
            return [load_wav(wav_content)]

        if isinstance(wav_content, list):
            return [load_wav(path) for path in wav_content]

        raise TypeError(f"The type of {wav_content} is not in [str, np.ndarray, list]")
    def extract_feat(self, waveform: np.ndarray,mean_nor=False) -> np.ndarray:  
        waveform=waveform/(1 << 15)
        speech, _ = self.frontend.fbank(waveform)   
        if mean_nor:
             feat_mean = np.mean(speech, axis=0, keepdims=True)
             speech = speech - feat_mean
        return speech  

    def infer(self, feats: List) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_infer(feats)    
        return outputs
