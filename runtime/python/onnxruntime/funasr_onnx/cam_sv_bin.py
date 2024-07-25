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
import torchaudio.compliance.kaldi as Kaldi
import torch 
import torchaudio
import soundfile  as  sf
logging = get_logger()


class FBank(object):
    def __init__(self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        sr = 16000
        assert sr==self.sample_rate
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # select single channel
        if wav.shape[0] > 1:
            wav = wav[0, :]
        assert len(wav.shape) == 2 and wav.shape[0]==1
        feat = Kaldi.fbank(wav, num_mel_bins=self.n_mels,
            sample_frequency=sr, dither=dither)
        # feat: [T, N]
        print("before sub",feat)
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat



class CamPlusPlus:
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
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
        cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)

        self.frontend = WavFrontend(cmvn_file=None,window="povey",dither=0,preemphasis_coefficient=0.97, **config["frontend_conf"])
        self.ort_infer = OrtInferSession(
            model_file, device_id, intra_op_num_threads=intra_op_num_threads
        )
        self.batch_size = batch_size   
        self.feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)   

    def __call__(self, audio_in:Union[str, np.ndarray], **kwargs) -> List:
        # #compare diff
        # waveform, sample_rate = torchaudio.load(audio_in)

        # # 使用 torchaudio.compliance.kaldi 提取 fbank 特征
        # sampling_rate = 16000
        # mel_bins = 80
        # frame_length = 25  # ms
        # frame_shift = 10  # ms
        # dither = 0
        # preemphasis_coefficient = 0.97
        # window_type = 'povey'
        # mel_bins = 80 
        # # kaldi_fbank = Kaldi.fbank(waveform, num_mel_bins=mel_bins, sample_frequency=sample_rate, dither=0.0)
        # kaldi_fbank = Kaldi.fbank(
        # waveform,
        # num_mel_bins=mel_bins,
        # frame_length=frame_length,
        # frame_shift=frame_shift,
        # dither=dither,
        # # preemph_coef=preemphasis_coefficient,
        # window_type=window_type,
        # sample_frequency=sampling_rate
        # )
        # # waveform2=self.load_data(audio_in) 
        # samples, _ = sf.read(audio_in)
        # # native_fbank = self.extract_feat(waveform2[0], mean_nor=False) 
        # native_fbank = self.extract_feat(samples, mean_nor=False) 
        # print(kaldi_fbank)
        # print("=======")
        # print(native_fbank)
        # print("======")
        # difference = kaldi_fbank - native_fbank
        # difference_l2_norm = np.linalg.norm(difference, ord=2)

        # print("L2 norm of the difference between the two fbank features:", difference_l2_norm)

        # # wav=self.load_wav(audio_in)
        # # feats = self.feature_extractor(wav).unsqueeze(0)
        # # print("feats",feats.shape)
        # # print(feats)
        # # feats = feats.numpy() 
        # # feats = np.expand_dims(feats, axis=0)   
        # # print(feats.shape)              
        # # output = self.infer(feats)
        # # print(output)
        # # return output

        waveforms=self.load_data(audio_in)   
        print("waveforms")  
        print(waveforms)  
        # feats = self.extract_feat(waveforms) 
        feats = self.extract_feat(waveforms[0], mean_nor=True) 
        # waveforms=torch.from_numpy(waveforms[0])
        # feats=self.feature_extractor(waveforms)
        # feats = FBank(80, sample_rate=16000, mean_nor=False) 
        print("feats",feats.shape)
        print(feats)      
        feats = np.expand_dims(feats, axis=0)
        feats = np.expand_dims(feats, axis=0)
        # # feats = np.array(feats) 
        output = self.infer(feats)
        print(output)
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
        speech, _ = self.frontend.fbank(waveform,is_cam=True)
        print("=====before submean")
        print(speech)
        if mean_nor:
             feat_mean = np.mean(speech, axis=0, keepdims=True)
             speech = speech - feat_mean
        return speech  

    def infer(self, feats: List) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_infer(feats)
        return outputs
