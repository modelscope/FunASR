# -*- encoding: utf-8 -*-

import os.path
from pathlib import Path
from typing import List, Union, Tuple

import copy
import librosa
import numpy as np

from .utils.utils import (ONNXRuntimeError,
                          OrtInferSession, get_logger,
                          read_yaml)
from .utils.frontend import WavFrontend
from .utils.e2e_vad import E2EVadModel

logging = get_logger()


class Fsmn_vad():
	def __init__(self, model_dir: Union[str, Path] = None,
	             batch_size: int = 1,
	             device_id: Union[str, int] = "-1",
	             quantize: bool = False,
	             intra_op_num_threads: int = 4,
	             max_end_sil: int = None,
	             ):
		
		if not Path(model_dir).exists():
			raise FileNotFoundError(f'{model_dir} does not exist.')
		
		model_file = os.path.join(model_dir, 'model.onnx')
		if quantize:
			model_file = os.path.join(model_dir, 'model_quant.onnx')
		config_file = os.path.join(model_dir, 'vad.yaml')
		cmvn_file = os.path.join(model_dir, 'vad.mvn')
		config = read_yaml(config_file)
		
		self.frontend = WavFrontend(
			cmvn_file=cmvn_file,
			**config['frontend_conf']
		)
		self.ort_infer = OrtInferSession(model_file, device_id, intra_op_num_threads=intra_op_num_threads)
		self.batch_size = batch_size
		self.vad_scorer = E2EVadModel(config["vad_post_conf"])
		self.max_end_sil = max_end_sil if max_end_sil is not None else config["vad_post_conf"]["max_end_silence_time"]
		self.encoder_conf = config["encoder_conf"]
	
	def prepare_cache(self, in_cache: list = []):
		if len(in_cache) > 0:
			return in_cache
		fsmn_layers = self.encoder_conf["fsmn_layers"]
		proj_dim = self.encoder_conf["proj_dim"]
		lorder = self.encoder_conf["lorder"]
		for i in range(fsmn_layers):
			cache = np.zeros((1, proj_dim, lorder-1, 1)).astype(np.float32)
			in_cache.append(cache)
		return in_cache
		
	
	def __call__(self, audio_in: Union[str, np.ndarray, List[str]], **kwargs) -> List:
		# waveform_list = self.load_data(audio_in, self.frontend.opts.frame_opts.samp_freq)
		
		param_dict = kwargs.get('param_dict', dict())
		is_final = param_dict.get('is_final', False)
		audio_in_cache = param_dict.get('audio_in_cache', None)
		audio_in_cum = audio_in
		if audio_in_cache is not None:
			audio_in_cum = np.concatenate((audio_in_cache, audio_in_cum))
		param_dict['audio_in_cache'] = audio_in_cum
		feats, feats_len = self.extract_feat([audio_in_cum])
		
		in_cache = param_dict.get('in_cache', list())
		in_cache = self.prepare_cache(in_cache)
		beg_idx = param_dict.get('beg_idx',0)
		feats = feats[:, beg_idx:beg_idx+8, :]
		param_dict['beg_idx'] = beg_idx + feats.shape[1]
		try:
			inputs = [feats]
			inputs.extend(in_cache)
			scores, out_caches = self.infer(inputs)
			param_dict['in_cache'] = out_caches
			segments = self.vad_scorer(scores, audio_in[None, :], is_final=is_final, max_end_sil=self.max_end_sil)
			# print(segments)
			if len(segments) == 1 and segments[0][0][1] != -1:
				self.frontend.reset_status()
			
			
		except ONNXRuntimeError:
			logging.warning(traceback.format_exc())
			logging.warning("input wav is silence or noise")
			segments = []
	
		return segments

	def load_data(self,
	              wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List:
		def load_wav(path: str) -> np.ndarray:
			waveform, _ = librosa.load(path, sr=fs)
			return waveform
		
		if isinstance(wav_content, np.ndarray):
			return [wav_content]
		
		if isinstance(wav_content, str):
			return [load_wav(wav_content)]
		
		if isinstance(wav_content, list):
			return [load_wav(path) for path in wav_content]
		
		raise TypeError(
			f'The type of {wav_content} is not in [str, np.ndarray, list]')
	
	def extract_feat(self,
	                 waveform_list: List[np.ndarray]
	                 ) -> Tuple[np.ndarray, np.ndarray]:
		feats, feats_len = [], []
		for waveform in waveform_list:
			speech, _ = self.frontend.fbank(waveform)
			feat, feat_len = self.frontend.lfr_cmvn(speech)
			feats.append(feat)
			feats_len.append(feat_len)
		
		feats = self.pad_feats(feats, np.max(feats_len))
		feats_len = np.array(feats_len).astype(np.int32)
		return feats, feats_len
	
	@staticmethod
	def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
		def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
			pad_width = ((0, max_feat_len - cur_len), (0, 0))
			return np.pad(feat, pad_width, 'constant', constant_values=0)
		
		feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
		feats = np.array(feat_res).astype(np.float32)
		return feats
	
	def infer(self, feats: List) -> Tuple[np.ndarray, np.ndarray]:
		
		outputs = self.ort_infer(feats)
		scores, out_caches = outputs[0], outputs[1:]
		return scores, out_caches
	