# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from espnet/espnet.

from typing import Tuple

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
from funasr.models.frontend.abs_frontend import AbsFrontend
from typeguard import check_argument_types
from torch.nn.utils.rnn import pad_sequence
import kaldi_native_fbank as knf

class WavFrontend(AbsFrontend):
	"""Conventional frontend structure for ASR.
	"""

	def __init__(
		self,
		cmvn_file: str = None,
		fs: int = 16000,
		window: str = 'hamming',
		n_mels: int = 80,
		frame_length: int = 25,
		frame_shift: int = 10,
		filter_length_min: int = -1,
		filter_length_max: int = -1,
		lfr_m: int = 1,
		lfr_n: int = 1,
		dither: float = 1.0,
		snip_edges: bool = True,
		upsacle_samples: bool = True,
	):
		assert check_argument_types()
		super().__init__()
		self.fs = fs
		self.window = window
		self.n_mels = n_mels
		self.frame_length = frame_length
		self.frame_shift = frame_shift
		self.filter_length_min = filter_length_min
		self.filter_length_max = filter_length_max
		self.lfr_m = lfr_m
		self.lfr_n = lfr_n
		self.cmvn_file = cmvn_file
		self.dither = dither
		self.snip_edges = snip_edges
		self.upsacle_samples = upsacle_samples

	def output_size(self) -> int:
		return self.n_mels * self.lfr_m

	def forward(
		self,
		input: torch.Tensor,
		input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		batch_size = input.size(0)
		feats = []
		feats_lens = []
		for i in range(batch_size):
			waveform_length = input_lengths[i]
			waveform = input[i][:waveform_length]
			waveform = waveform * (1 << 15)
			waveform = waveform.unsqueeze(0)
			mat = kaldi.fbank(waveform,
			                  num_mel_bins=self.n_mels,
			                  frame_length=self.frame_length,
			                  frame_shift=self.frame_shift,
			                  dither=self.dither,
			                  energy_floor=0.0,
			                  window_type=self.window,
			                  sample_frequency=self.fs)

			feat_length = mat.size(0)
			feats.append(mat)
			feats_lens.append(feat_length)

		feats_lens = torch.as_tensor(feats_lens)
		feats_pad = pad_sequence(feats,
		                         batch_first=True,
		                         padding_value=0.0)
		return feats_pad, feats_lens

import kaldi_native_fbank as knf

def fbank_knf(waveform):
	# sampling_rate = 16000
	# samples = torch.randn(16000 * 10)

	opts = knf.FbankOptions()
	opts.frame_opts.samp_freq = 16000
	opts.frame_opts.dither = 0.0
	opts.frame_opts.window_type = "hamming"
	opts.frame_opts.frame_shift_ms = 10.0
	opts.frame_opts.frame_length_ms = 25.0
	opts.mel_opts.num_bins = 80
	opts.energy_floor = 1
	opts.frame_opts.snip_edges = True
	opts.mel_opts.debug_mel = False
	
	fbank = knf.OnlineFbank(opts)
	waveform = waveform * (1 << 15)
	fbank.accept_waveform(opts.frame_opts.samp_freq, waveform.tolist())
	frames = fbank.num_frames_ready
	mat = np.empty([frames, opts.mel_opts.num_bins])
	for i in range(frames):
		mat[i, :] = fbank.get_frame(i)
	return mat

if __name__ == '__main__':
	import librosa
	
	path = "/home/zhifu.gzf/.cache/modelscope/hub/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav"
	waveform, fs = librosa.load(path, sr=None)
	fbank = fbank_knf(waveform)
	frontend = WavFrontend(dither=0.0)
	waveform_tensor = torch.from_numpy(waveform)[None, :]
	fbank_torch, _ = frontend.forward(waveform_tensor, [waveform_tensor.size(1)])
	fbank_torch = fbank_torch.cpu().numpy()[0, :, :]
	diff = fbank - fbank_torch
	diff_max = diff.max()
	diff_sum = diff.abs().sum()
	pass