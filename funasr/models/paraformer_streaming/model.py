import os
import logging
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import tempfile
import codecs
import requests
import re
import copy
import torch
import torch.nn as nn
import random
import numpy as np
import time
# from funasr.layers.abs_normalize import AbsNormalize
from funasr.losses.label_smoothing_loss import (
	LabelSmoothingLoss,  # noqa: H301
)

from funasr.models.paraformer.cif_predictor import mae_loss

from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.models.transformer.utils.nets_utils import make_pad_mask, pad_list
from funasr.metrics.compute_acc import th_accuracy
from funasr.train_utils.device_funcs import force_gatherable

from funasr.models.paraformer.search import Hypothesis

# from funasr.models.model_class_factory import *

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
	from torch.cuda.amp import autocast
else:
	# Nothing to do if torch<1.6.0
	@contextmanager
	def autocast(enabled=True):
		yield
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.utils import postprocess_utils
from funasr.utils.datadir_writer import DatadirWriter
from funasr.utils.timestamp_tools import ts_prediction_lfr6_standard
from funasr.register import tables
from funasr.models.ctc.ctc import CTC

class Paraformer(nn.Module):
	"""
	Author: Speech Lab of DAMO Academy, Alibaba Group
	Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
	https://arxiv.org/abs/2206.08317
	"""
	
	def __init__(
		self,
		# token_list: Union[Tuple[str, ...], List[str]],
		frontend: Optional[str] = None,
		frontend_conf: Optional[Dict] = None,
		specaug: Optional[str] = None,
		specaug_conf: Optional[Dict] = None,
		normalize: str = None,
		normalize_conf: Optional[Dict] = None,
		encoder: str = None,
		encoder_conf: Optional[Dict] = None,
		decoder: str = None,
		decoder_conf: Optional[Dict] = None,
		ctc: str = None,
		ctc_conf: Optional[Dict] = None,
		predictor: str = None,
		predictor_conf: Optional[Dict] = None,
		ctc_weight: float = 0.5,
		input_size: int = 80,
		vocab_size: int = -1,
		ignore_id: int = -1,
		blank_id: int = 0,
		sos: int = 1,
		eos: int = 2,
		lsm_weight: float = 0.0,
		length_normalized_loss: bool = False,
		# report_cer: bool = True,
		# report_wer: bool = True,
		# sym_space: str = "<space>",
		# sym_blank: str = "<blank>",
		# extract_feats_in_collect_stats: bool = True,
		# predictor=None,
		predictor_weight: float = 0.0,
		predictor_bias: int = 0,
		sampling_ratio: float = 0.2,
		share_embedding: bool = False,
		# preencoder: Optional[AbsPreEncoder] = None,
		# postencoder: Optional[AbsPostEncoder] = None,
		use_1st_decoder_loss: bool = False,
		**kwargs,
	):

		super().__init__()
		
		# import pdb;
		# pdb.set_trace()
		
		if frontend is not None:
			frontend_class = tables.frontend_classes.get_class(frontend.lower())
			frontend = frontend_class(**frontend_conf)
		if specaug is not None:
			specaug_class = tables.specaug_classes.get_class(specaug.lower())
			specaug = specaug_class(**specaug_conf)
		if normalize is not None:
			normalize_class = tables.normalize_classes.get_class(normalize.lower())
			normalize = normalize_class(**normalize_conf)
		encoder_class = tables.encoder_classes.get_class(encoder.lower())
		encoder = encoder_class(input_size=input_size, **encoder_conf)
		encoder_output_size = encoder.output_size()
		if decoder is not None:
			decoder_class = tables.decoder_classes.get_class(decoder.lower())
			decoder = decoder_class(
				vocab_size=vocab_size,
				encoder_output_size=encoder_output_size,
				**decoder_conf,
			)
		if ctc_weight > 0.0:
			
			if ctc_conf is None:
				ctc_conf = {}
			
			ctc = CTC(
				odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf
			)
		if predictor is not None:
			predictor_class = tables.predictor_classes.get_class(predictor.lower())
			predictor = predictor_class(**predictor_conf)
		
		# note that eos is the same as sos (equivalent ID)
		self.blank_id = blank_id
		self.sos = sos if sos is not None else vocab_size - 1
		self.eos = eos if eos is not None else vocab_size - 1
		self.vocab_size = vocab_size
		self.ignore_id = ignore_id
		self.ctc_weight = ctc_weight
		# self.token_list = token_list.copy()
		#
		self.frontend = frontend
		self.specaug = specaug
		self.normalize = normalize
		# self.preencoder = preencoder
		# self.postencoder = postencoder
		self.encoder = encoder
		#
		# if not hasattr(self.encoder, "interctc_use_conditioning"):
		# 	self.encoder.interctc_use_conditioning = False
		# if self.encoder.interctc_use_conditioning:
		# 	self.encoder.conditioning_layer = torch.nn.Linear(
		# 		vocab_size, self.encoder.output_size()
		# 	)
		#
		# self.error_calculator = None
		#
		if ctc_weight == 1.0:
			self.decoder = None
		else:
			self.decoder = decoder
		
		self.criterion_att = LabelSmoothingLoss(
			size=vocab_size,
			padding_idx=ignore_id,
			smoothing=lsm_weight,
			normalize_length=length_normalized_loss,
		)
		#
		# if report_cer or report_wer:
		# 	self.error_calculator = ErrorCalculator(
		# 		token_list, sym_space, sym_blank, report_cer, report_wer
		# 	)
		#
		if ctc_weight == 0.0:
			self.ctc = None
		else:
			self.ctc = ctc
		#
		# self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
		self.predictor = predictor
		self.predictor_weight = predictor_weight
		self.predictor_bias = predictor_bias
		self.sampling_ratio = sampling_ratio
		self.criterion_pre = mae_loss(normalize_length=length_normalized_loss)
		# self.step_cur = 0
		#
		self.share_embedding = share_embedding
		if self.share_embedding:
			self.decoder.embed = None
		
		self.use_1st_decoder_loss = use_1st_decoder_loss
		self.length_normalized_loss = length_normalized_loss
		self.beam_search = None
	
	def forward(
		self,
		speech: torch.Tensor,
		speech_lengths: torch.Tensor,
		text: torch.Tensor,
		text_lengths: torch.Tensor,
		**kwargs,
	) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
		"""Encoder + Decoder + Calc loss
		Args:
				speech: (Batch, Length, ...)
				speech_lengths: (Batch, )
				text: (Batch, Length)
				text_lengths: (Batch,)
		"""
		# import pdb;
		# pdb.set_trace()
		if len(text_lengths.size()) > 1:
			text_lengths = text_lengths[:, 0]
		if len(speech_lengths.size()) > 1:
			speech_lengths = speech_lengths[:, 0]
		
		batch_size = speech.shape[0]
		
		
		# Encoder
		encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

		
		loss_ctc, cer_ctc = None, None
		loss_pre = None
		stats = dict()
		
		# decoder: CTC branch
		if self.ctc_weight != 0.0:
			loss_ctc, cer_ctc = self._calc_ctc_loss(
				encoder_out, encoder_out_lens, text, text_lengths
			)
			
			# Collect CTC branch stats
			stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
			stats["cer_ctc"] = cer_ctc
		

		# decoder: Attention decoder branch
		loss_att, acc_att, cer_att, wer_att, loss_pre, pre_loss_att = self._calc_att_loss(
			encoder_out, encoder_out_lens, text, text_lengths
		)
		
		# 3. CTC-Att loss definition
		if self.ctc_weight == 0.0:
			loss = loss_att + loss_pre * self.predictor_weight
		else:
			loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att + loss_pre * self.predictor_weight
		
		
		# Collect Attn branch stats
		stats["loss_att"] = loss_att.detach() if loss_att is not None else None
		stats["pre_loss_att"] = pre_loss_att.detach() if pre_loss_att is not None else None
		stats["acc"] = acc_att
		stats["cer"] = cer_att
		stats["wer"] = wer_att
		stats["loss_pre"] = loss_pre.detach().cpu() if loss_pre is not None else None
		
		stats["loss"] = torch.clone(loss.detach())
		
		# force_gatherable: to-device and to-tensor if scalar for DataParallel
		if self.length_normalized_loss:
			batch_size = (text_lengths + self.predictor_bias).sum()
		loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
		return loss, stats, weight
	

	def encode(
		self, speech: torch.Tensor, speech_lengths: torch.Tensor, **kwargs,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Frontend + Encoder. Note that this method is used by asr_inference.py
		Args:
				speech: (Batch, Length, ...)
				speech_lengths: (Batch, )
				ind: int
		"""
		with autocast(False):

			# Data augmentation
			if self.specaug is not None and self.training:
				speech, speech_lengths = self.specaug(speech, speech_lengths)
			
			# Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
			if self.normalize is not None:
				speech, speech_lengths = self.normalize(speech, speech_lengths)
		

		# Forward encoder
		encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
		if isinstance(encoder_out, tuple):
			encoder_out = encoder_out[0]

		return encoder_out, encoder_out_lens
	
	def calc_predictor(self, encoder_out, encoder_out_lens):
		
		encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
			encoder_out.device)
		pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(encoder_out, None,
		                                                                               encoder_out_mask,
		                                                                               ignore_id=self.ignore_id)
		return pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index
	
	def cal_decoder_with_predictor(self, encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens):
		
		decoder_outs = self.decoder(
			encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens
		)
		decoder_out = decoder_outs[0]
		decoder_out = torch.log_softmax(decoder_out, dim=-1)
		return decoder_out, ys_pad_lens

	def _calc_att_loss(
		self,
		encoder_out: torch.Tensor,
		encoder_out_lens: torch.Tensor,
		ys_pad: torch.Tensor,
		ys_pad_lens: torch.Tensor,
	):
		encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
			encoder_out.device)
		if self.predictor_bias == 1:
			_, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
			ys_pad_lens = ys_pad_lens + self.predictor_bias
		pre_acoustic_embeds, pre_token_length, _, pre_peak_index = self.predictor(encoder_out, ys_pad, encoder_out_mask,
		                                                                          ignore_id=self.ignore_id)
		
		# 0. sampler
		decoder_out_1st = None
		pre_loss_att = None
		if self.sampling_ratio > 0.0:

			sematic_embeds, decoder_out_1st = self.sampler(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens,
			                                               pre_acoustic_embeds)
		else:
			sematic_embeds = pre_acoustic_embeds
		
		# 1. Forward decoder
		decoder_outs = self.decoder(
			encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens
		)
		decoder_out, _ = decoder_outs[0], decoder_outs[1]
		
		if decoder_out_1st is None:
			decoder_out_1st = decoder_out
		# 2. Compute attention loss
		loss_att = self.criterion_att(decoder_out, ys_pad)
		acc_att = th_accuracy(
			decoder_out_1st.view(-1, self.vocab_size),
			ys_pad,
			ignore_label=self.ignore_id,
		)
		loss_pre = self.criterion_pre(ys_pad_lens.type_as(pre_token_length), pre_token_length)
		
		# Compute cer/wer using attention-decoder
		if self.training or self.error_calculator is None:
			cer_att, wer_att = None, None
		else:
			ys_hat = decoder_out_1st.argmax(dim=-1)
			cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())
		
		return loss_att, acc_att, cer_att, wer_att, loss_pre, pre_loss_att
	
	def sampler(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, pre_acoustic_embeds):
		
		tgt_mask = (~make_pad_mask(ys_pad_lens, maxlen=ys_pad_lens.max())[:, :, None]).to(ys_pad.device)
		ys_pad_masked = ys_pad * tgt_mask[:, :, 0]
		if self.share_embedding:
			ys_pad_embed = self.decoder.output_layer.weight[ys_pad_masked]
		else:
			ys_pad_embed = self.decoder.embed(ys_pad_masked)
		with torch.no_grad():
			decoder_outs = self.decoder(
				encoder_out, encoder_out_lens, pre_acoustic_embeds, ys_pad_lens
			)
			decoder_out, _ = decoder_outs[0], decoder_outs[1]
			pred_tokens = decoder_out.argmax(-1)
			nonpad_positions = ys_pad.ne(self.ignore_id)
			seq_lens = (nonpad_positions).sum(1)
			same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
			input_mask = torch.ones_like(nonpad_positions)
			bsz, seq_len = ys_pad.size()
			for li in range(bsz):
				target_num = (((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio).long()
				if target_num > 0:
					input_mask[li].scatter_(dim=0,
					                        index=torch.randperm(seq_lens[li])[:target_num].to(input_mask.device),
					                        value=0)
			input_mask = input_mask.eq(1)
			input_mask = input_mask.masked_fill(~nonpad_positions, False)
			input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)
		
		sematic_embeds = pre_acoustic_embeds.masked_fill(~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
			input_mask_expand_dim, 0)
		return sematic_embeds * tgt_mask, decoder_out * tgt_mask
		
	def _calc_ctc_loss(
		self,
		encoder_out: torch.Tensor,
		encoder_out_lens: torch.Tensor,
		ys_pad: torch.Tensor,
		ys_pad_lens: torch.Tensor,
	):
		# Calc CTC loss
		loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
		
		# Calc CER using CTC
		cer_ctc = None
		if not self.training and self.error_calculator is not None:
			ys_hat = self.ctc.argmax(encoder_out).data
			cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
		return loss_ctc, cer_ctc

	
	def init_beam_search(self,
	                     **kwargs,
	                     ):
		from funasr.models.paraformer.search import BeamSearchPara
		from funasr.models.transformer.scorers.ctc import CTCPrefixScorer
		from funasr.models.transformer.scorers.length_bonus import LengthBonus
	
		# 1. Build ASR model
		scorers = {}
		
		if self.ctc != None:
			ctc = CTCPrefixScorer(ctc=self.ctc, eos=self.eos)
			scorers.update(
				ctc=ctc
			)
		token_list = kwargs.get("token_list")
		scorers.update(
			length_bonus=LengthBonus(len(token_list)),
		)

		
		# 3. Build ngram model
		# ngram is not supported now
		ngram = None
		scorers["ngram"] = ngram
		
		weights = dict(
			decoder=1.0 - kwargs.get("decoding_ctc_weight"),
			ctc=kwargs.get("decoding_ctc_weight", 0.0),
			lm=kwargs.get("lm_weight", 0.0),
			ngram=kwargs.get("ngram_weight", 0.0),
			length_bonus=kwargs.get("penalty", 0.0),
		)
		beam_search = BeamSearchPara(
			beam_size=kwargs.get("beam_size", 2),
			weights=weights,
			scorers=scorers,
			sos=self.sos,
			eos=self.eos,
			vocab_size=len(token_list),
			token_list=token_list,
			pre_beam_score_key=None if self.ctc_weight == 1.0 else "full",
		)
		# beam_search.to(device=kwargs.get("device", "cpu"), dtype=getattr(torch, kwargs.get("dtype", "float32"))).eval()
		# for scorer in scorers.values():
		# 	if isinstance(scorer, torch.nn.Module):
		# 		scorer.to(device=kwargs.get("device", "cpu"), dtype=getattr(torch, kwargs.get("dtype", "float32"))).eval()
		self.beam_search = beam_search
		
	def generate(self,
             data_in: list,
             data_lengths: list=None,
             key: list=None,
             tokenizer=None,
             **kwargs,
             ):
		
		# init beamsearch
		is_use_ctc = kwargs.get("decoding_ctc_weight", 0.0) > 0.00001 and self.ctc != None
		is_use_lm = kwargs.get("lm_weight", 0.0) > 0.00001 and kwargs.get("lm_file", None) is not None
		if self.beam_search is None and (is_use_lm or is_use_ctc):
			logging.info("enable beam_search")
			self.init_beam_search(**kwargs)
			self.nbest = kwargs.get("nbest", 1)
		
		meta_data = {}
		# extract fbank feats
		time1 = time.perf_counter()
		audio_sample_list = load_audio_text_image_video(data_in, fs=self.frontend.fs, audio_fs=kwargs.get("fs", 16000))
		time2 = time.perf_counter()
		meta_data["load_data"] = f"{time2 - time1:0.3f}"
		speech, speech_lengths = extract_fbank(audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=self.frontend)
		time3 = time.perf_counter()
		meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
		meta_data["batch_data_time"] = speech_lengths.sum().item() * self.frontend.frame_shift * self.frontend.lfr_n / 1000
		
		speech.to(device=kwargs["device"]), speech_lengths.to(device=kwargs["device"])

		# Encoder
		encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
		if isinstance(encoder_out, tuple):
			encoder_out = encoder_out[0]
		
		# predictor
		predictor_outs = self.calc_predictor(encoder_out, encoder_out_lens)
		pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = predictor_outs[0], predictor_outs[1], \
		                                                                predictor_outs[2], predictor_outs[3]
		pre_token_length = pre_token_length.round().long()
		if torch.max(pre_token_length) < 1:
			return []
		decoder_outs = self.cal_decoder_with_predictor(encoder_out, encoder_out_lens, pre_acoustic_embeds,
		                                                         pre_token_length)
		decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]


		results = []
		b, n, d = decoder_out.size()
		for i in range(b):
			x = encoder_out[i, :encoder_out_lens[i], :]
			am_scores = decoder_out[i, :pre_token_length[i], :]
			if self.beam_search is not None:
				nbest_hyps = self.beam_search(
					x=x, am_scores=am_scores, maxlenratio=kwargs.get("maxlenratio", 0.0), minlenratio=kwargs.get("minlenratio", 0.0)
				)
				
				nbest_hyps = nbest_hyps[: self.nbest]
			else:

				yseq = am_scores.argmax(dim=-1)
				score = am_scores.max(dim=-1)[0]
				score = torch.sum(score, dim=-1)
				# pad with mask tokens to ensure compatibility with sos/eos tokens
				yseq = torch.tensor(
					[self.sos] + yseq.tolist() + [self.eos], device=yseq.device
				)
				nbest_hyps = [Hypothesis(yseq=yseq, score=score)]
			for nbest_idx, hyp in enumerate(nbest_hyps):
				ibest_writer = None
				if ibest_writer is None and kwargs.get("output_dir") is not None:
					writer = DatadirWriter(kwargs.get("output_dir"))
					ibest_writer = writer[f"{nbest_idx+1}best_recog"]
				# remove sos/eos and get results
				last_pos = -1
				if isinstance(hyp.yseq, list):
					token_int = hyp.yseq[1:last_pos]
				else:
					token_int = hyp.yseq[1:last_pos].tolist()
					
				# remove blank symbol id, which is assumed to be 0
				token_int = list(filter(lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int))
				
				# Change integer-ids to tokens
				token = tokenizer.ids2tokens(token_int)
				text = tokenizer.tokens2text(token)
				
				text_postprocessed, _ = postprocess_utils.sentence_postprocess(token)
				result_i = {"key": key[i], "token": token, "text": text, "text_postprocessed": text_postprocessed}
				results.append(result_i)
				
				if ibest_writer is not None:
					ibest_writer["token"][key[i]] = " ".join(token)
					ibest_writer["text"][key[i]] = text
					ibest_writer["text_postprocessed"][key[i]] = text_postprocessed
		
		return results, meta_data



class BiCifParaformer(Paraformer):
	"""
	Author: Speech Lab of DAMO Academy, Alibaba Group
	Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
	https://arxiv.org/abs/2206.08317
	"""
	
	def __init__(
		self,
		*args,
		**kwargs,
	):
		super().__init__(*args, **kwargs)


	def _calc_pre2_loss(
		self,
		encoder_out: torch.Tensor,
		encoder_out_lens: torch.Tensor,
		ys_pad: torch.Tensor,
		ys_pad_lens: torch.Tensor,
	):
		encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
			encoder_out.device)
		if self.predictor_bias == 1:
			_, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
			ys_pad_lens = ys_pad_lens + self.predictor_bias
		_, _, _, _, pre_token_length2 = self.predictor(encoder_out, ys_pad, encoder_out_mask, ignore_id=self.ignore_id)
		
		# loss_pre = self.criterion_pre(ys_pad_lens.type_as(pre_token_length), pre_token_length)
		loss_pre2 = self.criterion_pre(ys_pad_lens.type_as(pre_token_length2), pre_token_length2)
		
		return loss_pre2
	
	
	def _calc_att_loss(
		self,
		encoder_out: torch.Tensor,
		encoder_out_lens: torch.Tensor,
		ys_pad: torch.Tensor,
		ys_pad_lens: torch.Tensor,
	):
		encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
			encoder_out.device)
		if self.predictor_bias == 1:
			_, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
			ys_pad_lens = ys_pad_lens + self.predictor_bias
		pre_acoustic_embeds, pre_token_length, _, pre_peak_index, _ = self.predictor(encoder_out, ys_pad,
		                                                                             encoder_out_mask,
		                                                                             ignore_id=self.ignore_id)
		
		# 0. sampler
		decoder_out_1st = None
		if self.sampling_ratio > 0.0:
			sematic_embeds, decoder_out_1st = self.sampler(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens,
			                                               pre_acoustic_embeds)
		else:
			sematic_embeds = pre_acoustic_embeds
		
		# 1. Forward decoder
		decoder_outs = self.decoder(
			encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens
		)
		decoder_out, _ = decoder_outs[0], decoder_outs[1]
		
		if decoder_out_1st is None:
			decoder_out_1st = decoder_out
		# 2. Compute attention loss
		loss_att = self.criterion_att(decoder_out, ys_pad)
		acc_att = th_accuracy(
			decoder_out_1st.view(-1, self.vocab_size),
			ys_pad,
			ignore_label=self.ignore_id,
		)
		loss_pre = self.criterion_pre(ys_pad_lens.type_as(pre_token_length), pre_token_length)
		
		# Compute cer/wer using attention-decoder
		if self.training or self.error_calculator is None:
			cer_att, wer_att = None, None
		else:
			ys_hat = decoder_out_1st.argmax(dim=-1)
			cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())
		
		return loss_att, acc_att, cer_att, wer_att, loss_pre


	def calc_predictor(self, encoder_out, encoder_out_lens):
		encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
			encoder_out.device)
		pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index, pre_token_length2 = self.predictor(encoder_out,
		                                                                                                  None,
		                                                                                                  encoder_out_mask,
		                                                                                                  ignore_id=self.ignore_id)
		return pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index


	def calc_predictor_timestamp(self, encoder_out, encoder_out_lens, token_num):
		encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
			encoder_out.device)
		ds_alphas, ds_cif_peak, us_alphas, us_peaks = self.predictor.get_upsample_timestamp(encoder_out,
		                                                                                    encoder_out_mask,
		                                                                                    token_num)
		return ds_alphas, ds_cif_peak, us_alphas, us_peaks
	
	
	def forward(
		self,
		speech: torch.Tensor,
		speech_lengths: torch.Tensor,
		text: torch.Tensor,
		text_lengths: torch.Tensor,
		**kwargs,
	) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
		"""Frontend + Encoder + Decoder + Calc loss
		Args:
				speech: (Batch, Length, ...)
				speech_lengths: (Batch, )
				text: (Batch, Length)
				text_lengths: (Batch,)
		"""
		if len(text_lengths.size()) > 1:
			text_lengths = text_lengths[:, 0]
		if len(speech_lengths.size()) > 1:
			speech_lengths = speech_lengths[:, 0]
		
		batch_size = speech.shape[0]
		
		# Encoder
		encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)


		loss_ctc, cer_ctc = None, None
		loss_pre = None
		stats = dict()
		
		# decoder: CTC branch
		if self.ctc_weight != 0.0:
			loss_ctc, cer_ctc = self._calc_ctc_loss(
				encoder_out, encoder_out_lens, text, text_lengths
			)
			
			# Collect CTC branch stats
			stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
			stats["cer_ctc"] = cer_ctc


		# decoder: Attention decoder branch
		loss_att, acc_att, cer_att, wer_att, loss_pre = self._calc_att_loss(
			encoder_out, encoder_out_lens, text, text_lengths
		)
		
		loss_pre2 = self._calc_pre2_loss(
			encoder_out, encoder_out_lens, text, text_lengths
		)
		
		# 3. CTC-Att loss definition
		if self.ctc_weight == 0.0:
			loss = loss_att + loss_pre * self.predictor_weight + loss_pre2 * self.predictor_weight * 0.5
		else:
			loss = self.ctc_weight * loss_ctc + (
				1 - self.ctc_weight) * loss_att + loss_pre * self.predictor_weight + loss_pre2 * self.predictor_weight * 0.5
		
		# Collect Attn branch stats
		stats["loss_att"] = loss_att.detach() if loss_att is not None else None
		stats["acc"] = acc_att
		stats["cer"] = cer_att
		stats["wer"] = wer_att
		stats["loss_pre"] = loss_pre.detach().cpu() if loss_pre is not None else None
		stats["loss_pre2"] = loss_pre2.detach().cpu()
		
		stats["loss"] = torch.clone(loss.detach())
		
		# force_gatherable: to-device and to-tensor if scalar for DataParallel
		if self.length_normalized_loss:
			batch_size = int((text_lengths + self.predictor_bias).sum())
		
		loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
		return loss, stats, weight
	
	def generate(self,
	             data_in: list,
	             data_lengths: list = None,
	             key: list = None,
	             tokenizer=None,
	             **kwargs,
	             ):
		
		# init beamsearch
		is_use_ctc = kwargs.get("decoding_ctc_weight", 0.0) > 0.00001 and self.ctc != None
		is_use_lm = kwargs.get("lm_weight", 0.0) > 0.00001 and kwargs.get("lm_file", None) is not None
		if self.beam_search is None and (is_use_lm or is_use_ctc):
			logging.info("enable beam_search")
			self.init_beam_search(**kwargs)
			self.nbest = kwargs.get("nbest", 1)
		
		meta_data = {}
		# extract fbank feats
		time1 = time.perf_counter()
		audio_sample_list = load_audio_text_image_video(data_in, fs=self.frontend.fs, audio_fs=kwargs.get("fs", 16000))
		time2 = time.perf_counter()
		meta_data["load_data"] = f"{time2 - time1:0.3f}"
		speech, speech_lengths = extract_fbank(audio_sample_list, data_type=kwargs.get("data_type", "sound"),
		                                       frontend=self.frontend)
		time3 = time.perf_counter()
		meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
		meta_data[
			"batch_data_time"] = speech_lengths.sum().item() * self.frontend.frame_shift * self.frontend.lfr_n / 1000
		
		speech.to(device=kwargs["device"]), speech_lengths.to(device=kwargs["device"])
		
		# Encoder
		encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
		if isinstance(encoder_out, tuple):
			encoder_out = encoder_out[0]
		
		# predictor
		predictor_outs = self.calc_predictor(encoder_out, encoder_out_lens)
		pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = predictor_outs[0], predictor_outs[1], \
		                                                                predictor_outs[2], predictor_outs[3]
		pre_token_length = pre_token_length.round().long()
		if torch.max(pre_token_length) < 1:
			return []
		decoder_outs = self.cal_decoder_with_predictor(encoder_out, encoder_out_lens, pre_acoustic_embeds,
		                                               pre_token_length)
		decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]
		
		# BiCifParaformer, test no bias cif2

		_, _, us_alphas, us_peaks = self.calc_predictor_timestamp(encoder_out, encoder_out_lens,
			                                                                    pre_token_length)
		
		results = []
		b, n, d = decoder_out.size()
		for i in range(b):
			x = encoder_out[i, :encoder_out_lens[i], :]
			am_scores = decoder_out[i, :pre_token_length[i], :]
			if self.beam_search is not None:
				nbest_hyps = self.beam_search(
					x=x, am_scores=am_scores, maxlenratio=kwargs.get("maxlenratio", 0.0),
					minlenratio=kwargs.get("minlenratio", 0.0)
				)
				
				nbest_hyps = nbest_hyps[: self.nbest]
			else:
				
				yseq = am_scores.argmax(dim=-1)
				score = am_scores.max(dim=-1)[0]
				score = torch.sum(score, dim=-1)
				# pad with mask tokens to ensure compatibility with sos/eos tokens
				yseq = torch.tensor(
					[self.sos] + yseq.tolist() + [self.eos], device=yseq.device
				)
				nbest_hyps = [Hypothesis(yseq=yseq, score=score)]
			for nbest_idx, hyp in enumerate(nbest_hyps):
				ibest_writer = None
				if ibest_writer is None and kwargs.get("output_dir") is not None:
					writer = DatadirWriter(kwargs.get("output_dir"))
					ibest_writer = writer[f"{nbest_idx + 1}best_recog"]
				# remove sos/eos and get results
				last_pos = -1
				if isinstance(hyp.yseq, list):
					token_int = hyp.yseq[1:last_pos]
				else:
					token_int = hyp.yseq[1:last_pos].tolist()
				
				# remove blank symbol id, which is assumed to be 0
				token_int = list(filter(lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int))
				
				# Change integer-ids to tokens
				token = tokenizer.ids2tokens(token_int)
				text = tokenizer.tokens2text(token)
				
				_, timestamp = ts_prediction_lfr6_standard(us_alphas[i][:encoder_out_lens[i] * 3],
				                                           us_peaks[i][:encoder_out_lens[i] * 3],
				                                           copy.copy(token),
				                                           vad_offset=kwargs.get("begin_time", 0))
				
				text_postprocessed, time_stamp_postprocessed, word_lists = postprocess_utils.sentence_postprocess(token, timestamp)
				
				result_i = {"key": key[i], "token": token, "text": text, "text_postprocessed": text_postprocessed,
				            "time_stamp_postprocessed": time_stamp_postprocessed,
				            "word_lists": word_lists
				            }
				results.append(result_i)
				
				if ibest_writer is not None:
					ibest_writer["token"][key[i]] = " ".join(token)
					ibest_writer["text"][key[i]] = text
					ibest_writer["text_postprocessed"][key[i]] = text_postprocessed
					
		
		return results, meta_data


class ParaformerStreaming(Paraformer):
	"""
	Author: Speech Lab of DAMO Academy, Alibaba Group
	Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
	https://arxiv.org/abs/2206.08317
	"""
	
	def __init__(
		self,
		*args,
		**kwargs,
	):
		
		super().__init__(*args, **kwargs)
		
		# import pdb;
		# pdb.set_trace()
		self.sampling_ratio = kwargs.get("sampling_ratio", 0.2)


		self.scama_mask = None
		if hasattr(self.encoder, "overlap_chunk_cls") and self.encoder.overlap_chunk_cls is not None:
			from funasr.models.scama.chunk_utilis import build_scama_mask_for_cross_attention_decoder
			self.build_scama_mask_for_cross_attention_decoder_fn = build_scama_mask_for_cross_attention_decoder
			self.decoder_attention_chunk_type = kwargs.get("decoder_attention_chunk_type", "chunk")


	
	def forward(
		self,
		speech: torch.Tensor,
		speech_lengths: torch.Tensor,
		text: torch.Tensor,
		text_lengths: torch.Tensor,
		**kwargs,
	) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
		"""Encoder + Decoder + Calc loss
		Args:
				speech: (Batch, Length, ...)
				speech_lengths: (Batch, )
				text: (Batch, Length)
				text_lengths: (Batch,)
		"""
		# import pdb;
		# pdb.set_trace()
		decoding_ind = kwargs.get("decoding_ind")
		if len(text_lengths.size()) > 1:
			text_lengths = text_lengths[:, 0]
		if len(speech_lengths.size()) > 1:
			speech_lengths = speech_lengths[:, 0]
		
		batch_size = speech.shape[0]
		
		# Encoder
		if hasattr(self.encoder, "overlap_chunk_cls"):
			ind = self.encoder.overlap_chunk_cls.random_choice(self.training, decoding_ind)
			encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, ind=ind)
		else:
			encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
		
		loss_ctc, cer_ctc = None, None
		loss_pre = None
		stats = dict()
		
		# decoder: CTC branch

		if self.ctc_weight > 0.0:
			if hasattr(self.encoder, "overlap_chunk_cls"):
				encoder_out_ctc, encoder_out_lens_ctc = self.encoder.overlap_chunk_cls.remove_chunk(encoder_out,
				                                                                                    encoder_out_lens,
				                                                                                    chunk_outs=None)
			else:
				encoder_out_ctc, encoder_out_lens_ctc = encoder_out, encoder_out_lens
				
			loss_ctc, cer_ctc = self._calc_ctc_loss(
				encoder_out_ctc, encoder_out_lens_ctc, text, text_lengths
			)
			# Collect CTC branch stats
			stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
			stats["cer_ctc"] = cer_ctc
		
		# decoder: Attention decoder branch
		loss_att, acc_att, cer_att, wer_att, loss_pre, pre_loss_att = self._calc_att_predictor_loss(
			encoder_out, encoder_out_lens, text, text_lengths
		)
		
		# 3. CTC-Att loss definition
		if self.ctc_weight == 0.0:
			loss = loss_att + loss_pre * self.predictor_weight
		else:
			loss = self.ctc_weight * loss_ctc + (
					1 - self.ctc_weight) * loss_att + loss_pre * self.predictor_weight
		
		# Collect Attn branch stats
		stats["loss_att"] = loss_att.detach() if loss_att is not None else None
		stats["pre_loss_att"] = pre_loss_att.detach() if pre_loss_att is not None else None
		stats["acc"] = acc_att
		stats["cer"] = cer_att
		stats["wer"] = wer_att
		stats["loss_pre"] = loss_pre.detach().cpu() if loss_pre is not None else None
		
		stats["loss"] = torch.clone(loss.detach())
		
		# force_gatherable: to-device and to-tensor if scalar for DataParallel
		if self.length_normalized_loss:
			batch_size = (text_lengths + self.predictor_bias).sum()
		loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
		return loss, stats, weight
	
	def encode_chunk(
		self, speech: torch.Tensor, speech_lengths: torch.Tensor, cache: dict = None, **kwargs,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Frontend + Encoder. Note that this method is used by asr_inference.py
		Args:
				speech: (Batch, Length, ...)
				speech_lengths: (Batch, )
				ind: int
		"""
		with autocast(False):
			
			# Data augmentation
			if self.specaug is not None and self.training:
				speech, speech_lengths = self.specaug(speech, speech_lengths)
			
			# Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
			if self.normalize is not None:
				speech, speech_lengths = self.normalize(speech, speech_lengths)
		
		# Forward encoder
		encoder_out, encoder_out_lens, _ = self.encoder.forward_chunk(speech, speech_lengths, cache=cache["encoder"])
		if isinstance(encoder_out, tuple):
			encoder_out = encoder_out[0]
		
		return encoder_out, torch.tensor([encoder_out.size(1)])
	
	def _calc_att_predictor_loss(
		self,
		encoder_out: torch.Tensor,
		encoder_out_lens: torch.Tensor,
		ys_pad: torch.Tensor,
		ys_pad_lens: torch.Tensor,
	):
		encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
			encoder_out.device)
		if self.predictor_bias == 1:
			_, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
			ys_pad_lens = ys_pad_lens + self.predictor_bias
		mask_chunk_predictor = None
		if self.encoder.overlap_chunk_cls is not None:
			mask_chunk_predictor = self.encoder.overlap_chunk_cls.get_mask_chunk_predictor(None,
			                                                                               device=encoder_out.device,
			                                                                               batch_size=encoder_out.size(
				                                                                               0))
			mask_shfit_chunk = self.encoder.overlap_chunk_cls.get_mask_shfit_chunk(None, device=encoder_out.device,
			                                                                       batch_size=encoder_out.size(0))
			encoder_out = encoder_out * mask_shfit_chunk
		pre_acoustic_embeds, pre_token_length, pre_alphas, _ = self.predictor(encoder_out,
		                                                                      ys_pad,
		                                                                      encoder_out_mask,
		                                                                      ignore_id=self.ignore_id,
		                                                                      mask_chunk_predictor=mask_chunk_predictor,
		                                                                      target_label_length=ys_pad_lens,
		                                                                      )
		predictor_alignments, predictor_alignments_len = self.predictor.gen_frame_alignments(pre_alphas,
		                                                                                     encoder_out_lens)
		
		scama_mask = None
		if self.encoder.overlap_chunk_cls is not None and self.decoder_attention_chunk_type == 'chunk':
			encoder_chunk_size = self.encoder.overlap_chunk_cls.chunk_size_pad_shift_cur
			attention_chunk_center_bias = 0
			attention_chunk_size = encoder_chunk_size
			decoder_att_look_back_factor = self.encoder.overlap_chunk_cls.decoder_att_look_back_factor_cur
			mask_shift_att_chunk_decoder = self.encoder.overlap_chunk_cls. \
				get_mask_shift_att_chunk_decoder(None,
			                                     device=encoder_out.device,
			                                     batch_size=encoder_out.size(0)
			                                     )
			scama_mask = self.build_scama_mask_for_cross_attention_decoder_fn(
				predictor_alignments=predictor_alignments,
				encoder_sequence_length=encoder_out_lens,
				chunk_size=1,
				encoder_chunk_size=encoder_chunk_size,
				attention_chunk_center_bias=attention_chunk_center_bias,
				attention_chunk_size=attention_chunk_size,
				attention_chunk_type=self.decoder_attention_chunk_type,
				step=None,
				predictor_mask_chunk_hopping=mask_chunk_predictor,
				decoder_att_look_back_factor=decoder_att_look_back_factor,
				mask_shift_att_chunk_decoder=mask_shift_att_chunk_decoder,
				target_length=ys_pad_lens,
				is_training=self.training,
			)
		elif self.encoder.overlap_chunk_cls is not None:
			encoder_out, encoder_out_lens = self.encoder.overlap_chunk_cls.remove_chunk(encoder_out,
			                                                                            encoder_out_lens,
			                                                                            chunk_outs=None)
		# 0. sampler
		decoder_out_1st = None
		pre_loss_att = None
		if self.sampling_ratio > 0.0:
			if self.step_cur < 2:
				logging.info("enable sampler in paraformer, sampling_ratio: {}".format(self.sampling_ratio))
			if self.use_1st_decoder_loss:
				sematic_embeds, decoder_out_1st, pre_loss_att = \
					self.sampler_with_grad(encoder_out, encoder_out_lens, ys_pad,
					                       ys_pad_lens, pre_acoustic_embeds, scama_mask)
			else:
				sematic_embeds, decoder_out_1st = \
					self.sampler(encoder_out, encoder_out_lens, ys_pad,
					             ys_pad_lens, pre_acoustic_embeds, scama_mask)
		else:
			if self.step_cur < 2:
				logging.info("disable sampler in paraformer, sampling_ratio: {}".format(self.sampling_ratio))
			sematic_embeds = pre_acoustic_embeds
		
		# 1. Forward decoder
		decoder_outs = self.decoder(
			encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens, scama_mask
		)
		decoder_out, _ = decoder_outs[0], decoder_outs[1]
		
		if decoder_out_1st is None:
			decoder_out_1st = decoder_out
		# 2. Compute attention loss
		loss_att = self.criterion_att(decoder_out, ys_pad)
		acc_att = th_accuracy(
			decoder_out_1st.view(-1, self.vocab_size),
			ys_pad,
			ignore_label=self.ignore_id,
		)
		loss_pre = self.criterion_pre(ys_pad_lens.type_as(pre_token_length), pre_token_length)
		
		# Compute cer/wer using attention-decoder
		if self.training or self.error_calculator is None:
			cer_att, wer_att = None, None
		else:
			ys_hat = decoder_out_1st.argmax(dim=-1)
			cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())
		
		return loss_att, acc_att, cer_att, wer_att, loss_pre, pre_loss_att
	
	def sampler(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, pre_acoustic_embeds, chunk_mask=None):
		
		tgt_mask = (~make_pad_mask(ys_pad_lens, maxlen=ys_pad_lens.max())[:, :, None]).to(ys_pad.device)
		ys_pad_masked = ys_pad * tgt_mask[:, :, 0]
		if self.share_embedding:
			ys_pad_embed = self.decoder.output_layer.weight[ys_pad_masked]
		else:
			ys_pad_embed = self.decoder.embed(ys_pad_masked)
		with torch.no_grad():
			decoder_outs = self.decoder(
				encoder_out, encoder_out_lens, pre_acoustic_embeds, ys_pad_lens, chunk_mask
			)
			decoder_out, _ = decoder_outs[0], decoder_outs[1]
			pred_tokens = decoder_out.argmax(-1)
			nonpad_positions = ys_pad.ne(self.ignore_id)
			seq_lens = (nonpad_positions).sum(1)
			same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
			input_mask = torch.ones_like(nonpad_positions)
			bsz, seq_len = ys_pad.size()
			for li in range(bsz):
				target_num = (((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio).long()
				if target_num > 0:
					input_mask[li].scatter_(dim=0, index=torch.randperm(seq_lens[li])[:target_num].cuda(), value=0)
			input_mask = input_mask.eq(1)
			input_mask = input_mask.masked_fill(~nonpad_positions, False)
			input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)
		
		sematic_embeds = pre_acoustic_embeds.masked_fill(~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
			input_mask_expand_dim, 0)
		return sematic_embeds * tgt_mask, decoder_out * tgt_mask
	

	def calc_predictor(self, encoder_out, encoder_out_lens):
		
		encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
			encoder_out.device)
		mask_chunk_predictor = None
		if self.encoder.overlap_chunk_cls is not None:
			mask_chunk_predictor = self.encoder.overlap_chunk_cls.get_mask_chunk_predictor(None,
			                                                                               device=encoder_out.device,
			                                                                               batch_size=encoder_out.size(
				                                                                               0))
			mask_shfit_chunk = self.encoder.overlap_chunk_cls.get_mask_shfit_chunk(None, device=encoder_out.device,
			                                                                       batch_size=encoder_out.size(0))
			encoder_out = encoder_out * mask_shfit_chunk
		pre_acoustic_embeds, pre_token_length, pre_alphas, pre_peak_index = self.predictor(encoder_out,
		                                                                                   None,
		                                                                                   encoder_out_mask,
		                                                                                   ignore_id=self.ignore_id,
		                                                                                   mask_chunk_predictor=mask_chunk_predictor,
		                                                                                   target_label_length=None,
		                                                                                   )
		predictor_alignments, predictor_alignments_len = self.predictor.gen_frame_alignments(pre_alphas,
		                                                                                     encoder_out_lens + 1 if self.predictor.tail_threshold > 0.0 else encoder_out_lens)
		
		scama_mask = None
		if self.encoder.overlap_chunk_cls is not None and self.decoder_attention_chunk_type == 'chunk':
			encoder_chunk_size = self.encoder.overlap_chunk_cls.chunk_size_pad_shift_cur
			attention_chunk_center_bias = 0
			attention_chunk_size = encoder_chunk_size
			decoder_att_look_back_factor = self.encoder.overlap_chunk_cls.decoder_att_look_back_factor_cur
			mask_shift_att_chunk_decoder = self.encoder.overlap_chunk_cls. \
				get_mask_shift_att_chunk_decoder(None,
			                                     device=encoder_out.device,
			                                     batch_size=encoder_out.size(0)
			                                     )
			scama_mask = self.build_scama_mask_for_cross_attention_decoder_fn(
				predictor_alignments=predictor_alignments,
				encoder_sequence_length=encoder_out_lens,
				chunk_size=1,
				encoder_chunk_size=encoder_chunk_size,
				attention_chunk_center_bias=attention_chunk_center_bias,
				attention_chunk_size=attention_chunk_size,
				attention_chunk_type=self.decoder_attention_chunk_type,
				step=None,
				predictor_mask_chunk_hopping=mask_chunk_predictor,
				decoder_att_look_back_factor=decoder_att_look_back_factor,
				mask_shift_att_chunk_decoder=mask_shift_att_chunk_decoder,
				target_length=None,
				is_training=self.training,
			)
		self.scama_mask = scama_mask
		
		return pre_acoustic_embeds, pre_token_length, pre_alphas, pre_peak_index
	
	def calc_predictor_chunk(self, encoder_out, cache=None):
		
		pre_acoustic_embeds, pre_token_length = \
			self.predictor.forward_chunk(encoder_out, cache["encoder"])
		return pre_acoustic_embeds, pre_token_length
	
	def cal_decoder_with_predictor(self, encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens):
		decoder_outs = self.decoder(
			encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens, self.scama_mask
		)
		decoder_out = decoder_outs[0]
		decoder_out = torch.log_softmax(decoder_out, dim=-1)
		return decoder_out, ys_pad_lens
	
	def cal_decoder_with_predictor_chunk(self, encoder_out, sematic_embeds, cache=None):
		decoder_outs = self.decoder.forward_chunk(
			encoder_out, sematic_embeds, cache["decoder"]
		)
		decoder_out = decoder_outs
		decoder_out = torch.log_softmax(decoder_out, dim=-1)
		return decoder_out

	def generate(self,
	             speech: torch.Tensor,
	             speech_lengths: torch.Tensor,
	             tokenizer=None,
	             **kwargs,
	             ):
		
		is_use_ctc = kwargs.get("ctc_weight", 0.0) > 0.00001 and self.ctc != None
		print(is_use_ctc)
		is_use_lm = kwargs.get("lm_weight", 0.0) > 0.00001 and kwargs.get("lm_file", None) is not None
		
		if self.beam_search is None and (is_use_lm or is_use_ctc):
			logging.info("enable beam_search")
			self.init_beam_search(speech, speech_lengths, **kwargs)
			self.nbest = kwargs.get("nbest", 1)
		
		# Forward Encoder
		encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
		if isinstance(encoder_out, tuple):
			encoder_out = encoder_out[0]
		
		# predictor
		predictor_outs = self.calc_predictor(encoder_out, encoder_out_lens)
		pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = predictor_outs[0], predictor_outs[1], \
		                                                                predictor_outs[2], predictor_outs[3]
		pre_token_length = pre_token_length.round().long()
		if torch.max(pre_token_length) < 1:
			return []
		decoder_outs = self.cal_decoder_with_predictor(encoder_out, encoder_out_lens, pre_acoustic_embeds,
		                                               pre_token_length)
		decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]
		
		results = []
		b, n, d = decoder_out.size()
		for i in range(b):
			x = encoder_out[i, :encoder_out_lens[i], :]
			am_scores = decoder_out[i, :pre_token_length[i], :]
			if self.beam_search is not None:
				nbest_hyps = self.beam_search(
					x=x, am_scores=am_scores, maxlenratio=kwargs.get("maxlenratio", 0.0),
					minlenratio=kwargs.get("minlenratio", 0.0)
				)
				
				nbest_hyps = nbest_hyps[: self.nbest]
			else:
				
				yseq = am_scores.argmax(dim=-1)
				score = am_scores.max(dim=-1)[0]
				score = torch.sum(score, dim=-1)
				# pad with mask tokens to ensure compatibility with sos/eos tokens
				yseq = torch.tensor(
					[self.sos] + yseq.tolist() + [self.eos], device=yseq.device
				)
				nbest_hyps = [Hypothesis(yseq=yseq, score=score)]
			for hyp in nbest_hyps:
				assert isinstance(hyp, (Hypothesis)), type(hyp)
				
				# remove sos/eos and get results
				last_pos = -1
				if isinstance(hyp.yseq, list):
					token_int = hyp.yseq[1:last_pos]
				else:
					token_int = hyp.yseq[1:last_pos].tolist()
				
				# remove blank symbol id, which is assumed to be 0
				token_int = list(filter(lambda x: x != 0 and x != 2, token_int))
				
				# Change integer-ids to tokens
				token = tokenizer.ids2tokens(token_int)
				text = tokenizer.tokens2text(token)
				
				timestamp = []
				
				results.append((text, token, timestamp))
		
		return results

