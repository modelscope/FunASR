#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import time
import copy
import torch
import logging
from torch.cuda.amp import autocast
from typing import Union, Dict, List, Tuple, Optional

from funasr.register import tables
from funasr.models.ctc.ctc import CTC
from funasr.utils import postprocess_utils
from funasr.metrics.compute_acc import th_accuracy
from funasr.train_utils.device_funcs import to_device
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.paraformer.search import Hypothesis
from funasr.models.paraformer.cif_predictor import mae_loss
from funasr.train_utils.device_funcs import force_gatherable
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.utils.timestamp_tools import ts_prediction_lfr6_standard
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from torch.nn.utils.rnn import pad_sequence
import torchaudio

@tables.register("model_classes", "Paraformer_v2_community")
class Paraformer(torch.nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
        self,
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
        share_embedding: bool = False,
        # preencoder: Optional[AbsPreEncoder] = None,
        # postencoder: Optional[AbsPostEncoder] = None,
        use_1st_decoder_loss: bool = False,
        **kwargs,
    ):

        super().__init__()

        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)
        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        if decoder is not None:
            decoder_class = tables.decoder_classes.get(decoder)
            decoder = decoder_class(
                vocab_size=vocab_size,
                encoder_output_size=encoder_output_size,
                **decoder_conf,
            )
        if ctc_weight > 0.0:

            if ctc_conf is None:
                ctc_conf = {}

            ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)

        # note that eos is the same as sos (equivalent ID)
        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        # self.token_list = token_list.copy()
        #
        # self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        # self.preencoder = preencoder
        # self.postencoder = postencoder
        self.encoder = encoder
        #
        # if not hasattr(self.encoder, "interctc_use_conditioning"):
        #     self.encoder.interctc_use_conditioning = False
        # if self.encoder.interctc_use_conditioning:
        #     self.encoder.conditioning_layer = torch.nn.Linear(
        #         vocab_size, self.encoder.output_size()
        #     )
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
        #     self.error_calculator = ErrorCalculator(
        #         token_list, sym_space, sym_blank, report_cer, report_wer
        #     )
        #
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        #
        # self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        self.share_embedding = share_embedding
        if self.share_embedding:
            self.decoder.embed = None

        self.use_1st_decoder_loss = use_1st_decoder_loss
        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None
        self.error_calculator = None

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
        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths
        )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att
        else:
            loss = (
                self.ctc_weight * loss_ctc
                + (1 - self.ctc_weight) * loss_att
            )

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = (text_lengths).sum()
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder. Note that this method is used by asr_inference.py
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

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):

        # 0. sampler
        decoder_out_1st = None

        batch_size = encoder_out.size(0)
        ctc_probs_all = self.ctc.softmax(encoder_out)
        compressed_ctc_list = []
        for b in range(batch_size):
            ctc_prob_b = ctc_probs_all[b, :encoder_out_lens[b]]
            text_b = ys_pad[b, :ys_pad_lens[b]]
            with torch.no_grad():
                ctc_log_prob_b = ctc_prob_b.log()
                align_path = self.force_align(ctc_log_prob_b.cpu(), text_b.cpu(), blank_id=self.blank_id)
                align_path = align_path.to(encoder_out.device)
                target_idx_path = self.map_alignment_to_target_index(align_path, self.blank_id)
            ctc_comp = self.average_repeats_training(ctc_prob_b, target_idx_path, ys_pad_lens[b])
            compressed_ctc_list.append(ctc_comp)

        # 4. Pad Batch to [B, U_max, V]
        padded_ctc_input = pad_sequence(compressed_ctc_list, batch_first=True).to(encoder_out.device)



        # 1. Forward decoder
        decoder_outs = self.decoder(encoder_out, encoder_out_lens, padded_ctc_input, ys_pad_lens)
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

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out_1st.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att


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
    
    def map_alignment_to_target_index(self, align_path, blank_id):
        """
        Robustly map CTC alignment path (Token IDs) to Target Indices.
        
        Logic:
            Detect boundaries where a new token segment begins.
            A segment starts if the current frame is a Token AND it is different from the previous frame 
            (considering CTC topology where repeats are separated by blanks or are distinct tokens).
        
        Example:
            Text: [A, B]
            Align Path: [A, A, _, B, B]
            Output:     [0, 0, -1, 1, 1]
        """
        # 1. Identify where the path is NOT blank
        is_token = align_path != blank_id
        
        # 2. Identify transitions
        prev_path = torch.roll(align_path, 1)
        # Handle the very first frame: if it's a token, it must be the start of segment 0.
        prev_path[0] = blank_id # force mismatch for the first element
        
        # A new segment starts if: It's a token AND (it differs from prev OR prev was blank)
        # Note: If align_path[i] == align_path[i-1] (and not blank), it's the same segment.
        new_segment_start = is_token & (align_path != prev_path)
        
        # 3. Cumulative sum to assign indices (1..U)
        segment_ids = torch.cumsum(new_segment_start.long(), dim=0) - 1
        
        # 4. Mask out blank positions with -1
        target_idx_path = torch.where(is_token, segment_ids, -1)
        
        return target_idx_path
    def force_align(self, ctc_probs: torch.Tensor, y: torch.Tensor, blank_id=0) -> list:
        """ctc forced alignment.

        Args:
            torch.Tensor ctc_probs: hidden state sequence, 2d tensor (T, D)
            torch.Tensor y: id sequence tensor 1d tensor (L)
            int blank_id: blank symbol index
        Returns:
            torch.Tensor: alignment result
        """
        ctc_probs = ctc_probs[None].cpu()
        y = y[None].cpu()
        alignments, _ = torchaudio.functional.forced_align(ctc_probs, y, blank=blank_id)
        return alignments[0]

    def average_repeats_training(self, ctc_probs, target_idx_path, target_len):
        """
        Aggregates frames belonging to the same target index using scatter_add.
        
        Args:
            ctc_probs: [T, V]
            target_idx_path: [T], values in [-1, 0, ... U-1]
            target_len: U
        Returns:
            compressed: [U, V]
        """
        U = target_len
        V = ctc_probs.size(1)
        
        compressed = torch.zeros((U, V), device=ctc_probs.device, dtype=ctc_probs.dtype)
        counts = torch.zeros((U, 1), device=ctc_probs.device, dtype=ctc_probs.dtype)
        
        # Filter valid frames (non-blank)
        mask = target_idx_path != -1
        valid_indices = target_idx_path[mask] # [T_valid]
        valid_probs = ctc_probs[mask]         # [T_valid, V]
        
        if valid_indices.numel() == 0:
            return compressed
            
        # Scatter Add Probs
        index_expanded = valid_indices.unsqueeze(1).repeat(1, V)
        compressed.scatter_add_(0, index_expanded, valid_probs)
        
        # Scatter Add Counts
        ones = torch.ones((valid_indices.size(0), 1), device=ctc_probs.device)
        counts.scatter_add_(0, valid_indices.unsqueeze(1), ones)
        
        # Average
        compressed = compressed / (counts + 1e-9)
        return compressed
    
    def average_repeats_inference(self, ctc_probs, greedy_path):
        """
        Returns:
            merged_probs: [U', V]
            timestamps: List[Tuple[int, int]] -> [(start_frame, end_frame), ...]
        """
        if greedy_path.numel() == 0:
            return torch.zeros((0, ctc_probs.size(1)), device=ctc_probs.device)

        # Find consecutive segments in the greedy path
        unique_tokens, counts = torch.unique_consecutive(greedy_path, return_counts=True)
        
        # Compute start and end indices for each segment
        end_indices = torch.cumsum(counts, dim=0)
        start_indices = torch.cat([torch.tensor([0], device=counts.device), end_indices[:-1]])
        
        merged_probs = []
    
        for i, token in enumerate(unique_tokens):
            if token != self.blank_id:
                start = start_indices[i].item()
                end = end_indices[i].item()
                
                # Extract and average probabilities for the decoder
                avg_prob = ctc_probs[start:end].mean(dim=0)
                merged_probs.append(avg_prob)

        
        if not merged_probs:
            return torch.zeros((0, ctc_probs.size(1)), device=ctc_probs.device)
            
        return torch.stack(merged_probs)

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):

        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is not None:
                speech_lengths = speech_lengths.squeeze(-1)
            else:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = (
                speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])
        # Encoder
        if kwargs.get("fp16", False):
            speech = speech.half()
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        ctc_probs = self.ctc.softmax(encoder_out)
        ctc_greedy_paths = ctc_probs.argmax(dim=-1)

        results = []
        batch_size, n, d = encoder_out.size()
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        if len(key) < batch_size:
            key = key * batch_size
        for b in range(batch_size):

            probs = ctc_probs[b, :encoder_out_lens[b]]
            path = ctc_greedy_paths[b, :encoder_out_lens[b]]
            
            # Get compressed probabilities and timestamp indices
            compressed_prob = self.average_repeats_inference(probs, path)
            
            # Handling Noise/Silence (Empty Output)
            if compressed_prob.size(0) == 0:
                token_int = []
            else:
                # 4. Decoder Forward
                compressed_prob_in = compressed_prob.unsqueeze(0) # [1, U', V]
                in_lens = torch.tensor([compressed_prob.size(0)], device=encoder_out.device)

                decoder_out, _ = self.decoder(
                        encoder_out[b:b+1],
                        encoder_out_lens[b:b+1],
                        compressed_prob_in,
                        in_lens,)

                
                yseq = decoder_out.argmax(dim=-1)[0]
                token_int = yseq.tolist()

                # remove blank symbol id, which is assumed to be 0
                token_int = list(
                    filter(
                        lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int
                    )
                )

                result_i = {"key": key[b], "token_int": token_int}
                results.append(result_i)

        return results, meta_data

    def export(self, **kwargs):
        from .export_meta import export_rebuild_model

        if "max_seq_len" not in kwargs:
            kwargs["max_seq_len"] = 512
        models = export_rebuild_model(model=self, **kwargs)
        return models
