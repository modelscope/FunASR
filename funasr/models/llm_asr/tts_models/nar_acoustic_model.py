import logging
from typing import List, Tuple, Dict, Optional, Union
import torch
import torch.nn as nn
from funasr.models.transformer.utils.nets_utils import make_pad_mask
import torch.nn.functional as F
from funasr.train_utils.device_funcs import force_gatherable
from funasr.models.llm_asr.label_smoothing_loss import LabelSmoothingLoss
from copy import deepcopy
from funasr.metrics.compute_acc import th_accuracy
from funasr.models.transformer.utils.nets_utils import pad_list
import random
import numpy as np
from funasr.utils.hinter import hint_once
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.models.llm_asr.tts_models.ctc_alignment import ctc_forced_align
from torch.nn.utils.rnn import pad_sequence
import itertools
from distutils.version import LooseVersion
from contextlib import contextmanager
if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class NARCTCModel(nn.Module):
    def __init__(
            self,
            input_size: int,
            vocab_size: int,
            encoder: Union[nn.Module, dict],
            decoder: Optional[nn.Module] = None,
            ctc_weight: float = 0.5,
            ignore_id: int = -1,
            lsm_weight: float = 0.0,
            length_normalized_loss: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.decoder = decoder
        self.encoder = encoder if isinstance(encoder, nn.Module) else self.build_encoder(encoder)
        self.output_size = self.encoder.output_size()
        self.ignore_id = ignore_id
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight

        # build ctc module
        from funasr.models.llm_asr.tts_models.ctc_alignment import CTC
        ctc_conf = kwargs.pop("ctc_conf", {})
        self.ctc = CTC(vocab_size, encoder_output_size=self.output_size, **ctc_conf)

        self.text_embedding = torch.nn.Embedding(self.vocab_size, input_size)
        self.token_embedding = torch.nn.Embedding(vocab_size, input_size)
        xvec_size = kwargs.get("xvec_size", None)
        if xvec_size is not None:
            self.xvec_proj = torch.nn.Linear(xvec_size, input_size)
        else:
            self.xvec_proj = None
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.sos = vocab_size - 2
        self.eos = vocab_size - 1
        self.length_regulator_conf = kwargs.get("length_regulator_conf", None)
        if self.length_regulator_conf is not None:
            self.length_regulator = self.build_length_regulator()
        else:
            self.length_regulator = None

    def build_encoder(self, encoder_conf: dict):
        if encoder_conf is None:
            assert hasattr(self, "encoder_conf"), \
                "function param encoder_conf is None and model doesn't has encoder_conf attribute either."
            encoder_conf = self.encoder_conf

        encoder_name = encoder_conf.pop("name", "transformer")
        model = None
        if encoder_name == "transformer":
            from funasr.models.llm_asr.conformer_encoder import ConformerEncoder
            model = ConformerEncoder(
                **encoder_conf,
                input_size=self.input_size,
                use_cnn_module=False,
                macaron_style=False,
            )
        elif encoder_name == "conformer":
            from funasr.models.llm_asr.conformer_encoder import ConformerEncoder
            model = ConformerEncoder(
                **encoder_conf,
                input_size=self.input_size,
            )
        elif encoder_name == "upsampling_conformer":
            from funasr.models.llm_asr.tts_models.encoders import UpsampleConformerEncoder
            model = UpsampleConformerEncoder(
                **encoder_conf,
                input_size=self.input_size,
            )

        encoder_conf["name"] = encoder_name

        return model

    def build_length_regulator(self):
        name = self.length_regulator_conf.pop("name", None)
        model = None
        if name == "upsampling":
            from funasr.models.llm_asr.diffusion_models.length_regulator import UpSamplingRegulator
            model = UpSamplingRegulator(self.input_size, self.length_regulator_conf.get("sampling_ratios"))
        elif name == "downsampling":
            from funasr.models.llm_asr.diffusion_models.length_regulator import DownSamplingRegulator
            model = DownSamplingRegulator(self.input_size, self.length_regulator_conf.get("sampling_ratios"))
        elif name == "interpolate":
            from funasr.models.llm_asr.diffusion_models.length_regulator import InterpolateRegulator
            model = InterpolateRegulator(self.input_size, **self.length_regulator_conf)
        elif name == "upsampling_cif":
            from funasr.models.llm_asr.diffusion_models.length_regulator import UpsamplingCifRegulator
            model = UpsamplingCifRegulator(self.input_size, **self.length_regulator_conf)

        self.length_regulator_conf["name"] = name

        return model

    @staticmethod
    def norm_and_sample_xvec(xvec, xvec_lengths):
        xvec_list = []
        for i, ilen in enumerate(xvec_lengths):
            idx = random.randint(0, ilen - 1)
            while torch.any(~torch.isfinite(xvec[i, idx])):
                idx = random.randint(0, ilen - 1)
            xvec_list.append(xvec[i, idx])
        rand_xvec = torch.vstack(xvec_list)
        rand_xvec = F.normalize(rand_xvec, dim=1)

        return rand_xvec

    def _calc_att_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.decoder.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def model_forward(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            xvec: Optional[torch.Tensor] = None,
            xvec_lengths: Optional[torch.Tensor] = None,
            **kwargs
    ):
        # 0. Up-sampling text length
        if self.length_regulator is not None:
            text, text_lengths = self.length_regulator(text, text_lengths)

        # 1. padding xvec
        if xvec is not None and self.xvec_proj is not None:
            xvec = xvec[:, :xvec_lengths.max()]
            # random select a xvec from xvec matrix
            xvec = self.norm_and_sample_xvec(xvec, xvec_lengths)
            xvec = self.xvec_proj(xvec)
            text = text + xvec.unsqueeze(1)
            hint_once("use xvec", "use_xvec")

        # 1. Encoder
        encoder_out, encoder_out_lens, _ = self.encoder(text, text_lengths)

        return encoder_out, encoder_out_lens

    def predictor(
            self,
            am: torch.Tensor,
            am_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
            alignment,
    ):
        acoustic_embeds = []
        use_pred_num = 0
        for am_xs, enc_len, ali, y, y_lens in zip(am, am_lens, alignment, ys_pad, ys_pad_lens):
            pred = itertools.groupby(ali[:enc_len])

            acoustic_embed = []
            _start = 0
            for pred_token, pred_frame in pred:
                _end = _start + len(list(pred_frame))
                if pred_token != 0:
                    acoustic_embed.append(torch.mean(am_xs[_start:_end, :], 0, keepdim=True))
                _start = _end
            if len(acoustic_embed) != y_lens:
                acoustic_embeds.append(y[:y_lens])
            else:
                acoustic_embeds.append(torch.cat(acoustic_embed, dim=0))
                use_pred_num += 1
        acoustic_embeds = pad_sequence(acoustic_embeds, batch_first=True, padding_value=0)
        return acoustic_embeds, use_pred_num / am.shape[0]

    def force_align_text(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            xvec: Optional[torch.Tensor] = None,
            xvec_lengths: Optional[torch.Tensor] = None,
            **kwargs
    ):
        # plus one to speech token, to make index 0 represent <blank>,
        # decoder vocab must be: 1 (blank) + num of token + 1 (sos) + 1 (eos)
        speech = torch.where(speech != -1, speech + 1, speech)

        encoder_out, encoder_out_lens = self.model_forward(
            text, text_lengths,
            xvec, xvec_lengths,
            **kwargs
        )
        log_probs = self.ctc.log_softmax(encoder_out)
        with torch.no_grad():
            alignment = ctc_forced_align(
                log_probs.float(),
                speech.long(),
                encoder_out_lens.long(),
                speech_lengths.long(),
                ignore_id=self.ignore_id,
            )
        aligned_token_emb, use_pred_ratio = self.predictor(
            encoder_out, encoder_out_lens,
            self.token_embedding(speech), speech_lengths,
            alignment,
        )

        loss = 0
        states = dict(
            use_pred_ratio=use_pred_ratio,
        )
        if self.ctc_weight != 0.0:
            loss_ctc, logits = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, speech, speech_lengths
            )
            states["loss_ctc"] = loss_ctc.item()
            loss = loss + self.ctc_weight * loss_ctc
        if self.ctc_weight != 1.0:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, speech, speech_lengths
            )
            states["loss_att"] = loss_att.item()
            loss = loss + (1.0 - self.ctc_weight) * loss_att

        states["loss"] = loss.item()
        return loss, aligned_token_emb, states

    def _calc_ctc_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        logits = self.ctc.log_softmax(encoder_out)

        return loss_ctc, logits

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            xvec: Optional[torch.Tensor] = None,
            xvec_lengths: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

                Args:
                    speech: (Batch, Length, ...), speech tokens
                    speech_lengths: (Batch, )
                    text: (Batch, Length), text tokens
                    text_lengths: (Batch, )
                    xvec: (Batch, Length, ...) x-vectors
                    xvec_lengths: (Batch, )
                """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
                speech.shape[0]
                == speech_lengths.shape[0]
                == text.shape[0]
                == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]
        speech = speech[:, : speech_lengths.max()]
        # plus one to speech token, to make index 0 represent <blank>,
        # decoder vocab must be: 1 (blank) + num of token + 1 (sos) + 1 (eos)
        speech = torch.where(speech != -1, speech + 1, speech)

        # embed text inputs
        mask = (text != -1).float().unsqueeze(-1)
        text = self.text_embedding(torch.clamp(text, min=0)) * mask

        encoder_out, encoder_out_lens = self.model_forward(
            text, text_lengths,
            xvec, xvec_lengths,
            **kwargs,
        )

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        stats = dict(
            batch_size=float(batch_size),
            text_len=float(text.shape[1]),
            enc_len=float(encoder_out.shape[1]),
            speech_len=float(speech.shape[1]),
            token_text_ratio=float(speech.shape[1])/float(text.shape[1]),
        )

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, logits = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, speech, speech_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # 2b. Attention decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, speech, speech_lengths
            )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def topp_sampling(self, probs, top_p=0.8):
        sorted_value, sorted_idx = probs.sort(descending=True, stable=True)
        cumulative_probs = torch.cumsum(sorted_value, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_idx[sorted_indices_to_remove]
        probs[indices_to_remove] = 0
        top_ids = torch.multinomial(probs, num_samples=1)

        return top_ids

    def sampling_ids(self, enc_outs, sampling="greedy", blank_penalty=None, return_probs=False):
        probs = self.ctc.softmax(enc_outs)
        if blank_penalty > 0:
            probs[:, :, 0] = probs[:, :, 0] * blank_penalty

        # top-p sampling
        if "." in sampling:
            sampling = float(sampling)
            tokens = self.topp_sampling(probs, top_p=sampling)
            tokens = torch.tensor(tokens, dtype=torch.long).to(probs.device)
        # top-k sampling
        elif sampling.isdigit():
            sampling = int(sampling)
            probs = probs.topk(sampling)
            tokens = probs.multinomial(1, replacement=True)
        else:
            if sampling == "greedy":
                tokens = torch.argmax(probs, dim=-1)
            elif "threshold_" in sampling:
                threshold = float(sampling.split("_")[1])
                hint_once(f"Decoding mode: blank threshold={threshold:.2f}", "decoding_mode")
                # mask out blank according to threshold
                mask = probs[:, :, 0] > threshold
                probs[:, :, 0] = probs[:, :, 0] * mask
                tokens = torch.argmax(probs, dim=-1)
            else:
                raise NotImplementedError(f"sampling method {sampling} not implemented")

        if not return_probs:
            return tokens

        return tokens, probs

    def inference(
            self,
            text: torch.Tensor, text_lengths: torch.Tensor,
            xvec=None, xvec_lengths=None,
            sampling="greedy",
            blank_penalty: float = 0.0,
            text_is_embedding=False,
            return_hidden=False,
            **kwargs,
    ):
        device = text.device
        # use casual mode at inference stage
        self.encoder.use_causal_prob = kwargs.get("use_causal_prob", 1.0)
        hint_once(f"use_causal_prob {self.encoder.use_causal_prob}.", "use_causal_prob")
        # embed text inputs
        if not text_is_embedding:
            mask = (text != -1).float().unsqueeze(-1)
            text = self.text_embedding(torch.clamp(text, min=0)) * mask

        # 1. Encoder
        encoder_out, encoder_out_lens = self.model_forward(
            text, text_lengths,
            xvec, xvec_lengths,
        )
        fa_tokens, enc_probs = self.sampling_ids(
            encoder_out,
            sampling=sampling,
            blank_penalty=blank_penalty,
            return_probs=True,
        )
        reduced_fa_tokens = []
        for pred_token, pred_frame in itertools.groupby(fa_tokens[0].cpu().tolist()):
            if pred_token != 0:
                reduced_fa_tokens.append(pred_token)
            else:
                reduced_fa_tokens.extend(list(pred_frame))
        fa_tokens = torch.tensor(reduced_fa_tokens).to(fa_tokens)
        # remove blanks (id=0) and convert token ids into the original format
        tokens = [[x-1] for x in fa_tokens[0].cpu().tolist() if x > 0]
        tokens = torch.tensor([tokens], dtype=torch.int64, device=device)

        if not return_hidden:
            return tokens

        acoustic_embs, acoustic_emb_lens = [], []
        for idx, (prob, enc) in enumerate(zip(enc_probs, encoder_out)):
            pred = itertools.groupby(prob.argmax(-1).cpu())
            acs_emb = []
            _start = 0
            for pred_token, pred_frame in pred:
                _end = _start + len(list(pred_frame))
                if pred_token != 0 and pred_token != -1:
                    acs_emb.append(torch.mean(enc[_start:_end, :], 0, keepdim=True))
                _start = _end
            acs_emb = torch.cat(acs_emb, dim=0)
            acoustic_embs.append(acs_emb)
            acoustic_emb_lens.append(acs_emb.shape[0])

        acoustic_embs = pad_list(acoustic_embs, 0.0)
        acoustic_emb_lens = torch.tensor(acoustic_emb_lens, dtype=torch.int64, device=device)

        return (tokens, fa_tokens), acoustic_embs, acoustic_emb_lens


class NARCTCProbModel(NARCTCModel):
    def __init__(self, input_size: int, vocab_size: int, encoder: Union[nn.Module, dict],
                 decoder: Optional[nn.Module] = None, ctc_weight: float = 0.5, ignore_id: int = -1,
                 lsm_weight: float = 0.0, length_normalized_loss: bool = False, **kwargs):
        super().__init__(input_size, vocab_size, encoder, decoder, ctc_weight, ignore_id, lsm_weight,
                         length_normalized_loss, **kwargs)

    def predictor(
            self,
            am_probs: torch.Tensor,
            am_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
            alignment,
    ):
        acoustic_embeds = []
        use_pred_num = 0
        for probs, enc_len, ali, y, y_lens in zip(am_probs, am_lens, alignment, ys_pad, ys_pad_lens):
            pred = itertools.groupby(ali[:enc_len])

            acoustic_embed = []
            _start = 0
            for pred_token, pred_frame in pred:
                _end = _start + len(list(pred_frame))
                if pred_token != 0:
                    acoustic_embed.append(torch.mean(probs[_start:_end, :], 0, keepdim=True))
                _start = _end
            if len(acoustic_embed) != y_lens:
                acoustic_embeds.append(F.one_hot(y[:y_lens], self.vocab_size).float())
            else:
                acoustic_embeds.append(torch.cat(acoustic_embed, dim=0))
                use_pred_num += 1
            acoustic_embeds[-1] = torch.matmul(acoustic_embeds[-1], self.token_embedding.weight)
        acoustic_embeds = pad_sequence(acoustic_embeds, batch_first=True, padding_value=0)
        return acoustic_embeds, use_pred_num / am_probs.shape[0]

    def force_align_text(self, speech: torch.Tensor, speech_lengths: torch.Tensor, text: torch.Tensor,
                         text_lengths: torch.Tensor, xvec: Optional[torch.Tensor] = None,
                         xvec_lengths: Optional[torch.Tensor] = None, **kwargs):
        # plus one to speech token, to make index 0 represent <blank>,
        # decoder vocab must be: 1 (blank) + num of token + 1 (sos) + 1 (eos)
        speech = torch.where(speech != -1, speech + 1, speech)

        encoder_out, encoder_out_lens = self.model_forward(
            text, text_lengths,
            xvec, xvec_lengths,
            **kwargs
        )
        log_probs = self.ctc.log_softmax(encoder_out)
        with torch.no_grad():
            alignment = ctc_forced_align(
                log_probs.float(),
                speech.long(),
                encoder_out_lens.long(),
                speech_lengths.long(),
                ignore_id=self.ignore_id,
            )
        aligned_token_emb, use_pred_ratio = self.predictor(
            log_probs.float(), encoder_out_lens.long(),
            speech.long(), speech_lengths.long(),
            alignment,
        )

        loss = 0
        states = dict(
            use_pred_ratio=use_pred_ratio,
        )
        if self.ctc_weight != 0.0:
            loss_ctc, logits = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, speech, speech_lengths
            )
            states["loss_ctc"] = loss_ctc.item()
            loss = loss + self.ctc_weight * loss_ctc
        if self.ctc_weight != 1.0:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, speech, speech_lengths
            )
            states["loss_att"] = loss_att.item()
            loss = loss + (1.0 - self.ctc_weight) * loss_att

        states["loss"] = loss.item()
        return loss, aligned_token_emb, states

    def inference(self, text: torch.Tensor, text_lengths: torch.Tensor, xvec=None, xvec_lengths=None, sampling="greedy",
                  blank_penalty: float = 0.0, text_is_embedding=False, return_hidden=False, **kwargs):
        device = text.device
        # embed text inputs
        if not text_is_embedding:
            mask = (text != -1).float().unsqueeze(-1)
            text = self.text_embedding(torch.clamp(text, min=0)) * mask

        # 0. Up-sampling text length
        if self.length_regulator is not None:
            text, text_lengths = self.length_regulator(text, text_lengths)

        # 1. padding xvec
        if xvec is not None and self.xvec_proj is not None:
            xvec = xvec[:, :xvec_lengths.max()]
            # random select a xvec from xvec matrix
            xvec = self.norm_and_sample_xvec(xvec, xvec_lengths)
            xvec = self.xvec_proj(xvec)
            text = text + xvec.unsqueeze(1)
            hint_once("use xvec", "use_xvec")

        # 1. Encoder
        encoder_out, encoder_out_lens = self.model_forward(
            text, text_lengths,
            xvec, xvec_lengths,
        )
        tokens, enc_probs = self.sampling_ids(
            encoder_out,
            sampling=sampling,
            blank_penalty=blank_penalty,
            return_probs=True,
        )
        # remove blanks (id=0) and convert token ids into the original format
        tokens = [[x - 1] for x in tokens[0].cpu().tolist() if x > 0]
        tokens = torch.tensor([tokens], dtype=torch.int64, device=device)

        if not return_hidden:
            return tokens

        acoustic_embs = self.token_embedding(tokens.squeeze(-1))
        acoustic_emb_lens = torch.tensor([acoustic_embs.shape[1]], dtype=torch.int64, device=device)

        return tokens, acoustic_embs, acoustic_emb_lens
