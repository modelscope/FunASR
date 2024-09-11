import logging
from typing import Dict, Tuple

import torch
import torchaudio
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder, cuda_ctc_decoder
from torch.cuda.amp import autocast

from funasr.metrics.compute_acc import th_accuracy
from funasr.models.conformer.model import Conformer
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.register import tables
from funasr.train_utils.device_funcs import force_gatherable

import editdistance

@tables.register("model_classes", "ConformerDPO")
class ConformerDPO(Conformer):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.ref_model = Conformer(
            *args,
            **kwargs,
        )
        for p in self.ref_model.parameters():
            p.requires_grad = False
        # self.ref_model.load_state_dict(super().state_dict())

        self.nbest = kwargs.get("nbest", 5)
        self.beam_size = kwargs.get("beam_size", 10)
        self.mwer_rate = kwargs.get("mwer_rate", 0)
        char_list = [str(x) for x in range(self.vocab_size)]
        
        self.ctc_decoder = cuda_ctc_decoder(
            char_list, nbest = self.nbest, beam_size = self.beam_size, blank_skip_threshold = 0.95)

    def load_state_dict(self, state_dict, strict=True):
        missing_keys_1, unexpected_keys_1 = super().load_state_dict(state_dict, strict)
        missing_keys_2, unexpected_keys_2 = self.ref_model.load_state_dict(state_dict, strict)
        return (missing_keys_1 + missing_keys_2, unexpected_keys_1 + unexpected_keys_2) 
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

        # 1. Encoder
        encoder_out, encoder_out_lens, ref_encoder_out, ref_encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
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
        loss_att, acc_att, cer_att, wer_att, decoder_out = self._calc_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths
        )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        if self.mwer_rate > 0:
            mwer_loss = self._calc_mwer_loss(encoder_out, encoder_out_lens, decoder_out, text, text_lengths)
        else:
            mwer_loss = 0

        loss = loss + self.mwer_rate * mwer_loss
        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att
        stats["mwer"] = mwer_loss

        # Collect total loss stats
        stats["loss"] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
        with torch.no_grad():
            ref_encoder_out, ref_encoder_out_lens, _ = self.ref_model.encoder(speech, speech_lengths)

        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]


        return encoder_out, encoder_out_lens, ref_encoder_out, ref_encoder_out_lens
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
        decoder_out, _ = self.decoder(encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens)
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att, decoder_out
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

    def _calc_dpo_loss(
        self,
        encoder_out, 
        encoder_out_lens,
        decoder_out,
        ref_encoder_out, 
        ys_pad, 
        ys_pad_lens, 
        n_best_tokens, 
        n_best_probs, 
        n_best_length
    ):
        pass

    def _calc_mwer_loss(
        self,
        encoder_out, 
        encoder_out_lens,
        decoder_out,
        ys_pad, 
        ys_pad_lens, 

    ):
        n_best_tokens, n_best_ctc_probs, n_best_length = self._get_nbest(encoder_out, encoder_out_lens) 
        nbest_dist = self._get_nbest_dist(ys_pad, ys_pad_lens, n_best_tokens, n_best_length)
        n_best_att_prob = self._get_nbest_attprob(encoder_out, encoder_out_lens, n_best_tokens, n_best_length)
        mwer_loss = torch.sum(nbest_dist * n_best_att_prob.softmax(dim=-1), dim = -1)
        return torch.mean(mwer_loss)

    @torch.no_grad()
    def _get_nbest(self, encoder_out, encoder_out_lens):
        # ctc_log_prob (B, T, V)
        nbest = self.nbest
        ctc_log_prob = self.ctc.log_softmax(encoder_out)
        decode_results = self.ctc_decoder(ctc_log_prob, encoder_out_lens.int())
        # (B, N, L)
        n_best_tokens = self.ignore_id * torch.ones(len(ctc_log_prob), nbest, ctc_log_prob.size(2), dtype=torch.long, device=ctc_log_prob.device)
        # (B, N)
        n_best_ctc_probs = torch.zeros(len(ctc_log_prob), nbest, dtype=torch.float, device=ctc_log_prob.device)
        # (B, N)
        n_best_length = torch.zeros(len(ctc_log_prob), nbest, dtype=torch.long, device=ctc_log_prob.device)
        for i in range(len(decode_results)):
            for j in range(len(decode_results[i])):
                n_best_tokens[i, j, :len(decode_results[i][j].tokens)] = torch.tensor(decode_results[i][j].tokens, dtype=torch.long, device=ctc_log_prob.device)
                n_best_ctc_probs[i, j] = torch.tensor(decode_results[i][j].score, dtype=torch.float, device=ctc_log_prob.device)
                n_best_length[i,j] = len(decode_results[i][j].tokens)
        n_best_tokens = n_best_tokens[:,:,:n_best_length.max()]
        return n_best_tokens, n_best_ctc_probs, n_best_length
    
    def _get_nbest_attprob(self, encoder_out, encoder_out_lens, n_best_tokens, n_best_length):
        encoder_out = encoder_out.unsqueeze(1) # (B, 1, T, C)
        encoder_out = encoder_out.expand(-1, self.nbest,-1, -1).contiguous().view(-1, encoder_out.size(-2), encoder_out.size(-1))
        encoder_out_lens = encoder_out_lens.unsqueeze(1).expand(-1, self.nbest).contiguous().view(-1)
        
        if n_best_tokens.nelement() == 0:
            n_best_tokens = torch.empty(encoder_out.size(0) * self.nbest, 0).to(encoder_out.device)
        else:
            n_best_tokens = n_best_tokens.view(-1, n_best_length.max()).contiguous()
        n_best_length = n_best_length.view(-1).contiguous()

        ys_in_pad, ys_out_pad = add_sos_eos(n_best_tokens, self.sos, self.eos, self.ignore_id)
        ys_in_lens = n_best_length + 1

        #print(encoder_out.size())
        #print(encoder_out_lens.size())
        #print(ys_in_pad.size())
        #print(ys_in_lens.size())
        #print(max(ys_in_lens))
        #exit()
        # 1. Forward decoder
        decoder_out, _ = self.decoder(encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens)
        n_best_prob = torch.log_softmax(decoder_out, dim=-1)
        n_best_target_prob, _ = self.get_batch_logps(n_best_prob, ys_out_pad, self.ignore_id, True)

        return n_best_target_prob.view(-1, self.nbest)

    def _get_nbest_dist(self, ys_pad, ys_pad_lens, n_best_tokens, n_best_length):
        n_best_dist = torch.zeros(len(ys_pad), self.nbest)
        for i in range(len(n_best_dist)):
            for j in range(len(n_best_dist[i])):
                n_best_dist[i,j] = editdistance.eval(
                    ys_pad[i,:ys_pad_lens[i]], 
                    n_best_tokens[i, j, :n_best_length[i, j]]
                )
        return n_best_dist.to(ys_pad.device)
    
    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                f"Logits (batch and sequence length dim) {logits.shape[:-1]} and labels must have the same shape {labels.shape}."
            )

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)
