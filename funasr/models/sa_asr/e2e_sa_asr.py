# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as F

from funasr.layers.abs_normalize import AbsNormalize
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss, NllLoss  # noqa: H301
from funasr.models.ctc import CTC
from funasr.models.decoder.abs_decoder import AbsDecoder
from funasr.models.encoder.abs_encoder import AbsEncoder
from funasr.frontends.abs_frontend import AbsFrontend
from funasr.models.postencoder.abs_postencoder import AbsPostEncoder
from funasr.models.preencoder.abs_preencoder import AbsPreEncoder
from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.metrics import ErrorCalculator
from funasr.metrics.compute_acc import th_accuracy
from funasr.train_utils.device_funcs import force_gatherable
from funasr.models.base_model import FunASRModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class SAASRModel(FunASRModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        max_spk_num: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        asr_encoder: AbsEncoder,
        spk_encoder: torch.nn.Module,
        decoder: AbsDecoder,
        ctc: CTC,
        spk_weight: float = 0.5,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = 1
        self.eos = 2
        self.vocab_size = vocab_size
        self.max_spk_num = max_spk_num
        self.ignore_id = ignore_id
        self.spk_weight = spk_weight
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.asr_encoder = asr_encoder
        self.spk_encoder = spk_encoder

        if not hasattr(self.asr_encoder, "interctc_use_conditioning"):
            self.asr_encoder.interctc_use_conditioning = False
        if self.asr_encoder.interctc_use_conditioning:
            self.asr_encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.asr_encoder.output_size()
            )

        self.error_calculator = None

        # we set self.decoder = None in the CTC mode since
        # self.decoder parameters were never used and PyTorch complained
        # and threw an Exception in the multi-GPU experiment.
        # thanks Jeff Farris for pointing out the issue.
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

        self.criterion_spk = NllLoss(
            size=max_spk_num,
            padding_idx=ignore_id,
            normalize_length=length_normalized_loss,
        )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        profile: torch.Tensor,
        profile_lengths: torch.Tensor,
        text_id: torch.Tensor,
        text_id_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            profile: (Batch, Length, Dim)
            profile_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0] == speech_lengths.shape[0] == text.shape[0] == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        asr_encoder_out, encoder_out_lens, spk_encoder_out = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(asr_encoder_out, tuple):
            intermediate_outs = asr_encoder_out[1]
            asr_encoder_out = asr_encoder_out[0]

        loss_att, loss_spk, acc_att, acc_spk, cer_att, wer_att = None, None, None, None, None, None
        loss_ctc, cer_ctc = None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                asr_encoder_out, encoder_out_lens, text, text_lengths
            )

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text, text_lengths
                )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (1 - self.interctc_weight) * loss_ctc + self.interctc_weight * loss_interctc

        # 2b. Attention decoder branch
        if self.ctc_weight != 1.0:
            loss_att, loss_spk, acc_att, acc_spk, cer_att, wer_att = self._calc_att_loss(
                asr_encoder_out,
                spk_encoder_out,
                encoder_out_lens,
                text,
                text_lengths,
                profile,
                profile_lengths,
                text_id,
                text_id_lengths,
            )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss_asr = loss_att
        elif self.ctc_weight == 1.0:
            loss_asr = loss_ctc
        else:
            loss_asr = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        if self.spk_weight == 0.0:
            loss = loss_asr
        else:
            loss = self.spk_weight * loss_spk + (1 - self.spk_weight) * loss_asr

        stats = dict(
            loss=loss.detach(),
            loss_asr=loss_asr.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            loss_spk=loss_spk.detach() if loss_spk is not None else None,
            acc=acc_att,
            acc_spk=acc_spk,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            feats_raw = feats.clone()
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.asr_encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.asr_encoder(feats, feats_lengths, ctc=self.ctc)
        else:
            encoder_out, encoder_out_lens, _ = self.asr_encoder(feats, feats_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        encoder_out_spk_ori = self.spk_encoder(feats_raw, feats_lengths)[0]
        # import ipdb;ipdb.set_trace()
        if encoder_out_spk_ori.size(1) != encoder_out.size(1):
            encoder_out_spk = F.interpolate(
                encoder_out_spk_ori.transpose(-2, -1), size=(encoder_out.size(1)), mode="nearest"
            ).transpose(-2, -1)
        else:
            encoder_out_spk = encoder_out_spk_ori

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )
        assert encoder_out_spk.size(0) == speech.size(0), (
            encoder_out_spk.size(),
            speech.size(0),
        )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens, encoder_out_spk

        return encoder_out, encoder_out_lens, encoder_out_spk

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
        self,
        asr_encoder_out: torch.Tensor,
        spk_encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        profile: torch.Tensor,
        profile_lens: torch.Tensor,
        text_id: torch.Tensor,
        text_id_lengths: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, weights_no_pad, _ = self.decoder(
            asr_encoder_out,
            spk_encoder_out,
            encoder_out_lens,
            ys_in_pad,
            ys_in_lens,
            profile,
            profile_lens,
        )

        spk_num_no_pad = weights_no_pad.size(-1)
        pad = (0, self.max_spk_num - spk_num_no_pad)
        weights = F.pad(weights_no_pad, pad, mode="constant", value=0)

        # pre_id=weights.argmax(-1)
        # pre_text=decoder_out.argmax(-1)
        # id_mask=(pre_id==text_id).to(dtype=text_id.dtype)
        # pre_text_mask=pre_text*id_mask+1-id_mask #相同的地方不变，不同的地方设为1(<unk>)
        # padding_mask= ys_out_pad != self.ignore_id
        # numerator = torch.sum(pre_text_mask.masked_select(padding_mask) == ys_out_pad.masked_select(padding_mask))
        # denominator = torch.sum(padding_mask)
        # sd_acc = float(numerator) / float(denominator)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        loss_spk = self.criterion_spk(torch.log(weights), text_id)

        acc_spk = th_accuracy(
            weights.view(-1, self.max_spk_num),
            text_id,
            ignore_label=self.ignore_id,
        )
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

        return loss_att, loss_spk, acc_att, acc_spk, cer_att, wer_att

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
