#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Common functions for ASR."""

from typing import List, Optional, Tuple

import json
import logging
import sys

from itertools import groupby
import numpy as np
import six
import torch

from funasr.modules.beam_search.beam_search_transducer import BeamSearchTransducer
from funasr.models.rnnt_decoder.abs_decoder import AbsDecoder
from funasr.models.joint_network import JointNetwork

def end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
    """End detection.

    described in Eq. (50) of S. Watanabe et al
    "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"

    :param ended_hyps:
    :param i:
    :param M:
    :param D_end:
    :return:
    """
    if len(ended_hyps) == 0:
        return False
    count = 0
    best_hyp = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[0]
    for m in six.moves.range(M):
        # get ended_hyps with their length is i - m
        hyp_length = i - m
        hyps_same_length = [x for x in ended_hyps if len(x["yseq"]) == hyp_length]
        if len(hyps_same_length) > 0:
            best_hyp_same_length = sorted(
                hyps_same_length, key=lambda x: x["score"], reverse=True
            )[0]
            if best_hyp_same_length["score"] - best_hyp["score"] < D_end:
                count += 1

    if count == M:
        return True
    else:
        return False


# TODO(takaaki-hori): add different smoothing methods
def label_smoothing_dist(odim, lsm_type, transcript=None, blank=0):
    """Obtain label distribution for loss smoothing.

    :param odim:
    :param lsm_type:
    :param blank:
    :param transcript:
    :return:
    """
    if transcript is not None:
        with open(transcript, "rb") as f:
            trans_json = json.load(f)["utts"]

    if lsm_type == "unigram":
        assert transcript is not None, (
            "transcript is required for %s label smoothing" % lsm_type
        )
        labelcount = np.zeros(odim)
        for k, v in trans_json.items():
            ids = np.array([int(n) for n in v["output"][0]["tokenid"].split()])
            # to avoid an error when there is no text in an uttrance
            if len(ids) > 0:
                labelcount[ids] += 1
        labelcount[odim - 1] = len(transcript)  # count <eos>
        labelcount[labelcount == 0] = 1  # flooring
        labelcount[blank] = 0  # remove counts for blank
        labeldist = labelcount.astype(np.float32) / np.sum(labelcount)
    else:
        logging.error("Error: unexpected label smoothing type: %s" % lsm_type)
        sys.exit()

    return labeldist


def get_vgg2l_odim(idim, in_channel=3, out_channel=128):
    """Return the output size of the VGG frontend.

    :param in_channel: input channel size
    :param out_channel: output channel size
    :return: output size
    :rtype int
    """
    idim = idim / in_channel
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 1st max pooling
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 2nd max pooling
    return int(idim) * out_channel  # numer of channels


class ErrorCalculator(object):
    """Calculate CER and WER for E2E_ASR and CTC models during training.

    :param y_hats: numpy array with predicted text
    :param y_pads: numpy array with true (target) text
    :param char_list:
    :param sym_space:
    :param sym_blank:
    :return:
    """

    def __init__(
        self, char_list, sym_space, sym_blank, report_cer=False, report_wer=False
    ):
        """Construct an ErrorCalculator object."""
        super(ErrorCalculator, self).__init__()

        self.report_cer = report_cer
        self.report_wer = report_wer

        self.char_list = char_list
        self.space = sym_space
        self.blank = sym_blank
        self.idx_blank = self.char_list.index(self.blank)
        if self.space in self.char_list:
            self.idx_space = self.char_list.index(self.space)
        else:
            self.idx_space = None

    def __call__(self, ys_hat, ys_pad, is_ctc=False):
        """Calculate sentence-level WER/CER score.

        :param torch.Tensor ys_hat: prediction (batch, seqlen)
        :param torch.Tensor ys_pad: reference (batch, seqlen)
        :param bool is_ctc: calculate CER score for CTC
        :return: sentence-level WER score
        :rtype float
        :return: sentence-level CER score
        :rtype float
        """
        cer, wer = None, None
        if is_ctc:
            return self.calculate_cer_ctc(ys_hat, ys_pad)
        elif not self.report_cer and not self.report_wer:
            return cer, wer

        seqs_hat, seqs_true = self.convert_to_char(ys_hat, ys_pad)
        if self.report_cer:
            cer = self.calculate_cer(seqs_hat, seqs_true)

        if self.report_wer:
            wer = self.calculate_wer(seqs_hat, seqs_true)
        return cer, wer

    def calculate_cer_ctc(self, ys_hat, ys_pad):
        """Calculate sentence-level CER score for CTC.

        :param torch.Tensor ys_hat: prediction (batch, seqlen)
        :param torch.Tensor ys_pad: reference (batch, seqlen)
        :return: average sentence-level CER score
        :rtype float
        """
        import editdistance

        cers, char_ref_lens = [], []
        for i, y in enumerate(ys_hat):
            y_hat = [x[0] for x in groupby(y)]
            y_true = ys_pad[i]
            seq_hat, seq_true = [], []
            for idx in y_hat:
                idx = int(idx)
                if idx != -1 and idx != self.idx_blank and idx != self.idx_space:
                    seq_hat.append(self.char_list[int(idx)])

            for idx in y_true:
                idx = int(idx)
                if idx != -1 and idx != self.idx_blank and idx != self.idx_space:
                    seq_true.append(self.char_list[int(idx)])

            hyp_chars = "".join(seq_hat)
            ref_chars = "".join(seq_true)
            if len(ref_chars) > 0:
                cers.append(editdistance.eval(hyp_chars, ref_chars))
                char_ref_lens.append(len(ref_chars))

        cer_ctc = float(sum(cers)) / sum(char_ref_lens) if cers else None
        return cer_ctc

    def convert_to_char(self, ys_hat, ys_pad):
        """Convert index to character.

        :param torch.Tensor seqs_hat: prediction (batch, seqlen)
        :param torch.Tensor seqs_true: reference (batch, seqlen)
        :return: token list of prediction
        :rtype list
        :return: token list of reference
        :rtype list
        """
        seqs_hat, seqs_true = [], []
        for i, y_hat in enumerate(ys_hat):
            y_true = ys_pad[i]
            eos_true = np.where(y_true == -1)[0]
            ymax = eos_true[0] if len(eos_true) > 0 else len(y_true)
            # NOTE: padding index (-1) in y_true is used to pad y_hat
            seq_hat = [self.char_list[int(idx)] for idx in y_hat[:ymax]]
            seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
            seq_hat_text = "".join(seq_hat).replace(self.space, " ")
            seq_hat_text = seq_hat_text.replace(self.blank, "")
            seq_true_text = "".join(seq_true).replace(self.space, " ")
            seqs_hat.append(seq_hat_text)
            seqs_true.append(seq_true_text)
        return seqs_hat, seqs_true

    def calculate_cer(self, seqs_hat, seqs_true):
        """Calculate sentence-level CER score.

        :param list seqs_hat: prediction
        :param list seqs_true: reference
        :return: average sentence-level CER score
        :rtype float
        """
        import editdistance

        char_eds, char_ref_lens = [], []
        for i, seq_hat_text in enumerate(seqs_hat):
            seq_true_text = seqs_true[i]
            hyp_chars = seq_hat_text.replace(" ", "")
            ref_chars = seq_true_text.replace(" ", "")
            char_eds.append(editdistance.eval(hyp_chars, ref_chars))
            char_ref_lens.append(len(ref_chars))
        return float(sum(char_eds)) / sum(char_ref_lens)

    def calculate_wer(self, seqs_hat, seqs_true):
        """Calculate sentence-level WER score.

        :param list seqs_hat: prediction
        :param list seqs_true: reference
        :return: average sentence-level WER score
        :rtype float
        """
        import editdistance

        word_eds, word_ref_lens = [], []
        for i, seq_hat_text in enumerate(seqs_hat):
            seq_true_text = seqs_true[i]
            hyp_words = seq_hat_text.split()
            ref_words = seq_true_text.split()
            word_eds.append(editdistance.eval(hyp_words, ref_words))
            word_ref_lens.append(len(ref_words))
        return float(sum(word_eds)) / sum(word_ref_lens)

class ErrorCalculatorTransducer:
    """Calculate CER and WER for transducer models.
    Args:
        decoder: Decoder module.
        joint_network: Joint Network module.
        token_list: List of token units.
        sym_space: Space symbol.
        sym_blank: Blank symbol.
        report_cer: Whether to compute CER.
        report_wer: Whether to compute WER.
    """

    def __init__(
        self,
        decoder: AbsDecoder,
        joint_network: JointNetwork,
        token_list: List[int],
        sym_space: str,
        sym_blank: str,
        report_cer: bool = False,
        report_wer: bool = False,
    ) -> None:
        """Construct an ErrorCalculatorTransducer object."""
        super().__init__()

        self.beam_search = BeamSearchTransducer(
            decoder=decoder,
            joint_network=joint_network,
            beam_size=1,
            search_type="default",
            score_norm=False,
        )

        self.decoder = decoder

        self.token_list = token_list
        self.space = sym_space
        self.blank = sym_blank

        self.report_cer = report_cer
        self.report_wer = report_wer

    def __call__(
        self, encoder_out: torch.Tensor, target: torch.Tensor
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate sentence-level WER or/and CER score for Transducer model.
        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            target: Target label ID sequences. (B, L)
        Returns:
            : Sentence-level CER score.
            : Sentence-level WER score.
        """
        cer, wer = None, None

        batchsize = int(encoder_out.size(0))

        encoder_out = encoder_out.to(next(self.decoder.parameters()).device)

        batch_nbest = [self.beam_search(encoder_out[b]) for b in range(batchsize)]
        pred = [nbest_hyp[0].yseq[1:] for nbest_hyp in batch_nbest]

        char_pred, char_target = self.convert_to_char(pred, target)

        if self.report_cer:
            cer = self.calculate_cer(char_pred, char_target)

        if self.report_wer:
            wer = self.calculate_wer(char_pred, char_target)

        return cer, wer

    def convert_to_char(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[List, List]:
        """Convert label ID sequences to character sequences.
        Args:
            pred: Prediction label ID sequences. (B, U)
            target: Target label ID sequences. (B, L)
        Returns:
            char_pred: Prediction character sequences. (B, ?)
            char_target: Target character sequences. (B, ?)
        """
        char_pred, char_target = [], []

        for i, pred_i in enumerate(pred):
            char_pred_i = [self.token_list[int(h)] for h in pred_i]
            char_target_i = [self.token_list[int(r)] for r in target[i]]

            char_pred_i = "".join(char_pred_i).replace(self.space, " ")
            char_pred_i = char_pred_i.replace(self.blank, "")

            char_target_i = "".join(char_target_i).replace(self.space, " ")
            char_target_i = char_target_i.replace(self.blank, "")

            char_pred.append(char_pred_i)
            char_target.append(char_target_i)

        return char_pred, char_target

    def calculate_cer(
        self, char_pred: torch.Tensor, char_target: torch.Tensor
    ) -> float:
        """Calculate sentence-level CER score.
        Args:
            char_pred: Prediction character sequences. (B, ?)
            char_target: Target character sequences. (B, ?)
        Returns:
            : Average sentence-level CER score.
        """
        import editdistance

        distances, lens = [], []

        for i, char_pred_i in enumerate(char_pred):
            pred = char_pred_i.replace(" ", "")
            target = char_target[i].replace(" ", "")
            distances.append(editdistance.eval(pred, target))
            lens.append(len(target))

        return float(sum(distances)) / sum(lens)

    def calculate_wer(
        self, char_pred: torch.Tensor, char_target: torch.Tensor
    ) -> float:
        """Calculate sentence-level WER score.
        Args:
            char_pred: Prediction character sequences. (B, ?)
            char_target: Target character sequences. (B, ?)
        Returns:
            : Average sentence-level WER score
        """
        import editdistance

        distances, lens = [], []

        for i, char_pred_i in enumerate(char_pred):
            pred = char_pred_i.replace("▁", " ").split()
            target = char_target[i].replace("▁", " ").split()

            distances.append(editdistance.eval(pred, target))
            lens.append(len(target))

        return float(sum(distances)) / sum(lens)
