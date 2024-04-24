"""RNN decoder module."""

import logging
import math
import random
from argparse import Namespace

import numpy as np
import six
import torch
import torch.nn.functional as F

from funasr.models.transformer.utils.scorers.ctc_prefix_score import CTCPrefixScore
from funasr.models.transformer.utils.scorers.ctc_prefix_score import CTCPrefixScoreTH
from funasr.models.transformer.utils.scorers.scorer_interface import ScorerInterface
from funasr.metrics import end_detect
from funasr.models.transformer.utils.nets_utils import mask_by_length
from funasr.models.transformer.utils.nets_utils import pad_list
from funasr.metrics.compute_acc import th_accuracy
from funasr.models.transformer.utils.nets_utils import to_device
from funasr.models.language_model.rnn.attentions import att_to_numpy

MAX_DECODER_OUTPUT = 5
CTC_SCORING_RATIO = 1.5


class Decoder(torch.nn.Module, ScorerInterface):
    """Decoder module

    :param int eprojs: encoder projection units
    :param int odim: dimension of outputs
    :param str dtype: gru or lstm
    :param int dlayers: decoder layers
    :param int dunits: decoder units
    :param int sos: start of sequence symbol id
    :param int eos: end of sequence symbol id
    :param torch.nn.Module att: attention module
    :param int verbose: verbose level
    :param list char_list: list of character strings
    :param ndarray labeldist: distribution of label smoothing
    :param float lsm_weight: label smoothing weight
    :param float sampling_probability: scheduled sampling probability
    :param float dropout: dropout rate
    :param float context_residual: if True, use context vector for token generation
    :param float replace_sos: use for multilingual (speech/text) translation
    """

    def __init__(
        self,
        eprojs,
        odim,
        dtype,
        dlayers,
        dunits,
        sos,
        eos,
        att,
        verbose=0,
        char_list=None,
        labeldist=None,
        lsm_weight=0.0,
        sampling_probability=0.0,
        dropout=0.0,
        context_residual=False,
        replace_sos=False,
        num_encs=1,
    ):

        torch.nn.Module.__init__(self)
        self.dtype = dtype
        self.dunits = dunits
        self.dlayers = dlayers
        self.context_residual = context_residual
        self.embed = torch.nn.Embedding(odim, dunits)
        self.dropout_emb = torch.nn.Dropout(p=dropout)

        self.decoder = torch.nn.ModuleList()
        self.dropout_dec = torch.nn.ModuleList()
        self.decoder += [
            (
                torch.nn.LSTMCell(dunits + eprojs, dunits)
                if self.dtype == "lstm"
                else torch.nn.GRUCell(dunits + eprojs, dunits)
            )
        ]
        self.dropout_dec += [torch.nn.Dropout(p=dropout)]
        for _ in six.moves.range(1, self.dlayers):
            self.decoder += [
                (
                    torch.nn.LSTMCell(dunits, dunits)
                    if self.dtype == "lstm"
                    else torch.nn.GRUCell(dunits, dunits)
                )
            ]
            self.dropout_dec += [torch.nn.Dropout(p=dropout)]
            # NOTE: dropout is applied only for the vertical connections
            # see https://arxiv.org/pdf/1409.2329.pdf
        self.ignore_id = -1

        if context_residual:
            self.output = torch.nn.Linear(dunits + eprojs, odim)
        else:
            self.output = torch.nn.Linear(dunits, odim)

        self.loss = None
        self.att = att
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.odim = odim
        self.verbose = verbose
        self.char_list = char_list
        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight
        self.sampling_probability = sampling_probability
        self.dropout = dropout
        self.num_encs = num_encs

        # for multilingual E2E-ST
        self.replace_sos = replace_sos

        self.logzero = -10000000000.0

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def rnn_forward(self, ey, z_list, c_list, z_prev, c_prev):
        if self.dtype == "lstm":
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))
            for i in six.moves.range(1, self.dlayers):
                z_list[i], c_list[i] = self.decoder[i](
                    self.dropout_dec[i - 1](z_list[i - 1]), (z_prev[i], c_prev[i])
                )
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])
            for i in six.moves.range(1, self.dlayers):
                z_list[i] = self.decoder[i](self.dropout_dec[i - 1](z_list[i - 1]), z_prev[i])
        return z_list, c_list

    def forward(self, hs_pad, hlens, ys_pad, strm_idx=0, lang_ids=None):
        """Decoder forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
                                    [in multi-encoder case,
                                    list of torch.Tensor,
                                    [(B, Tmax_1, D), (B, Tmax_2, D), ..., ] ]
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
                                   [in multi-encoder case, list of torch.Tensor,
                                   [(B), (B), ..., ]
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor
                                    (B, Lmax)
        :param int strm_idx: stream index indicates the index of decoding stream.
        :param torch.Tensor lang_ids: batch of target language id tensor (B, 1)
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy
        :rtype: float
        """
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            hs_pad = [hs_pad]
            hlens = [hlens]

        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        # attention index for the attention module
        # in SPA (speaker parallel attention),
        # att_idx is used to select attention module. In other cases, it is 0.
        att_idx = min(strm_idx, len(self.att) - 1)

        # hlens should be list of list of integer
        hlens = [list(map(int, hlens[idx])) for idx in range(self.num_encs)]

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        if self.replace_sos:
            ys_in = [torch.cat([idx, y], dim=0) for idx, y in zip(lang_ids, ys)]
        else:
            ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get dim, length info
        batch = ys_out_pad.size(0)
        olength = ys_out_pad.size(1)
        for idx in range(self.num_encs):
            logging.info(
                self.__class__.__name__
                + "Number of Encoder:{}; enc{}: input lengths: {}.".format(
                    self.num_encs, idx + 1, hlens[idx]
                )
            )
        logging.info(
            self.__class__.__name__ + " output lengths: " + str([y.size(0) for y in ys_out])
        )

        # initialization
        c_list = [self.zero_state(hs_pad[0])]
        z_list = [self.zero_state(hs_pad[0])]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad[0]))
            z_list.append(self.zero_state(hs_pad[0]))
        z_all = []
        if self.num_encs == 1:
            att_w = None
            self.att[att_idx].reset()  # reset pre-computation of h
        else:
            att_w_list = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * (self.num_encs)  # atts
            for idx in range(self.num_encs + 1):
                self.att[idx].reset()  # reset pre-computation of h in atts and han

        # pre-computation of embedding
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            if self.num_encs == 1:
                att_c, att_w = self.att[att_idx](
                    hs_pad[0], hlens[0], self.dropout_dec[0](z_list[0]), att_w
                )
            else:
                for idx in range(self.num_encs):
                    att_c_list[idx], att_w_list[idx] = self.att[idx](
                        hs_pad[idx],
                        hlens[idx],
                        self.dropout_dec[0](z_list[0]),
                        att_w_list[idx],
                    )
                hs_pad_han = torch.stack(att_c_list, dim=1)
                hlens_han = [self.num_encs] * len(ys_in)
                att_c, att_w_list[self.num_encs] = self.att[self.num_encs](
                    hs_pad_han,
                    hlens_han,
                    self.dropout_dec[0](z_list[0]),
                    att_w_list[self.num_encs],
                )
            if i > 0 and random.random() < self.sampling_probability:
                logging.info(" scheduled sampling ")
                z_out = self.output(z_all[-1])
                z_out = np.argmax(z_out.detach().cpu(), axis=1)
                z_out = self.dropout_emb(self.embed(to_device(hs_pad[0], z_out)))
                ey = torch.cat((z_out, att_c), dim=1)  # utt x (zdim + hdim)
            else:
                ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)
            if self.context_residual:
                z_all.append(
                    torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1)
                )  # utt x (zdim + hdim)
            else:
                z_all.append(self.dropout_dec[-1](z_list[-1]))  # utt x (zdim)

        z_all = torch.stack(z_all, dim=1).view(batch * olength, -1)
        # compute loss
        y_all = self.output(z_all)
        self.loss = F.cross_entropy(
            y_all,
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="mean",
        )
        # compute perplexity
        ppl = math.exp(self.loss.item())
        # -1: eos, which is removed in the loss computation
        self.loss *= np.mean([len(x) for x in ys_in]) - 1
        acc = th_accuracy(y_all, ys_out_pad, ignore_label=self.ignore_id)
        logging.info("att loss:" + "".join(str(self.loss.item()).split("\n")))

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            ys_hat = y_all.view(batch, olength, -1)
            ys_true = ys_out_pad
            for (i, y_hat), y_true in zip(
                enumerate(ys_hat.detach().cpu().numpy()), ys_true.detach().cpu().numpy()
            ):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = np.argmax(y_hat[y_true != self.ignore_id], axis=1)
                idx_true = y_true[y_true != self.ignore_id]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat)
                seq_true = "".join(seq_true)
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = to_device(hs_pad[0], torch.from_numpy(self.labeldist))
            loss_reg = -torch.sum(
                (F.log_softmax(y_all, dim=1) * self.vlabeldist).view(-1), dim=0
            ) / len(ys_in)
            self.loss = (1.0 - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc, ppl

    def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None, strm_idx=0):
        """beam search implementation

        :param torch.Tensor h: encoder hidden state (T, eprojs)
                                [in multi-encoder case, list of torch.Tensor,
                                [(T1, eprojs), (T2, eprojs), ...] ]
        :param torch.Tensor lpz: ctc log softmax output (T, odim)
                                [in multi-encoder case, list of torch.Tensor,
                                [(T1, odim), (T2, odim), ...] ]
        :param Namespace recog_args: argument Namespace containing options
        :param char_list: list of character strings
        :param torch.nn.Module rnnlm: language module
        :param int strm_idx:
            stream index for speaker parallel attention in multi-speaker case
        :return: N-best decoding results
        :rtype: list of dicts
        """
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            h = [h]
            lpz = [lpz]
        if self.num_encs > 1 and lpz is None:
            lpz = [lpz] * self.num_encs

        for idx in range(self.num_encs):
            logging.info(
                "Number of Encoder:{}; enc{}: input lengths: {}.".format(
                    self.num_encs, idx + 1, h[0].size(0)
                )
            )
        att_idx = min(strm_idx, len(self.att) - 1)
        # initialization
        c_list = [self.zero_state(h[0].unsqueeze(0))]
        z_list = [self.zero_state(h[0].unsqueeze(0))]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(h[0].unsqueeze(0)))
            z_list.append(self.zero_state(h[0].unsqueeze(0)))
        if self.num_encs == 1:
            a = None
            self.att[att_idx].reset()  # reset pre-computation of h
        else:
            a = [None] * (self.num_encs + 1)  # atts + han
            att_w_list = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * (self.num_encs)  # atts
            for idx in range(self.num_encs + 1):
                self.att[idx].reset()  # reset pre-computation of h in atts and han

        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = getattr(recog_args, "ctc_weight", False)  # for NMT

        if lpz[0] is not None and self.num_encs > 1:
            # weights-ctc,
            # e.g. ctc_loss = w_1*ctc_1_loss + w_2 * ctc_2_loss + w_N * ctc_N_loss
            weights_ctc_dec = recog_args.weights_ctc_dec / np.sum(
                recog_args.weights_ctc_dec
            )  # normalize
            logging.info("ctc weights (decoding): " + " ".join([str(x) for x in weights_ctc_dec]))
        else:
            weights_ctc_dec = [1.0]

        # preprate sos
        if self.replace_sos and recog_args.tgt_lang:
            y = char_list.index(recog_args.tgt_lang)
        else:
            y = self.sos
        logging.info("<sos> index: " + str(y))
        logging.info("<sos> mark: " + char_list[y])
        vy = h[0].new_zeros(1).long()

        maxlen = np.amin([h[idx].size(0) for idx in range(self.num_encs)])
        if recog_args.maxlenratio != 0:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * maxlen))
        minlen = int(recog_args.minlenratio * maxlen)
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {
                "score": 0.0,
                "yseq": [y],
                "c_prev": c_list,
                "z_prev": z_list,
                "a_prev": a,
                "rnnlm_prev": None,
            }
        else:
            hyp = {
                "score": 0.0,
                "yseq": [y],
                "c_prev": c_list,
                "z_prev": z_list,
                "a_prev": a,
            }
        if lpz[0] is not None:
            ctc_prefix_score = [
                CTCPrefixScore(lpz[idx].detach().numpy(), 0, self.eos, np)
                for idx in range(self.num_encs)
            ]
            hyp["ctc_state_prev"] = [
                ctc_prefix_score[idx].initial_state() for idx in range(self.num_encs)
            ]
            hyp["ctc_score_prev"] = [0.0] * self.num_encs
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz[0].shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz[0].shape[-1]
        hyps = [hyp]
        ended_hyps = []

        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]
                ey = self.dropout_emb(self.embed(vy))  # utt list (1) x zdim
                if self.num_encs == 1:
                    att_c, att_w = self.att[att_idx](
                        h[0].unsqueeze(0),
                        [h[0].size(0)],
                        self.dropout_dec[0](hyp["z_prev"][0]),
                        hyp["a_prev"],
                    )
                else:
                    for idx in range(self.num_encs):
                        att_c_list[idx], att_w_list[idx] = self.att[idx](
                            h[idx].unsqueeze(0),
                            [h[idx].size(0)],
                            self.dropout_dec[0](hyp["z_prev"][0]),
                            hyp["a_prev"][idx],
                        )
                    h_han = torch.stack(att_c_list, dim=1)
                    att_c, att_w_list[self.num_encs] = self.att[self.num_encs](
                        h_han,
                        [self.num_encs],
                        self.dropout_dec[0](hyp["z_prev"][0]),
                        hyp["a_prev"][self.num_encs],
                    )
                ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
                z_list, c_list = self.rnn_forward(ey, z_list, c_list, hyp["z_prev"], hyp["c_prev"])

                # get nbest local scores and their ids
                if self.context_residual:
                    logits = self.output(
                        torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1)
                    )
                else:
                    logits = self.output(self.dropout_dec[-1](z_list[-1]))
                local_att_scores = F.log_softmax(logits, dim=1)
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz[0] is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1
                    )
                    ctc_scores, ctc_states = (
                        [None] * self.num_encs,
                        [None] * self.num_encs,
                    )
                    for idx in range(self.num_encs):
                        ctc_scores[idx], ctc_states[idx] = ctc_prefix_score[idx](
                            hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"][idx]
                        )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]]
                    if self.num_encs == 1:
                        local_scores += ctc_weight * torch.from_numpy(
                            ctc_scores[0] - hyp["ctc_score_prev"][0]
                        )
                    else:
                        for idx in range(self.num_encs):
                            local_scores += (
                                ctc_weight
                                * weights_ctc_dec[idx]
                                * torch.from_numpy(ctc_scores[idx] - hyp["ctc_score_prev"][idx])
                            )
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    # [:] is needed!
                    new_hyp["z_prev"] = z_list[:]
                    new_hyp["c_prev"] = c_list[:]
                    if self.num_encs == 1:
                        new_hyp["a_prev"] = att_w[:]
                    else:
                        new_hyp["a_prev"] = [att_w_list[idx][:] for idx in range(self.num_encs + 1)]
                    new_hyp["score"] = hyp["score"] + local_best_scores[0, j]
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    if lpz[0] is not None:
                        new_hyp["ctc_state_prev"] = [
                            ctc_states[idx][joint_best_ids[0, j]] for idx in range(self.num_encs)
                        ]
                        new_hyp["ctc_score_prev"] = [
                            ctc_scores[idx][joint_best_ids[0, j]] for idx in range(self.num_encs)
                        ]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(hyps_best_kept, key=lambda x: x["score"], reverse=True)[
                    :beam
                ]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypotheses: " + str(len(hyps)))
            logging.debug("best hypo: " + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last position in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # add ended hypotheses to a final list,
            # and removed them from current hypotheses
            # (this will be a problem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += recog_args.lm_weight * rnnlm.final(hyp["rnnlm_prev"])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remaining hypotheses: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            for hyp in hyps:
                logging.debug("hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]]))

            logging.debug("number of ended hypotheses: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), recog_args.nbest)
        ]

        # check number of hypotheses
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, " "perform recognition again with smaller minlenratio."
            )
            # should copy because Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            if self.num_encs == 1:
                return self.recognize_beam(h[0], lpz[0], recog_args, char_list, rnnlm)
            else:
                return self.recognize_beam(h, lpz, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )

        # remove sos
        return nbest_hyps

    def recognize_beam_batch(
        self,
        h,
        hlens,
        lpz,
        recog_args,
        char_list,
        rnnlm=None,
        normalize_score=True,
        strm_idx=0,
        lang_ids=None,
    ):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            h = [h]
            hlens = [hlens]
            lpz = [lpz]
        if self.num_encs > 1 and lpz is None:
            lpz = [lpz] * self.num_encs

        att_idx = min(strm_idx, len(self.att) - 1)
        for idx in range(self.num_encs):
            logging.info(
                "Number of Encoder:{}; enc{}: input lengths: {}.".format(
                    self.num_encs, idx + 1, h[idx].size(1)
                )
            )
            h[idx] = mask_by_length(h[idx], hlens[idx], 0.0)

        # search params
        batch = len(hlens[0])
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = getattr(recog_args, "ctc_weight", 0)  # for NMT
        att_weight = 1.0 - ctc_weight
        ctc_margin = getattr(
            recog_args, "ctc_window_margin", 0
        )  # use getattr to keep compatibility
        # weights-ctc,
        # e.g. ctc_loss = w_1*ctc_1_loss + w_2 * ctc_2_loss + w_N * ctc_N_loss
        if lpz[0] is not None and self.num_encs > 1:
            weights_ctc_dec = recog_args.weights_ctc_dec / np.sum(
                recog_args.weights_ctc_dec
            )  # normalize
            logging.info("ctc weights (decoding): " + " ".join([str(x) for x in weights_ctc_dec]))
        else:
            weights_ctc_dec = [1.0]

        n_bb = batch * beam
        pad_b = to_device(h[0], torch.arange(batch) * beam).view(-1, 1)

        max_hlen = np.amin([max(hlens[idx]) for idx in range(self.num_encs)])
        if recog_args.maxlenratio == 0:
            maxlen = max_hlen
        else:
            maxlen = max(1, int(recog_args.maxlenratio * max_hlen))
        minlen = int(recog_args.minlenratio * max_hlen)
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialization
        c_prev = [to_device(h[0], torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_prev = [to_device(h[0], torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        c_list = [to_device(h[0], torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_list = [to_device(h[0], torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        vscores = to_device(h[0], torch.zeros(batch, beam))

        rnnlm_state = None
        if self.num_encs == 1:
            a_prev = [None]
            att_w_list, ctc_scorer, ctc_state = [None], [None], [None]
            self.att[att_idx].reset()  # reset pre-computation of h
        else:
            a_prev = [None] * (self.num_encs + 1)  # atts + han
            att_w_list = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * (self.num_encs)  # atts
            ctc_scorer, ctc_state = [None] * (self.num_encs), [None] * (self.num_encs)
            for idx in range(self.num_encs + 1):
                self.att[idx].reset()  # reset pre-computation of h in atts and han

        if self.replace_sos and recog_args.tgt_lang:
            logging.info("<sos> index: " + str(char_list.index(recog_args.tgt_lang)))
            logging.info("<sos> mark: " + recog_args.tgt_lang)
            yseq = [[char_list.index(recog_args.tgt_lang)] for _ in six.moves.range(n_bb)]
        elif lang_ids is not None:
            # NOTE: used for evaluation during training
            yseq = [[lang_ids[b // recog_args.beam_size]] for b in six.moves.range(n_bb)]
        else:
            logging.info("<sos> index: " + str(self.sos))
            logging.info("<sos> mark: " + char_list[self.sos])
            yseq = [[self.sos] for _ in six.moves.range(n_bb)]

        accum_odim_ids = [self.sos for _ in six.moves.range(n_bb)]
        stop_search = [False for _ in six.moves.range(batch)]
        nbest_hyps = [[] for _ in six.moves.range(batch)]
        ended_hyps = [[] for _ in range(batch)]

        exp_hlens = [
            hlens[idx].repeat(beam).view(beam, batch).transpose(0, 1).contiguous()
            for idx in range(self.num_encs)
        ]
        exp_hlens = [exp_hlens[idx].view(-1).tolist() for idx in range(self.num_encs)]
        exp_h = [
            h[idx].unsqueeze(1).repeat(1, beam, 1, 1).contiguous() for idx in range(self.num_encs)
        ]
        exp_h = [
            exp_h[idx].view(n_bb, h[idx].size()[1], h[idx].size()[2])
            for idx in range(self.num_encs)
        ]

        if lpz[0] is not None:
            scoring_num = min(
                int(beam * CTC_SCORING_RATIO) if att_weight > 0.0 and not lpz[0].is_cuda else 0,
                lpz[0].size(-1),
            )
            ctc_scorer = [
                CTCPrefixScoreTH(
                    lpz[idx],
                    hlens[idx],
                    0,
                    self.eos,
                    margin=ctc_margin,
                )
                for idx in range(self.num_encs)
            ]

        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            vy = to_device(h[0], torch.LongTensor(self._get_last_yseq(yseq)))
            ey = self.dropout_emb(self.embed(vy))
            if self.num_encs == 1:
                att_c, att_w = self.att[att_idx](
                    exp_h[0], exp_hlens[0], self.dropout_dec[0](z_prev[0]), a_prev[0]
                )
                att_w_list = [att_w]
            else:
                for idx in range(self.num_encs):
                    att_c_list[idx], att_w_list[idx] = self.att[idx](
                        exp_h[idx],
                        exp_hlens[idx],
                        self.dropout_dec[0](z_prev[0]),
                        a_prev[idx],
                    )
                exp_h_han = torch.stack(att_c_list, dim=1)
                att_c, att_w_list[self.num_encs] = self.att[self.num_encs](
                    exp_h_han,
                    [self.num_encs] * n_bb,
                    self.dropout_dec[0](z_prev[0]),
                    a_prev[self.num_encs],
                )
            ey = torch.cat((ey, att_c), dim=1)

            # attention decoder
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_prev, c_prev)
            if self.context_residual:
                logits = self.output(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))
            else:
                logits = self.output(self.dropout_dec[-1](z_list[-1]))
            local_scores = att_weight * F.log_softmax(logits, dim=1)

            # rnnlm
            if rnnlm:
                rnnlm_state, local_lm_scores = rnnlm.buff_predict(rnnlm_state, vy, n_bb)
                local_scores = local_scores + recog_args.lm_weight * local_lm_scores

            # ctc
            if ctc_scorer[0]:
                local_scores[:, 0] = self.logzero  # avoid choosing blank
                part_ids = (
                    torch.topk(local_scores, scoring_num, dim=-1)[1] if scoring_num > 0 else None
                )
                for idx in range(self.num_encs):
                    att_w = att_w_list[idx]
                    att_w_ = att_w if isinstance(att_w, torch.Tensor) else att_w[0]
                    local_ctc_scores, ctc_state[idx] = ctc_scorer[idx](
                        yseq, ctc_state[idx], part_ids, att_w_
                    )
                    local_scores = (
                        local_scores + ctc_weight * weights_ctc_dec[idx] * local_ctc_scores
                    )

            local_scores = local_scores.view(batch, beam, self.odim)
            if i == 0:
                local_scores[:, 1:, :] = self.logzero

            # accumulate scores
            eos_vscores = local_scores[:, :, self.eos] + vscores
            vscores = vscores.view(batch, beam, 1).repeat(1, 1, self.odim)
            vscores[:, :, self.eos] = self.logzero
            vscores = (vscores + local_scores).view(batch, -1)

            # global pruning
            accum_best_scores, accum_best_ids = torch.topk(vscores, beam, 1)
            accum_odim_ids = torch.fmod(accum_best_ids, self.odim).view(-1).data.cpu().tolist()
            accum_padded_beam_ids = (
                (accum_best_ids // self.odim + pad_b).view(-1).data.cpu().tolist()
            )

            y_prev = yseq[:][:]
            yseq = self._index_select_list(yseq, accum_padded_beam_ids)
            yseq = self._append_ids(yseq, accum_odim_ids)
            vscores = accum_best_scores
            vidx = to_device(h[0], torch.LongTensor(accum_padded_beam_ids))

            a_prev = []
            num_atts = self.num_encs if self.num_encs == 1 else self.num_encs + 1
            for idx in range(num_atts):
                if isinstance(att_w_list[idx], torch.Tensor):
                    _a_prev = torch.index_select(
                        att_w_list[idx].view(n_bb, *att_w_list[idx].shape[1:]), 0, vidx
                    )
                elif isinstance(att_w_list[idx], list):
                    # handle the case of multi-head attention
                    _a_prev = [
                        torch.index_select(att_w_one.view(n_bb, -1), 0, vidx)
                        for att_w_one in att_w_list[idx]
                    ]
                else:
                    # handle the case of location_recurrent when return is a tuple
                    _a_prev_ = torch.index_select(att_w_list[idx][0].view(n_bb, -1), 0, vidx)
                    _h_prev_ = torch.index_select(att_w_list[idx][1][0].view(n_bb, -1), 0, vidx)
                    _c_prev_ = torch.index_select(att_w_list[idx][1][1].view(n_bb, -1), 0, vidx)
                    _a_prev = (_a_prev_, (_h_prev_, _c_prev_))
                a_prev.append(_a_prev)
            z_prev = [
                torch.index_select(z_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)
            ]
            c_prev = [
                torch.index_select(c_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)
            ]

            # pick ended hyps
            if i >= minlen:
                k = 0
                penalty_i = (i + 1) * penalty
                thr = accum_best_scores[:, -1]
                for samp_i in six.moves.range(batch):
                    if stop_search[samp_i]:
                        k = k + beam
                        continue
                    for beam_j in six.moves.range(beam):
                        _vscore = None
                        if eos_vscores[samp_i, beam_j] > thr[samp_i]:
                            yk = y_prev[k][:]
                            if len(yk) <= min(hlens[idx][samp_i] for idx in range(self.num_encs)):
                                _vscore = eos_vscores[samp_i][beam_j] + penalty_i
                        elif i == maxlen - 1:
                            yk = yseq[k][:]
                            _vscore = vscores[samp_i][beam_j] + penalty_i
                        if _vscore:
                            yk.append(self.eos)
                            if rnnlm:
                                _vscore += recog_args.lm_weight * rnnlm.final(rnnlm_state, index=k)
                            _score = _vscore.data.cpu().numpy()
                            ended_hyps[samp_i].append(
                                {"yseq": yk, "vscore": _vscore, "score": _score}
                            )
                        k = k + 1

            # end detection
            stop_search = [
                stop_search[samp_i] or end_detect(ended_hyps[samp_i], i)
                for samp_i in six.moves.range(batch)
            ]
            stop_search_summary = list(set(stop_search))
            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

            if rnnlm:
                rnnlm_state = self._index_select_lm_state(rnnlm_state, 0, vidx)
            if ctc_scorer[0]:
                for idx in range(self.num_encs):
                    ctc_state[idx] = ctc_scorer[idx].index_select_state(
                        ctc_state[idx], accum_best_ids
                    )

        torch.cuda.empty_cache()

        dummy_hyps = [{"yseq": [self.sos, self.eos], "score": np.array([-float("inf")])}]
        ended_hyps = [
            ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
            for samp_i in six.moves.range(batch)
        ]
        if normalize_score:
            for samp_i in six.moves.range(batch):
                for x in ended_hyps[samp_i]:
                    x["score"] /= len(x["yseq"])

        nbest_hyps = [
            sorted(ended_hyps[samp_i], key=lambda x: x["score"], reverse=True)[
                : min(len(ended_hyps[samp_i]), recog_args.nbest)
            ]
            for samp_i in six.moves.range(batch)
        ]

        return nbest_hyps

    def calculate_all_attentions(self, hs_pad, hlen, ys_pad, strm_idx=0, lang_ids=None):
        """Calculate all of attentions

        :param torch.Tensor hs_pad: batch of padded hidden state sequences
                                    (B, Tmax, D)
                                    in multi-encoder case, list of torch.Tensor,
                                    [(B, Tmax_1, D), (B, Tmax_2, D), ..., ] ]
        :param torch.Tensor hlen: batch of lengths of hidden state sequences (B)
                                    [in multi-encoder case, list of torch.Tensor,
                                    [(B), (B), ..., ]
        :param torch.Tensor ys_pad:
            batch of padded character id sequence tensor (B, Lmax)
        :param int strm_idx:
            stream index for parallel speaker attention in multi-speaker case
        :param torch.Tensor lang_ids: batch of target language id tensor (B, 1)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) multi-encoder case =>
                [(B, Lmax, Tmax1), (B, Lmax, Tmax2), ..., (B, Lmax, NumEncs)]
            3) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            hs_pad = [hs_pad]
            hlen = [hlen]

        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        att_idx = min(strm_idx, len(self.att) - 1)

        # hlen should be list of list of integer
        hlen = [list(map(int, hlen[idx])) for idx in range(self.num_encs)]

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        if self.replace_sos:
            ys_in = [torch.cat([idx, y], dim=0) for idx, y in zip(lang_ids, ys)]
        else:
            ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get length info
        olength = ys_out_pad.size(1)

        # initialization
        c_list = [self.zero_state(hs_pad[0])]
        z_list = [self.zero_state(hs_pad[0])]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad[0]))
            z_list.append(self.zero_state(hs_pad[0]))
        att_ws = []
        if self.num_encs == 1:
            att_w = None
            self.att[att_idx].reset()  # reset pre-computation of h
        else:
            att_w_list = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * (self.num_encs)  # atts
            for idx in range(self.num_encs + 1):
                self.att[idx].reset()  # reset pre-computation of h in atts and han

        # pre-computation of embedding
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            if self.num_encs == 1:
                att_c, att_w = self.att[att_idx](
                    hs_pad[0], hlen[0], self.dropout_dec[0](z_list[0]), att_w
                )
                att_ws.append(att_w)
            else:
                for idx in range(self.num_encs):
                    att_c_list[idx], att_w_list[idx] = self.att[idx](
                        hs_pad[idx],
                        hlen[idx],
                        self.dropout_dec[0](z_list[0]),
                        att_w_list[idx],
                    )
                hs_pad_han = torch.stack(att_c_list, dim=1)
                hlen_han = [self.num_encs] * len(ys_in)
                att_c, att_w_list[self.num_encs] = self.att[self.num_encs](
                    hs_pad_han,
                    hlen_han,
                    self.dropout_dec[0](z_list[0]),
                    att_w_list[self.num_encs],
                )
                att_ws.append(att_w_list.copy())
            ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)

        if self.num_encs == 1:
            # convert to numpy array with the shape (B, Lmax, Tmax)
            att_ws = att_to_numpy(att_ws, self.att[att_idx])
        else:
            _att_ws = []
            for idx, ws in enumerate(zip(*att_ws)):
                ws = att_to_numpy(ws, self.att[idx])
                _att_ws.append(ws)
            att_ws = _att_ws
        return att_ws

    @staticmethod
    def _get_last_yseq(exp_yseq):
        last = []
        for y_seq in exp_yseq:
            last.append(y_seq[-1])
        return last

    @staticmethod
    def _append_ids(yseq, ids):
        if isinstance(ids, list):
            for i, j in enumerate(ids):
                yseq[i].append(j)
        else:
            for i in range(len(yseq)):
                yseq[i].append(ids)
        return yseq

    @staticmethod
    def _index_select_list(yseq, lst):
        new_yseq = []
        for i in lst:
            new_yseq.append(yseq[i][:])
        return new_yseq

    @staticmethod
    def _index_select_lm_state(rnnlm_state, dim, vidx):
        if isinstance(rnnlm_state, dict):
            new_state = {}
            for k, v in rnnlm_state.items():
                new_state[k] = [torch.index_select(vi, dim, vidx) for vi in v]
        elif isinstance(rnnlm_state, list):
            new_state = []
            for i in vidx:
                new_state.append(rnnlm_state[int(i)][:])
        return new_state

    # scorer interface methods
    def init_state(self, x):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            x = [x]

        c_list = [self.zero_state(x[0].unsqueeze(0))]
        z_list = [self.zero_state(x[0].unsqueeze(0))]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(x[0].unsqueeze(0)))
            z_list.append(self.zero_state(x[0].unsqueeze(0)))
        # TODO(karita): support strm_index for `asr_mix`
        strm_index = 0
        att_idx = min(strm_index, len(self.att) - 1)
        if self.num_encs == 1:
            a = None
            self.att[att_idx].reset()  # reset pre-computation of h
        else:
            a = [None] * (self.num_encs + 1)  # atts + han
            for idx in range(self.num_encs + 1):
                self.att[idx].reset()  # reset pre-computation of h in atts and han
        return dict(
            c_prev=c_list[:],
            z_prev=z_list[:],
            a_prev=a,
            workspace=(att_idx, z_list, c_list),
        )

    def score(self, yseq, state, x):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            x = [x]

        att_idx, z_list, c_list = state["workspace"]
        vy = yseq[-1].unsqueeze(0)
        ey = self.dropout_emb(self.embed(vy))  # utt list (1) x zdim
        if self.num_encs == 1:
            att_c, att_w = self.att[att_idx](
                x[0].unsqueeze(0),
                [x[0].size(0)],
                self.dropout_dec[0](state["z_prev"][0]),
                state["a_prev"],
            )
        else:
            att_w = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * (self.num_encs)  # atts
            for idx in range(self.num_encs):
                att_c_list[idx], att_w[idx] = self.att[idx](
                    x[idx].unsqueeze(0),
                    [x[idx].size(0)],
                    self.dropout_dec[0](state["z_prev"][0]),
                    state["a_prev"][idx],
                )
            h_han = torch.stack(att_c_list, dim=1)
            att_c, att_w[self.num_encs] = self.att[self.num_encs](
                h_han,
                [self.num_encs],
                self.dropout_dec[0](state["z_prev"][0]),
                state["a_prev"][self.num_encs],
            )
        ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
        z_list, c_list = self.rnn_forward(ey, z_list, c_list, state["z_prev"], state["c_prev"])
        if self.context_residual:
            logits = self.output(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))
        else:
            logits = self.output(self.dropout_dec[-1](z_list[-1]))
        logp = F.log_softmax(logits, dim=1).squeeze(0)
        return (
            logp,
            dict(
                c_prev=c_list[:],
                z_prev=z_list[:],
                a_prev=att_w,
                workspace=(att_idx, z_list, c_list),
            ),
        )


def decoder_for(args, odim, sos, eos, att, labeldist):
    return Decoder(
        args.eprojs,
        odim,
        args.dtype,
        args.dlayers,
        args.dunits,
        sos,
        eos,
        att,
        args.verbose,
        args.char_list,
        labeldist,
        args.lsm_weight,
        args.sampling_probability,
        args.dropout_rate_decoder,
        getattr(args, "context_residual", False),  # use getattr to keep compatibility
        getattr(args, "replace_sos", False),  # use getattr to keep compatibility
        getattr(args, "num_encs", 1),
    )  # use getattr to keep compatibility
