#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from funasr.models.transducer.joint_network import JointNetwork


@dataclass
class Hypothesis:
    """Default hypothesis definition for Transducer search algorithms.

    Args:
        score: Total log-probability.
        yseq: Label sequence as integer ID sequence.
        dec_state: RNNDecoder or StatelessDecoder state.
                     ((N, 1, D_dec), (N, 1, D_dec) or None) or None
        lm_state: RNNLM state. ((N, D_lm), (N, D_lm)) or None

    """

    score: float
    yseq: List[int]
    dec_state: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None
    lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None


@dataclass
class ExtendedHypothesis(Hypothesis):
    """Extended hypothesis definition for NSC beam search and mAES.

    Args:
        : Hypothesis dataclass arguments.
        dec_out: Decoder output sequence. (B, D_dec)
        lm_score: Log-probabilities of the LM for given label. (vocab_size)

    """

    dec_out: torch.Tensor = None
    lm_score: torch.Tensor = None


class BeamSearchTransducer:
    """Beam search implementation for Transducer.

    Args:
        decoder: Decoder module.
        joint_network: Joint network module.
        beam_size: Size of the beam.
        lm: LM class.
        lm_weight: LM weight for soft fusion.
        search_type: Search algorithm to use during inference.
        max_sym_exp: Number of maximum symbol expansions at each time step. (TSD)
        u_max: Maximum expected target sequence length. (ALSD)
        nstep: Number of maximum expansion steps at each time step. (mAES)
        expansion_gamma: Allowed logp difference for prune-by-value method. (mAES)
        expansion_beta:
             Number of additional candidates for expanded hypotheses selection. (mAES)
        score_norm: Normalize final scores by length.
        nbest: Number of final hypothesis.
        streaming: Whether to perform chunk-by-chunk beam search.

    """

    def __init__(
        self,
        decoder,
        joint_network: JointNetwork,
        beam_size: int,
        lm: Optional[torch.nn.Module] = None,
        lm_weight: float = 0.1,
        search_type: str = "default",
        max_sym_exp: int = 3,
        u_max: int = 50,
        nstep: int = 2,
        expansion_gamma: float = 2.3,
        expansion_beta: int = 2,
        score_norm: bool = False,
        nbest: int = 1,
        streaming: bool = False,
    ) -> None:
        """Construct a BeamSearchTransducer object."""
        super().__init__()

        self.decoder = decoder
        self.joint_network = joint_network

        self.vocab_size = decoder.vocab_size

        assert (
            beam_size <= self.vocab_size
        ), "beam_size (%d) should be smaller than or equal to vocabulary size (%d)." % (
            beam_size,
            self.vocab_size,
        )
        self.beam_size = beam_size

        if search_type == "default":
            self.search_algorithm = self.default_beam_search
        elif search_type == "tsd":
            assert max_sym_exp > 1, "max_sym_exp (%d) should be greater than one." % (max_sym_exp)
            self.max_sym_exp = max_sym_exp

            self.search_algorithm = self.time_sync_decoding
        elif search_type == "alsd":
            assert not streaming, "ALSD is not available in streaming mode."

            assert u_max >= 0, "u_max should be a positive integer, a portion of max_T."
            self.u_max = u_max

            self.search_algorithm = self.align_length_sync_decoding
        elif search_type == "maes":
            assert self.vocab_size >= beam_size + expansion_beta, (
                "beam_size (%d) + expansion_beta (%d) "
                " should be smaller than or equal to vocab size (%d)."
                % (beam_size, expansion_beta, self.vocab_size)
            )
            self.max_candidates = beam_size + expansion_beta

            self.nstep = nstep
            self.expansion_gamma = expansion_gamma

            self.search_algorithm = self.modified_adaptive_expansion_search
        else:
            raise NotImplementedError("Specified search type (%s) is not supported." % search_type)

        self.use_lm = lm is not None

        if self.use_lm:
            assert hasattr(lm, "rnn_type"), "Transformer LM is currently not supported."

            self.sos = self.vocab_size - 1

            self.lm = lm
            self.lm_weight = lm_weight

        self.score_norm = score_norm
        self.nbest = nbest

        self.reset_inference_cache()

    def __call__(
        self,
        enc_out: torch.Tensor,
        is_final: bool = True,
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)
            is_final: Whether enc_out is the final chunk of data.

        Returns:
            nbest_hyps: N-best decoding results

        """
        self.decoder.set_device(enc_out.device)

        hyps = self.search_algorithm(enc_out)

        if is_final:
            self.reset_inference_cache()

            return self.sort_nbest(hyps)

        self.search_cache = hyps

        return hyps

    def reset_inference_cache(self) -> None:
        """Reset cache for decoder scoring and streaming."""
        self.decoder.score_cache = {}
        self.search_cache = None

    def sort_nbest(self, hyps: List[Hypothesis]) -> List[Hypothesis]:
        """Sort in-place hypotheses by score or score given sequence length.

        Args:
            hyps: Hypothesis.

        Return:
            hyps: Sorted hypothesis.

        """
        if self.score_norm:
            hyps.sort(key=lambda x: x.score / len(x.yseq), reverse=True)
        else:
            hyps.sort(key=lambda x: x.score, reverse=True)

        return hyps[: self.nbest]

    def recombine_hyps(self, hyps: List[Hypothesis]) -> List[Hypothesis]:
        """Recombine hypotheses with same label ID sequence.

        Args:
            hyps: Hypotheses.

        Returns:
            final: Recombined hypotheses.

        """
        final = {}

        for hyp in hyps:
            str_yseq = "_".join(map(str, hyp.yseq))

            if str_yseq in final:
                final[str_yseq].score = np.logaddexp(final[str_yseq].score, hyp.score)
            else:
                final[str_yseq] = hyp

        return [*final.values()]

    def select_k_expansions(
        self,
        hyps: List[ExtendedHypothesis],
        topk_idx: torch.Tensor,
        topk_logp: torch.Tensor,
    ) -> List[ExtendedHypothesis]:
        """Return K hypotheses candidates for expansion from a list of hypothesis.

        K candidates are selected according to the extended hypotheses probabilities
        and a prune-by-value method. Where K is equal to beam_size + beta.

        Args:
            hyps: Hypotheses.
            topk_idx: Indices of candidates hypothesis.
            topk_logp: Log-probabilities of candidates hypothesis.

        Returns:
            k_expansions: Best K expansion hypotheses candidates.

        """
        k_expansions = []

        for i, hyp in enumerate(hyps):
            hyp_i = [(int(k), hyp.score + float(v)) for k, v in zip(topk_idx[i], topk_logp[i])]
            k_best_exp = max(hyp_i, key=lambda x: x[1])[1]

            k_expansions.append(
                sorted(
                    filter(lambda x: (k_best_exp - self.expansion_gamma) <= x[1], hyp_i),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )

        return k_expansions

    def create_lm_batch_inputs(self, hyps_seq: List[List[int]]) -> torch.Tensor:
        """Make batch of inputs with left padding for LM scoring.

        Args:
            hyps_seq: Hypothesis sequences.

        Returns:
            : Padded batch of sequences.

        """
        max_len = max([len(h) for h in hyps_seq])

        return torch.LongTensor(
            [[self.sos] + ([0] * (max_len - len(h))) + h[1:] for h in hyps_seq],
            device=self.decoder.device,
        )

    def default_beam_search(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Beam search implementation without prefix search.

        Modified from https://arxiv.org/pdf/1211.3711.pdf

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        beam_k = min(self.beam_size, (self.vocab_size - 1))
        max_t = len(enc_out)

        if self.search_cache is not None:
            kept_hyps = self.search_cache
        else:
            kept_hyps = [
                Hypothesis(
                    score=0.0,
                    yseq=[0],
                    dec_state=self.decoder.init_state(1),
                )
            ]

        for t in range(max_t):
            hyps = kept_hyps
            kept_hyps = []

            while True:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                label = torch.full(
                    (1, 1),
                    max_hyp.yseq[-1],
                    dtype=torch.long,
                    device=self.decoder.device,
                )
                dec_out, state = self.decoder.score(
                    label,
                    max_hyp.yseq,
                    max_hyp.dec_state,
                )

                logp = torch.log_softmax(
                    self.joint_network(enc_out[t : t + 1, :], dec_out),
                    dim=-1,
                ).squeeze(0)
                top_k = logp[1:].topk(beam_k, dim=-1)

                kept_hyps.append(
                    Hypothesis(
                        score=(max_hyp.score + float(logp[0:1])),
                        yseq=max_hyp.yseq,
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                    )
                )

                if self.use_lm:
                    lm_scores, lm_state = self.lm.score(
                        torch.LongTensor([self.sos] + max_hyp.yseq[1:], device=self.decoder.device),
                        max_hyp.lm_state,
                        None,
                    )
                else:
                    lm_state = max_hyp.lm_state

                for logp, k in zip(*top_k):
                    score = max_hyp.score + float(logp)

                    if self.use_lm:
                        score += self.lm_weight * lm_scores[k + 1]

                    hyps.append(
                        Hypothesis(
                            score=score,
                            yseq=max_hyp.yseq + [int(k + 1)],
                            dec_state=state,
                            lm_state=lm_state,
                        )
                    )

                hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )
                if len(kept_most_prob) >= self.beam_size:
                    kept_hyps = kept_most_prob
                    break

        return kept_hyps

    def align_length_sync_decoding(
        self,
        enc_out: torch.Tensor,
    ) -> List[Hypothesis]:
        """Alignment-length synchronous beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            h: Encoder output sequences. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        t_max = int(enc_out.size(0))
        u_max = min(self.u_max, (t_max - 1))

        B = [Hypothesis(yseq=[0], score=0.0, dec_state=self.decoder.init_state(1))]
        final = []

        if self.use_lm:
            B[0].lm_state = self.lm.zero_state()

        for i in range(t_max + u_max):
            A = []

            B_ = []
            B_enc_out = []
            for hyp in B:
                u = len(hyp.yseq) - 1
                t = i - u

                if t > (t_max - 1):
                    continue

                B_.append(hyp)
                B_enc_out.append((t, enc_out[t]))

            if B_:
                beam_enc_out = torch.stack([b[1] for b in B_enc_out])
                beam_dec_out, beam_state = self.decoder.batch_score(B_)

                beam_logp = torch.log_softmax(
                    self.joint_network(beam_enc_out, beam_dec_out),
                    dim=-1,
                )
                beam_topk = beam_logp[:, 1:].topk(self.beam_size, dim=-1)

                if self.use_lm:
                    beam_lm_scores, beam_lm_states = self.lm.batch_score(
                        self.create_lm_batch_inputs([b.yseq for b in B_]),
                        [b.lm_state for b in B_],
                        None,
                    )

                for i, hyp in enumerate(B_):
                    new_hyp = Hypothesis(
                        score=(hyp.score + float(beam_logp[i, 0])),
                        yseq=hyp.yseq[:],
                        dec_state=hyp.dec_state,
                        lm_state=hyp.lm_state,
                    )

                    A.append(new_hyp)

                    if B_enc_out[i][0] == (t_max - 1):
                        final.append(new_hyp)

                    for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                        new_hyp = Hypothesis(
                            score=(hyp.score + float(logp)),
                            yseq=(hyp.yseq[:] + [int(k)]),
                            dec_state=self.decoder.select_state(beam_state, i),
                            lm_state=hyp.lm_state,
                        )

                        if self.use_lm:
                            new_hyp.score += self.lm_weight * beam_lm_scores[i, k]
                            new_hyp.lm_state = beam_lm_states[i]

                        A.append(new_hyp)

                B = sorted(A, key=lambda x: x.score, reverse=True)[: self.beam_size]
                B = self.recombine_hyps(B)

        if final:
            return final

        return B

    def time_sync_decoding(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Time synchronous beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        if self.search_cache is not None:
            B = self.search_cache
        else:
            B = [
                Hypothesis(
                    yseq=[0],
                    score=0.0,
                    dec_state=self.decoder.init_state(1),
                )
            ]

            if self.use_lm:
                B[0].lm_state = self.lm.zero_state()

        for enc_out_t in enc_out:
            A = []
            C = B

            enc_out_t = enc_out_t.unsqueeze(0)

            for v in range(self.max_sym_exp):
                D = []

                beam_dec_out, beam_state = self.decoder.batch_score(C)

                beam_logp = torch.log_softmax(
                    self.joint_network(enc_out_t, beam_dec_out),
                    dim=-1,
                )
                beam_topk = beam_logp[:, 1:].topk(self.beam_size, dim=-1)

                seq_A = [h.yseq for h in A]

                for i, hyp in enumerate(C):
                    if hyp.yseq not in seq_A:
                        A.append(
                            Hypothesis(
                                score=(hyp.score + float(beam_logp[i, 0])),
                                yseq=hyp.yseq[:],
                                dec_state=hyp.dec_state,
                                lm_state=hyp.lm_state,
                            )
                        )
                    else:
                        dict_pos = seq_A.index(hyp.yseq)

                        A[dict_pos].score = np.logaddexp(
                            A[dict_pos].score, (hyp.score + float(beam_logp[i, 0]))
                        )

                if v < (self.max_sym_exp - 1):
                    if self.use_lm:
                        beam_lm_scores, beam_lm_states = self.lm.batch_score(
                            self.create_lm_batch_inputs([c.yseq for c in C]),
                            [c.lm_state for c in C],
                            None,
                        )

                    for i, hyp in enumerate(C):
                        for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                            new_hyp = Hypothesis(
                                score=(hyp.score + float(logp)),
                                yseq=(hyp.yseq + [int(k)]),
                                dec_state=self.decoder.select_state(beam_state, i),
                                lm_state=hyp.lm_state,
                            )

                            if self.use_lm:
                                new_hyp.score += self.lm_weight * beam_lm_scores[i, k]
                                new_hyp.lm_state = beam_lm_states[i]

                            D.append(new_hyp)

                C = sorted(D, key=lambda x: x.score, reverse=True)[: self.beam_size]

            B = sorted(A, key=lambda x: x.score, reverse=True)[: self.beam_size]

        return B

    def modified_adaptive_expansion_search(
        self,
        enc_out: torch.Tensor,
    ) -> List[ExtendedHypothesis]:
        """Modified version of Adaptive Expansion Search (mAES).

        Based on AES (https://ieeexplore.ieee.org/document/9250505) and
                 NSC (https://arxiv.org/abs/2201.05420).

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        if self.search_cache is not None:
            kept_hyps = self.search_cache
        else:
            init_tokens = [
                ExtendedHypothesis(
                    yseq=[0],
                    score=0.0,
                    dec_state=self.decoder.init_state(1),
                )
            ]

            beam_dec_out, beam_state = self.decoder.batch_score(
                init_tokens,
            )

            if self.use_lm:
                beam_lm_scores, beam_lm_states = self.lm.batch_score(
                    self.create_lm_batch_inputs([h.yseq for h in init_tokens]),
                    [h.lm_state for h in init_tokens],
                    None,
                )

                lm_state = beam_lm_states[0]
                lm_score = beam_lm_scores[0]
            else:
                lm_state = None
                lm_score = None

            kept_hyps = [
                ExtendedHypothesis(
                    yseq=[0],
                    score=0.0,
                    dec_state=self.decoder.select_state(beam_state, 0),
                    dec_out=beam_dec_out[0],
                    lm_state=lm_state,
                    lm_score=lm_score,
                )
            ]

        for enc_out_t in enc_out:
            hyps = kept_hyps
            kept_hyps = []

            beam_enc_out = enc_out_t.unsqueeze(0)

            list_b = []
            for n in range(self.nstep):
                beam_dec_out = torch.stack([h.dec_out for h in hyps])

                beam_logp, beam_idx = torch.log_softmax(
                    self.joint_network(beam_enc_out, beam_dec_out),
                    dim=-1,
                ).topk(self.max_candidates, dim=-1)

                k_expansions = self.select_k_expansions(hyps, beam_idx, beam_logp)

                list_exp = []
                for i, hyp in enumerate(hyps):
                    for k, new_score in k_expansions[i]:
                        new_hyp = ExtendedHypothesis(
                            yseq=hyp.yseq[:],
                            score=new_score,
                            dec_out=hyp.dec_out,
                            dec_state=hyp.dec_state,
                            lm_state=hyp.lm_state,
                            lm_score=hyp.lm_score,
                        )

                        if k == 0:
                            list_b.append(new_hyp)
                        else:
                            new_hyp.yseq.append(int(k))

                            if self.use_lm:
                                new_hyp.score += self.lm_weight * float(hyp.lm_score[k])

                            list_exp.append(new_hyp)

                if not list_exp:
                    kept_hyps = sorted(
                        self.recombine_hyps(list_b), key=lambda x: x.score, reverse=True
                    )[: self.beam_size]

                    break
                else:
                    beam_dec_out, beam_state = self.decoder.batch_score(
                        list_exp,
                    )

                    if self.use_lm:
                        beam_lm_scores, beam_lm_states = self.lm.batch_score(
                            self.create_lm_batch_inputs([h.yseq for h in list_exp]),
                            [h.lm_state for h in list_exp],
                            None,
                        )

                    if n < (self.nstep - 1):
                        for i, hyp in enumerate(list_exp):
                            hyp.dec_out = beam_dec_out[i]
                            hyp.dec_state = self.decoder.select_state(beam_state, i)

                            if self.use_lm:
                                hyp.lm_state = beam_lm_states[i]
                                hyp.lm_score = beam_lm_scores[i]

                        hyps = list_exp[:]
                    else:
                        beam_logp = torch.log_softmax(
                            self.joint_network(beam_enc_out, beam_dec_out),
                            dim=-1,
                        )

                        for i, hyp in enumerate(list_exp):
                            hyp.score += float(beam_logp[i, 0])

                            hyp.dec_out = beam_dec_out[i]
                            hyp.dec_state = self.decoder.select_state(beam_state, i)

                            if self.use_lm:
                                hyp.lm_state = beam_lm_states[i]
                                hyp.lm_score = beam_lm_scores[i]

                        kept_hyps = sorted(
                            self.recombine_hyps(list_b + list_exp),
                            key=lambda x: x.score,
                            reverse=True,
                        )[: self.beam_size]

        return kept_hyps
