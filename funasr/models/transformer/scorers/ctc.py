"""ScorerInterface implementation for CTC."""

import numpy as np
import torch

from funasr.models.transformer.scorers.ctc_prefix_score import CTCPrefixScore
from funasr.models.transformer.scorers.ctc_prefix_score import CTCPrefixScoreTH
from funasr.models.transformer.scorers.scorer_interface import BatchPartialScorerInterface


class CTCPrefixScorer(BatchPartialScorerInterface):
    """Decoder interface wrapper for CTCPrefixScore."""

    def __init__(self, ctc: torch.nn.Module, eos: int):
        """Initialize class.

        Args:
            ctc (torch.nn.Module): The CTC implementation.
                For example, :class:`espnet.nets.pytorch_backend.ctc.CTC`
            eos (int): The end-of-sequence id.

        """
        self.ctc = ctc
        self.eos = eos
        self.impl = None

    def init_state(self, x: torch.Tensor):
        """Get an initial state for decoding.

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        logp = self.ctc.log_softmax(x.unsqueeze(0)).detach().squeeze(0).cpu().numpy()
        # TODO(karita): use CTCPrefixScoreTH
        self.impl = CTCPrefixScore(logp, 0, self.eos, np)
        return 0, self.impl.initial_state()

    def select_state(self, state, i, new_id=None):
        """Select state with relative ids in the main beam search.

        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label id to select a state if necessary

        Returns:
            state: pruned state

        """
        if type(state) == tuple:
            if len(state) == 2:  # for CTCPrefixScore
                sc, st = state
                return sc[i], st[i]
            else:  # for CTCPrefixScoreTH (need new_id > 0)
                r, log_psi, f_min, f_max, scoring_idmap = state
                s = log_psi[i, new_id].expand(log_psi.size(1))
                if scoring_idmap is not None:
                    return r[:, :, i, scoring_idmap[i, new_id]], s, f_min, f_max
                else:
                    return r[:, :, i, new_id], s, f_min, f_max
        return None if state is None else state[i]

    def score_partial(self, y, ids, state, x):
        """Score new token.

        Args:
            y (torch.Tensor): 1D prefix token
            next_tokens (torch.Tensor): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (torch.Tensor): 2D encoder feature that generates ys

        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys

        """
        prev_score, state = state
        presub_score, new_st = self.impl(y.cpu(), ids.cpu(), state)
        tscore = torch.as_tensor(presub_score - prev_score, device=x.device, dtype=x.dtype)
        return tscore, (presub_score, new_st)

    def batch_init_state(self, x: torch.Tensor):
        """Get an initial state for decoding.

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        logp = self.ctc.log_softmax(x.unsqueeze(0))  # assuming batch_size = 1
        xlen = torch.tensor([logp.size(1)])
        self.impl = CTCPrefixScoreTH(logp, xlen, 0, self.eos)
        return None

    def batch_score_partial(self, y, ids, state, x):
        """Score new token.

        Args:
            y (torch.Tensor): 1D prefix token
            ids (torch.Tensor): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (torch.Tensor): 2D encoder feature that generates ys

        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys

        """
        batch_state = (
            (
                torch.stack([s[0] for s in state], dim=2),
                torch.stack([s[1] for s in state]),
                state[0][2],
                state[0][3],
            )
            if state[0] is not None
            else None
        )
        return self.impl(y, batch_state, ids)

    def extend_prob(self, x: torch.Tensor):
        """Extend probs for decoding.

        This extension is for streaming decoding
        as in Eq (14) in https://arxiv.org/abs/2006.14941

        Args:
            x (torch.Tensor): The encoded feature tensor

        """
        logp = self.ctc.log_softmax(x.unsqueeze(0))
        self.impl.extend_prob(logp)

    def extend_state(self, state):
        """Extend state for decoding.

        This extension is for streaming decoding
        as in Eq (14) in https://arxiv.org/abs/2006.14941

        Args:
            state: The states of hyps

        Returns: exteded state

        """
        new_state = []
        for s in state:
            new_state.append(self.impl.extend_state(s))

        return new_state
