'''
Search algorithms for ASR refer to WeNet
'''

import math
from collections import defaultdict
import math
from typing import Any, List, Optional, Tuple, Union, Dict

import torch
from torch.nn.utils.rnn import pad_sequence

WHISPER_LANGS = None




class DecodeResult:

    def __init__(self,
                 tokens: List[int],
                 score: float = 0.0,
                 confidence: float = 0.0,
                 tokens_confidence: List[float] = None,
                 times: List[int] = None,
                 nbest: List[List[int]] = None,
                 nbest_scores: List[float] = None,
                 nbest_times: List[List[int]] = None):
        """
        Args:
            tokens: decode token list
            score: the total decode score of this result
            confidence: the total confidence of this result, it's in 0~1
            tokens_confidence: confidence of each token
            times: timestamp of each token, list of (start, end)
            nbest: nbest result
            nbest_scores: score of each nbest
            nbest_times:
        """
        self.tokens = tokens
        self.score = score
        self.confidence = confidence
        self.tokens_confidence = tokens_confidence
        self.times = times
        self.nbest = nbest
        self.nbest_scores = nbest_scores
        self.nbest_times = nbest_times

def remove_duplicates_and_blank(hyp: List[int],
                                blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def ctc_greedy_search(ctc_probs: torch.Tensor,
                      ctc_lens: torch.Tensor,
                      blank_id: int = 0) -> List[DecodeResult]:
    batch_size = ctc_probs.shape[0]
    maxlen = ctc_probs.size(1)
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    
    max_len  = maxlen
    max_len = max_len if max_len > 0 else ctc_lens.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=ctc_lens.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = ctc_lens.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand  # (B, maxlen)
    mask = mask.to(ctc_probs.device)

    topk_index = topk_index.masked_fill_(mask, blank_id)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]
    scores = topk_prob.max(1)
    results = []
    for hyp in hyps:
        r = DecodeResult(remove_duplicates_and_blank(hyp, blank_id))
        results.append(r)
    return results[0].tokens




def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def make_non_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    This pad_mask is used in both encoder and decoder.

    1 for non-padded part and 0 for padded part.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    return ~make_pad_mask(lengths)


def subsequent_mask(
        size: int,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    arange = torch.arange(size, device=device)
    mask = arange.expand(size, size)
    arange = arange.unsqueeze(-1)
    mask = mask <= arange
    return mask


def mask_finished_preds(pred: torch.Tensor, flag: torch.Tensor,
                        eos: int) -> torch.Tensor:
    """
    If a sequence is finished, all of its branch should be <eos>

    Args:
        pred (torch.Tensor): A int array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size).
    """
    beam_size = pred.size(-1)
    finished = flag.repeat([1, beam_size])
    return pred.masked_fill_(finished, eos)


def mask_finished_scores(score: torch.Tensor,
                         flag: torch.Tensor) -> torch.Tensor:
    """
    If a sequence is finished, we only allow one alive branch. This function
    aims to give one branch a zero score and the rest -inf score.

    Args:
        score (torch.Tensor): A real value array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size, beam_size).
    """
    beam_size = score.size(-1)
    zero_mask = torch.zeros_like(flag, dtype=torch.bool)
    if beam_size > 1:
        unfinished = torch.cat((zero_mask, flag.repeat([1, beam_size - 1])),
                               dim=1)
        finished = torch.cat((flag, zero_mask.repeat([1, beam_size - 1])),
                             dim=1)
    else:
        unfinished = zero_mask
        finished = flag
    score.masked_fill_(unfinished, -float('inf'))
    score.masked_fill_(finished, 0)
    return score


def remove_duplicates_and_blank(hyp: List[int],
                                blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def pad_list(xs: List[torch.Tensor], pad_value: int):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    max_len = max([len(item) for item in xs])
    batchs = len(xs)
    ndim = xs[0].ndim
    if ndim == 1:
        pad_res = torch.zeros(batchs,
                              max_len,
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 2:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 3:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              xs[0].shape[2],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")
    pad_res.fill_(pad_value)
    for i in range(batchs):
        pad_res[i, :len(xs[i])] = xs[i]
    return pad_res


def add_sos_eos(ys_pad: torch.Tensor, sos: int, eos: int,
                ignore_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add <sos> and <eos> labels.

    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        sos (int): index of <sos>
        eos (int): index of <eeos>
        ignore_id (int): index of padding

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + 1)
        ys_out (torch.Tensor) : (B, Lmax + 1)

    Examples:
        >>> sos_id = 10
        >>> eos_id = 11
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,  4,  5],
                [ 4,  5,  6, -1, -1],
                [ 7,  8,  9, -1, -1]], dtype=torch.int32)
        >>> ys_in,ys_out=add_sos_eos(ys_pad, sos_id , eos_id, ignore_id)
        >>> ys_in
        tensor([[10,  1,  2,  3,  4,  5],
                [10,  4,  5,  6, 11, 11],
                [10,  7,  8,  9, 11, 11]])
        >>> ys_out
        tensor([[ 1,  2,  3,  4,  5, 11],
                [ 4,  5,  6, 11, -1, -1],
                [ 7,  8,  9, 11, -1, -1]])
    """
    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def add_whisper_tokens(special_tokens, ys_pad: torch.Tensor, ignore_id: int,
                       tasks: List[str], no_timestamp: bool, langs: List[str],
                       use_prev: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add whisper-style tokens.

    ([PREV] -> [previous text tokens or hotwords]).optional --
      ┌------------------------------------------------------↲
      ↓
    [sot] -> [language id] -> [transcribe] -> [begin time] -> [text tokens] -> [end time] -> ... -> [eot]    # noqa
        |          |                |-------> [no timestamps] -> [text tokens] ----------------------↑       # noqa
        |          |                                                                                 |       # noqa
        |          |--------> [translate]  -> [begin time] -> [text tokens] -> [end time] -> ... --->|       # noqa
        |                           |-------> [no timestamps] -> [text tokens] --------------------->|       # noqa
        |                                                                                            |       # noqa
        |--> [no speech(VAD)] ---------------------------------------------------------------------->|       # noqa

    Args:
        special_tokens: get IDs of special tokens
        ignore_id (int): index of padding
        no_timestamp (bool): whether to add timestamps tokens
        tasks (List[str]): list of task tags
        langs (List[str]): list of language tags

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + ?)
        ys_out (torch.Tensor) : (B, Lmax + ?)

    """
    assert len(langs) == ys_pad.size(0)
    assert len(tasks) == ys_pad.size(0)
    if use_prev:
        # i.e., hotword list
        _prev = [special_tokens["sot_prev"]]
        # append hotword list to _prev
        # ...
        raise NotImplementedError
    else:
        _prev = []

    _sot = []
    for task, lang in zip(tasks, langs):
        if task == "transcribe":
            task_id = special_tokens["transcribe"]
        elif task == "translate":
            task_id = special_tokens["translate"]
        elif task == "vad":
            task_id = special_tokens["no_speech"]
        else:
            raise NotImplementedError("unsupported task {}".format(task))
        language_id = special_tokens["sot"] + 1 + WHISPER_LANGS.index(lang)
        prefix = _prev + [special_tokens["sot"], language_id, task_id]
        if task == "transcribe" or task == "translate":
            if no_timestamp:
                prefix.append(special_tokens["no_timestamps"])
            else:
                prefix.append(special_tokens["timestamp_begin"])
                # add subsequent tokens
                # ...
                raise NotImplementedError
        elif task == "vad":
            prefix.append(special_tokens["no_speech"])
        else:
            raise NotImplementedError
        prefix = torch.tensor(prefix,
                              dtype=torch.long,
                              requires_grad=False,
                              device=ys_pad.device)
        _sot.append(prefix)

    _eot = torch.tensor([special_tokens["eot"]],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys

    ys_in = [torch.cat([prefix, y], dim=0) for prefix, y in zip(_sot, ys)]
    ys_out = [
        torch.cat([prefix[1:], y, _eot], dim=0) for prefix, y in zip(_sot, ys)
    ]
    return pad_list(ys_in, special_tokens["eot"]), pad_list(ys_out, ignore_id)


def log_add(*args) -> float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp



class DecodeResult:

    def __init__(self,
                 tokens: List[int],
                 score: float = 0.0,
                 confidence: float = 0.0,
                 tokens_confidence: List[float] = None,
                 times: List[int] = None,
                 nbest: List[List[int]] = None,
                 nbest_scores: List[float] = None,
                 nbest_times: List[List[int]] = None):
        """
        Args:
            tokens: decode token list
            score: the total decode score of this result
            confidence: the total confidence of this result, it's in 0~1
            tokens_confidence: confidence of each token
            times: timestamp of each token, list of (start, end)
            nbest: nbest result
            nbest_scores: score of each nbest
            nbest_times:
        """
        self.tokens = tokens
        self.score = score
        self.confidence = confidence
        self.tokens_confidence = tokens_confidence
        self.times = times
        self.nbest = nbest
        self.nbest_scores = nbest_scores
        self.nbest_times = nbest_times


class PrefixScore:
    """ For CTC prefix beam search """

    def __init__(self,
                 s: float = float('-inf'),
                 ns: float = float('-inf'),
                 v_s: float = float('-inf'),
                 v_ns: float = float('-inf'),
                 context_state = None,
                #  context_state: ContextState = None,
                 context_score: float = 0.0):
        self.s = s  # blank_ending_score
        self.ns = ns  # none_blank_ending_score
        self.v_s = v_s  # viterbi blank ending score
        self.v_ns = v_ns  # viterbi none blank ending score
        self.cur_token_prob = float('-inf')  # prob of current token
        self.times_s = []  # times of viterbi blank path
        self.times_ns = []  # times of viterbi none blank path
        self.context_state = context_state
        self.context_score = context_score
        self.has_context = False

    def score(self):
        return log_add(self.s, self.ns)

    def viterbi_score(self):
        return self.v_s if self.v_s > self.v_ns else self.v_ns

    def times(self):
        return self.times_s if self.v_s > self.v_ns else self.times_ns

    def total_score(self):
        return self.score() + self.context_score

    def copy_context(self, prefix_score):
        self.context_score = prefix_score.context_score
        self.context_state = prefix_score.context_state

    def update_context(self, context_graph, prefix_score, word_id):
        self.copy_context(prefix_score)
        (score, context_state) = context_graph.forward_one_step(
            prefix_score.context_state, word_id)
        self.context_score += score
        self.context_state = context_state


def ctc_greedy_search(ctc_probs: torch.Tensor,
                      ctc_lens: torch.Tensor,
                      blank_id: int = 0) -> List[DecodeResult]:
    batch_size = ctc_probs.shape[0]
    maxlen = ctc_probs.size(1)
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    mask = make_pad_mask(ctc_lens, maxlen).to(ctc_probs.device)  # (B, maxlen)
    topk_index = topk_index.masked_fill_(mask, blank_id)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]
    scores = topk_prob.max(1)
    results = []
    for hyp in hyps:
        r = DecodeResult(remove_duplicates_and_blank(hyp, blank_id))
        results.append(r)
    return results


def ctc_prefix_beam_search(
    ctc_probs: torch.Tensor,
    ctc_lens: torch.Tensor,
    beam_size: int,
    context_graph = None,
    # context_graph: ContextGraph = None,
    blank_id: int = 0,
) -> List[DecodeResult]:
    """
        Returns:
            List[List[List[int]]]: nbest result for each utterance
    """
    batch_size = ctc_probs.shape[0]
    results = []
    # CTC prefix beam search can not be paralleled, so search one by one
    for i in range(batch_size):
        ctc_prob = ctc_probs[i]
        num_t = ctc_lens[i]
        cur_hyps = [(tuple(),
                     PrefixScore(s=0.0,
                                 ns=-float('inf'),
                                 v_s=0.0,
                                 v_ns=0.0,
                                 context_state=None if context_graph is None
                                 else context_graph.root,
                                 context_score=0.0))]
        # 2. CTC beam search step by step
        for t in range(0, num_t):
            logp = ctc_prob[t]  # (vocab_size,)
            # key: prefix, value: PrefixScore
            next_hyps = defaultdict(lambda: PrefixScore())
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for u in top_k_index:
                u = u.item()
                prob = logp[u].item()
                for prefix, prefix_score in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if u == blank_id:  # blank
                        next_score = next_hyps[prefix]
                        next_score.s = log_add(next_score.s,
                                               prefix_score.score() + prob)
                        next_score.v_s = prefix_score.viterbi_score() + prob
                        next_score.times_s = prefix_score.times().copy()
                        # perfix not changed, copy the context from prefix
                        if context_graph and not next_score.has_context:
                            next_score.copy_context(prefix_score)
                            next_score.has_context = True
                    elif u == last:
                        #  Update *uu -> *u;
                        next_score1 = next_hyps[prefix]
                        next_score1.ns = log_add(next_score1.ns,
                                                 prefix_score.ns + prob)
                        if next_score1.v_ns < prefix_score.v_ns + prob:
                            next_score1.vs_ns = prefix_score.v_ns + prob
                            if next_score1.cur_token_prob < prob:
                                next_score1.cur_token_prob = prob
                                next_score1.times_ns = prefix_score.times_ns.copy(
                                )
                                next_score1.times_ns[-1] = t
                        if context_graph and not next_score1.has_context:
                            next_score1.copy_context(prefix_score)
                            next_score1.has_context = True

                        # Update *u-u -> *uu, - is for blank
                        n_prefix = prefix + (u, )
                        next_score2 = next_hyps[n_prefix]
                        next_score2.ns = log_add(next_score2.ns,
                                                 prefix_score.s + prob)
                        if next_score2.v_ns < prefix_score.v_s + prob:
                            next_score2.v_ns = prefix_score.v_s + prob
                            next_score2.cur_token_prob = prob
                            next_score2.times_ns = prefix_score.times_s.copy()
                            next_score2.times_ns.append(t)
                        if context_graph and not next_score2.has_context:
                            next_score2.update_context(context_graph,
                                                       prefix_score, u)
                            next_score2.has_context = True
                    else:
                        n_prefix = prefix + (u, )
                        next_score = next_hyps[n_prefix]
                        next_score.ns = log_add(next_score.ns,
                                                prefix_score.score() + prob)
                        if next_score.v_ns < prefix_score.viterbi_score(
                        ) + prob:
                            next_score.v_ns = prefix_score.viterbi_score(
                            ) + prob
                            next_score.cur_token_prob = prob
                            next_score.times_ns = prefix_score.times().copy()
                            next_score.times_ns.append(t)
                        if context_graph and not next_score.has_context:
                            next_score.update_context(context_graph,
                                                      prefix_score, u)
                            next_score.has_context = True

            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: x[1].total_score(),
                               reverse=True)
            cur_hyps = next_hyps[:beam_size]

        # We should backoff the context score/state when the context is
        # not fully matched at the last time.
        if context_graph is not None:
            for i, hyp in enumerate(cur_hyps):
                context_score, new_context_state = context_graph.finalize(
                    hyp[1].context_state)
                cur_hyps[i][1].context_score = context_score
                cur_hyps[i][1].context_state = new_context_state

        nbest = [y[0] for y in cur_hyps]
        nbest_scores = [y[1].total_score() for y in cur_hyps]
        nbest_times = [y[1].times() for y in cur_hyps]
        best = nbest[0]
        best_score = nbest_scores[0]
        best_time = nbest_times[0]
        results.append(
            DecodeResult(tokens=best,
                         score=best_score,
                         times=best_time,
                         nbest=nbest,
                         nbest_scores=nbest_scores,
                         nbest_times=nbest_times))
    return results


def attention_beam_search(
    model,
    encoder_out: torch.Tensor,
    encoder_mask: torch.Tensor,
    beam_size: int = 10,
    length_penalty: float = 0.0,
    infos: Dict[str, List[str]] = None,
) -> List[DecodeResult]:
    device = encoder_out.device
    batch_size = encoder_out.shape[0]
    # Let's assume B = batch_size and N = beam_size
    # 1. Encoder
    maxlen = encoder_out.size(1)
    encoder_dim = encoder_out.size(2)
    running_size = batch_size * beam_size
    encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
    # encoder_mask = encoder_mask.unsqueeze(1).repeat(1, beam_size, 1, 1).view(running_size, 1, maxlen)  # (B*N, 1, max_len)
    hyps = torch.ones([running_size, 1], dtype=torch.long,  device=device).fill_(model.sos)  # (B*N, 1)
    prefix_len = hyps.size(1)
    scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1), dtype=torch.float)
    scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(device)  # (B*N, 1)
    end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
    cache: Optional[List[torch.Tensor]] = None
    # 2. Decoder forward step by step
    for i in range(prefix_len, maxlen + 1):
        # Stop if all batch and all beam produce eos
        if end_flag.sum() == running_size:
            break
        # 2.1 Forward decoder step
        hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(running_size, 1, 1).to(device)  # (B*N, i, i)
        # logp: (B*N, vocab)
        logp, cache = model.decoder.forward_one_step(hyps, hyps_mask, encoder_out, cache)
        # logp, cache = model.hyp_decoder.forward_one_step_wenet_version(encoder_out, encoder_mask, hyps, hyps_mask, cache)
        # 2.2 First beam prune: select topk best prob at current time
        top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
        top_k_logp = mask_finished_scores(top_k_logp, end_flag)
        top_k_index = mask_finished_preds(top_k_index, end_flag, model.eos)
        # 2.3 Second beam prune: select topk score with history
        scores = scores + top_k_logp  # (B*N, N), broadcast add
        scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
        scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
        # Update cache to be consistent with new topk scores / hyps
        cache_index = (offset_k_index // beam_size).view(-1)  # (B*N)
        base_cache_index = (torch.arange(batch_size, device=device).view(-1, 1).repeat([1, beam_size]) * beam_size).view(-1)  # (B*N)
        cache_index = base_cache_index + cache_index
        cache = [torch.index_select(c, dim=0, index=cache_index) for c in cache]
        scores = scores.view(-1, 1)  # (B*N, 1)
        # 2.4. Compute base index in top_k_index,
        # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
        # then find offset_k_index in top_k_index
        base_k_index = torch.arange(batch_size, device=device).view(-1, 1).repeat([1, beam_size])  # (B, N)
        base_k_index = base_k_index * beam_size * beam_size
        best_k_index = base_k_index.view(-1) + offset_k_index.view(-1)  # (B*N)

        # 2.5 Update best hyps
        best_k_pred = torch.index_select(top_k_index.view(-1), dim=-1, index=best_k_index)  # (B*N)
        best_hyps_index = best_k_index // beam_size
        last_best_k_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)  # (B*N, i)
        hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)), dim=1)  # (B*N, i+1)

        # 2.6 Update end flag
        end_flag = torch.eq(hyps[:, -1], model.eos).view(-1, 1)

    # 3. Select best of best
    scores = scores.view(batch_size, beam_size)
    lengths = hyps.ne(model.eos).sum(dim=1).view(batch_size, beam_size).float()
    scores = scores / lengths.pow(length_penalty)
    best_scores, best_index = scores.max(dim=-1)
    best_hyps_index = best_index + torch.arange(batch_size, dtype=torch.long, device=device) * beam_size
    best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
    best_hyps = best_hyps[:, prefix_len:]

    results = []
    for i in range(batch_size):
        hyp = best_hyps[i]
        hyp = hyp[hyp != model.eos]
        results.append(DecodeResult(hyp.tolist()))
    return results


def attention_rescoring(
    model,
    ctc_prefix_results: List[DecodeResult],
    encoder_outs: torch.Tensor,
    encoder_lens: torch.Tensor,
    ctc_weight: float = 0.0,
    reverse_weight: float = 0.0,
    infos: Dict[str, List[str]] = None,
) -> List[DecodeResult]:
    """
        Args:
            ctc_prefix_results(List[DecodeResult]): ctc prefix beam search results
    """
    sos, eos = model.sos, model.eos
    device = encoder_outs.device
    assert encoder_outs.shape[0] == len(ctc_prefix_results)
    batch_size = encoder_outs.shape[0]
    results = []
    for b in range(batch_size):
        encoder_out = encoder_outs[b, :encoder_lens[b], :].unsqueeze(0)
        hyps = ctc_prefix_results[b].nbest
        ctc_scores = ctc_prefix_results[b].nbest_scores
        hyps_pad = pad_sequence([torch.tensor(hyp, device=device, dtype=torch.long) for hyp in hyps], True, model.ignore_id)  # (beam_size, max_hyps_len)
        hyps_lens = torch.tensor([len(hyp) for hyp in hyps], device=device, dtype=torch.long)  # (beam_size,)

        hyps_pad, _ = add_sos_eos(hyps_pad, sos, eos, model.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        prefix_len = 1
        decoder_out, r_decoder_out = model.forward_attention_decoder(hyps_pad, hyps_lens, encoder_out, reverse_weight)
        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        confidences = []
        tokens_confidences = []
        for i, hyp in enumerate(hyps):
            score = 0.0
            tc = []  # tokens confidences
            for j, w in enumerate(hyp):
                s = decoder_out[i][j + (prefix_len - 1)][w]
                score += s
                tc.append(math.exp(s))
            score += decoder_out[i][len(hyp) + (prefix_len - 1)][eos]
            # add right to left decoder score
            if reverse_weight > 0 and r_decoder_out.dim() > 0:
                r_score = 0.0
                for j, w in enumerate(hyp):
                    s = r_decoder_out[i][len(hyp) - j - 1 + (prefix_len - 1)][w]
                    r_score += s
                    tc[j] = (tc[j] + math.exp(s)) / 2
                r_score += r_decoder_out[i][len(hyp) + (prefix_len - 1)][eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            confidences.append(math.exp(score / (len(hyp) + 1)))
            # add ctc score
            score += ctc_scores[i] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
            tokens_confidences.append(tc)
        # results.append(
        #     DecodeResult(hyps[best_index],
        #                  best_score,
        #                  confidence=confidences[best_index],
        #                  times=ctc_prefix_results[b].nbest_times[best_index],
        #                  tokens_confidence=tokens_confidences[best_index]))
        results.append(
            DecodeResult(hyps[best_index],
                         best_score,
                         confidence=confidences[best_index],
                         times=[],
                         tokens_confidence=tokens_confidences[best_index]))
    return results





def paraformer_greedy_search(
        decoder_out: torch.Tensor,
        decoder_out_lens: torch.Tensor,
        cif_peaks: Optional[torch.Tensor] = None) -> List[DecodeResult]:
    batch_size = decoder_out.shape[0]
    maxlen = decoder_out.size(1)
    topk_prob, topk_index = decoder_out.topk(1, dim=2)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    topk_prob = topk_prob.view(batch_size, maxlen)
    results: List[DecodeResult] = []
    topk_index = topk_index.cpu().tolist()
    topk_prob = topk_prob.cpu().tolist()
    decoder_out_lens = decoder_out_lens.cpu().numpy()
    for (i, hyp) in enumerate(topk_index):
        confidence = 0.0
        tokens_confidence = []
        lens = decoder_out_lens[i]
        for logp in topk_prob[i][:lens]:
            tokens_confidence.append(math.exp(logp))
            confidence += logp
        r = DecodeResult(hyp[:lens],
                         tokens_confidence=tokens_confidence,
                         confidence=math.exp(confidence / lens))
        results.append(r)

    if cif_peaks is not None:
        for (b, peaks) in enumerate(cif_peaks):
            result = results[b]
            times = []
            n_token = 0
            for (i, peak) in enumerate(peaks):
                if n_token >= len(result.tokens):
                    break
                if peak > 1 - 1e-4:
                    times.append(i)
                    n_token += 1
            result.times = times
            # assert len(result.times) == len(result.tokens)
    return results



def _batch_beam_search(
    logit: torch.Tensor,
    masks: torch.Tensor,
    beam_size: int = 10,
    eos: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Perform batch beam search

        Args:
            logit: shape (batch_size, seq_length, vocab_size)
            masks: shape (batch_size, seq_length)
            beam_size: beam size

        Returns:
            indices: shape (batch_size, beam_size, seq_length)
            log_prob: shape (batch_size, beam_size)

        """

    batch_size, seq_length, vocab_size = logit.shape
    masks = ~masks
    # beam search
    with torch.no_grad():
        # b,t,v
        log_post = torch.nn.functional.log_softmax(logit, dim=-1)
        # b,k
        log_prob, indices = log_post[:, 0, :].topk(beam_size, sorted=True)
        end_flag = torch.eq(masks[:, 0], 1).view(-1, 1)
        # mask predictor and scores if end
        log_prob = mask_finished_scores(log_prob, end_flag)
        indices = mask_finished_preds(indices, end_flag, eos)
        # b,k,1
        indices = indices.unsqueeze(-1)

        for i in range(1, seq_length):
            # b,v
            scores = mask_finished_scores(log_post[:, i, :], end_flag)
            # b,v -> b,k,v
            topk_scores = scores.unsqueeze(1).repeat(1, beam_size, 1)
            # b,k,1 + b,k,v -> b,k,v
            top_k_logp = log_prob.unsqueeze(-1) + topk_scores

            # b,k,v -> b,k*v -> b,k
            log_prob, top_k_index = top_k_logp.view(batch_size,
                                                    -1).topk(beam_size,
                                                             sorted=True)

            index = mask_finished_preds(top_k_index, end_flag, eos)

            indices = torch.cat([indices, index.unsqueeze(-1)], dim=-1)

            end_flag = torch.eq(masks[:, i], 1).view(-1, 1)

        indices = torch.fmod(indices, vocab_size)

    return indices, log_prob





def paraformer_beam_search(decoder_out: torch.Tensor,
                           decoder_out_lens: torch.Tensor,
                           beam_size: int = 10,
                           eos: int = -1) -> List[DecodeResult]:
    mask = make_non_pad_mask(decoder_out_lens)
    nbest, nbest_scores = _para_beam_search(decoder_out, beam_size=beam_size)
    best = list(nbest[0])
    best_score = nbest_scores[0]
    best_time = []
    nbest_times = []
    # indices, _ = _batch_beam_search(decoder_out, mask, beam_size=beam_size, eos=eos)
    # best_hyps = indices[:, 0, :].cpu()
    # decoder_out_lens = decoder_out_lens.cpu()
    # results_original = []
    # # TODO(Mddct): scores, times etc
    # for (i, hyp) in enumerate(best_hyps.tolist()):
    #     r = DecodeResult(hyp[:decoder_out_lens.numpy()[i]])
    #     results_original.append(r)

    results = []
    results.append(
    DecodeResult(
        tokens=best,
        score=best_score,
        times=best_time,
        nbest=nbest,
        nbest_scores=nbest_scores,
        nbest_times=nbest_times))
    
    return results


def _para_beam_search(decoder_out, beam_size=10):
    seq_len, vocab_size = decoder_out.shape[1], decoder_out.shape[2]
    # 初始步骤
    log_probs, indices = torch.log_softmax(decoder_out[0, 0, :], dim=-1).topk(beam_size)
    # 初始化激活的假设（hypotheses），初始时每个假设只包含一个token
    hypotheses = [(log_prob, [index]) for log_prob, index in zip(log_probs, indices)]
    
    # 遍历每个时间步
    for t in range(1, seq_len):
        all_candidates = []
        # 扩展每个当前假设
        for log_prob, seq in hypotheses:
            # 计算当前假设下每个可能扩展的概率
            next_log_probs, next_indices = torch.log_softmax(decoder_out[0, t, :], dim=-1).topk(beam_size)
            all_candidates.extend(
                (log_prob + next_log_prob, seq + [next_index])
                for next_log_prob, next_index in zip(next_log_probs, next_indices)
            )
        
        # 选出新的 beam_size 个最佳假设
        all_candidates.sort(reverse=True, key=lambda x: x[0])
        hypotheses = all_candidates[:beam_size]
    
    nbest_scores = []
    nbest = []
    for log_prob, seq in hypotheses:
        nbest.append(tuple(item.item() for item in seq))
        nbest_scores.append(log_prob.item())
    return nbest, nbest_scores


def hyp_beam_search(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_len: torch.Tensor,
    prefix: torch.Tensor,
    beam_size: int = 10,
    length_penalty: float = 0.0,
) -> List[DecodeResult]:
    decoder = model.hyp_decoder
    device = encoder_out.device
    batch_size = encoder_out.shape[0]
    maxlen = encoder_out.size(1)
    encoder_dim = encoder_out.size(2)
    running_size = batch_size * beam_size
    hyps = torch.full([1], model.sos, dtype=torch.long, device=device)

    if prefix[0].nbest is not None:
        prefix = torch.tensor(prefix[0].nbest).to(device)
        hyps = hyps.unsqueeze(1).repeat(1, beam_size, 1).view(running_size, -1)
        hyps = torch.cat([hyps, prefix], dim=-1)
    else:
        prefix = torch.tensor(prefix[0].tokens).to(device)
        prefix = torch.cat([hyps, prefix], dim=-1)
        hyps = prefix.unsqueeze(1).repeat(1, beam_size, 1).view(running_size, -1)

    # 扩展encoder输出和掩码以适应每个beam
    encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(running_size, maxlen, encoder_dim) # (B*N, maxlen, encoder_dim)
    encoder_mask = make_non_pad_mask(encoder_len).unsqueeze(1).repeat(1, beam_size, 1).view(running_size, 1, maxlen)  # (B*N, 1, max_len)
    hyps_len = torch.tensor([hyps.size(-1)])
    hyps_mask = (~make_pad_mask(hyps_len))[:, None, :].to(device).repeat(beam_size, 1, 1)  # (B*N, 1, max_len)    
    
    scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1), dtype=torch.float)
    scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(device)  # (B*N, 1)
    end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)

    # 解码每一步
    for decoding_idx in range(hyps_len):
        # Stop if all batch and all beam produce eos
        if end_flag.sum() == running_size:
            break
        # 单步解码
        decoder_out, _ = decoder.forward_one_step(encoder_out, encoder_mask, hyps, hyps_mask, decoding_idx)
        # top_k_logp, top_k_index = torch.softmax(decoder_out, dim=-1).topk(beam_size)
        top_k_logp, top_k_index = torch.log_softmax(decoder_out, dim=-1).topk(beam_size)
        top_k_logp = mask_finished_scores(top_k_logp, end_flag)
        top_k_index = mask_finished_preds(top_k_index, end_flag, model.eos)
        # 2.3 Second beam prune: select topk score with history
        scores = scores + top_k_logp  # (B*N, N), broadcast add
        scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
        scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
        scores = scores.view(-1, 1)  # (B*N, 1)
        base_k_index = torch.arange(batch_size, device=device).view(-1, 1).repeat([1, beam_size])  # (B, N)
        base_k_index = base_k_index * beam_size * beam_size
        best_k_index = base_k_index.view(-1) + offset_k_index.view(-1)  # (B*N)

        # 2.5 Update best hyps
        best_k_pred = torch.index_select(top_k_index.view(-1), dim=-1, index=best_k_index)  # (B*N)
        best_hyps_index = best_k_index // beam_size
        last_best_k_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)  # (B*N, i)
        if decoding_idx+1 < prefix.size(-1):
            last_best_k_hyps[:, decoding_idx+1] = best_k_pred
        else:
            last_best_k_hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)), dim=1)
        hyps = last_best_k_hyps 

        # 2.6 Update end flag
        end_flag = torch.eq(hyps[:, -1], model.eos).view(-1, 1)

    # 3. Select best of best
    scores = scores.view(batch_size, beam_size)
    scores =  torch.exp(scores)
    lengths = hyps.ne(model.eos).sum(dim=1).view(batch_size, beam_size).float()
    scores = scores / lengths.pow(length_penalty)
    best_scores, best_index = scores.max(dim=-1)
    best_hyps_index = best_index + torch.arange(batch_size, dtype=torch.long, device=device) * beam_size
    best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
    best_hyps = best_hyps[:, 1:-1]

    results = []
    nbest = []
    for i in range(beam_size):
        a = hyps[i][1:]
        a = a[a != model.eos]
        nbest.append(tuple(a.tolist()))

    
    for i in range(batch_size):
        hyp = best_hyps[i]
        hyp = hyp[hyp != model.eos]
        results.append(
            DecodeResult(
                tokens=hyp.tolist(),
                score=[],
                times=[],
                nbest=nbest,
                nbest_scores=scores,
                nbest_times=[]))
    return results



def autoregressive_beam_rescoring(
    model,
    prefix_results: List[DecodeResult],
    encoder_outs: torch.Tensor,
    encoder_lens: torch.Tensor,
    beam_size: int = 5,  # 增加束大小作为参数
    reverse_weight: float = 0.0,
) -> List[DecodeResult]:

    sos, eos = model.sos, model.eos
    device = encoder_outs.device
    batch_size = encoder_outs.shape[0]
    results = []

    for b in range(batch_size):
        encoder_out = encoder_outs[b, :encoder_lens[b], :].unsqueeze(0)
        hyps = prefix_results[b].tokens
        hyps_score = [prefix_results[b].score]
        hyps_pad = torch.tensor(hyps, device=device, dtype=torch.long).unsqueeze(0)  # (1, max_hyps_len)  
        hyps_lens = torch.tensor([len(hyp) for hyp in hyps_pad], device=device, dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, sos, eos, model.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        
        # Initialize beam with the prefix result
        beam = [(hyps_pad, 0.0)]

        for step in range(hyps_lens - 1):  # Iterate from the first token after sos up to before eos
            new_beam = []
            for hyps_pad, hyp_score in beam:
                # Forward pass through the decoder
                decoder_out, _ = model.forward_attention_decoder(hyps_pad, hyps_lens, encoder_out, reverse_weight)
                recurrent_logits = decoder_out[0, step].unsqueeze(0)  # (1, vocab)
                log_probs = torch.nn.functional.log_softmax(recurrent_logits, dim=-1)

                # Get top k tokens for the current time step
                topk_probs, topk_indices = log_probs.topk(beam_size)
                topk_probs = topk_probs.squeeze(0)
                topk_indices = topk_indices.squeeze(0)

                # Update beam with new candidates
                for prob, idx in zip(topk_probs, topk_indices):
                    new_hyp = hyps_pad.clone()  # Copy the current hypothesis
                    new_hyp[0, step+1] = idx  # Update the token at the current step
                    new_score = hyp_score + prob.item()
                    new_beam.append((new_hyp, new_score))

            # Keep top beam_size elements in the beam
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_size]


        # Select the best hypothesis from the final beam
        nbest, nbest_scores = [], []
        for hyp, score in beam:
            nbest.append(hyp[0, 1:].tolist())
            nbest_scores.append(0.5*score + 0.5*hyps_score[0])
        tokens, score = max(beam, key=lambda x: x[1])
        score = 0.5*score + 0.5*hyps_score[0]
        results.append(
            DecodeResult(tokens=tokens,
                         score=score,
                         nbest=nbest,
                         nbest_scores=nbest_scores))

    return results



def autoregressive_beam_rescoring(
    model,
    prefix_results: List[DecodeResult],
    encoder_outs: torch.Tensor,
    encoder_lens: torch.Tensor,
    beam_size: int = 5,  # 增加束大小作为参数
    reverse_weight: float = 0.0,
) -> List[DecodeResult]:

    sos, eos = model.sos, model.eos
    device = encoder_outs.device
    batch_size = encoder_outs.shape[0]
    results = []

    for b in range(batch_size):
        encoder_out = encoder_outs[b, :encoder_lens[b], :].unsqueeze(0)
        hyps = prefix_results[b].tokens
        hyps_score = [prefix_results[b].score]
        hyps_pad = torch.tensor(hyps, device=device, dtype=torch.long).unsqueeze(0)  # (1, max_hyps_len)  
        hyps_lens = torch.tensor([len(hyp) for hyp in hyps_pad], device=device, dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, sos, eos, model.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        
        # Initialize beam with the prefix result
        beam = [(hyps_pad, 0.0)]

        for step in range(hyps_lens - 1):  # Iterate from the first token after sos up to before eos
            new_beam = []
            for hyps_pad, hyp_score in beam:
                # Forward pass through the decoder
                decoder_out, _ = model.forward_attention_decoder(hyps_pad, hyps_lens, encoder_out, reverse_weight)
                recurrent_logits = decoder_out[0, step].unsqueeze(0)  # (1, vocab)
                log_probs = torch.nn.functional.log_softmax(recurrent_logits, dim=-1)

                # Get top k tokens for the current time step
                topk_probs, topk_indices = log_probs.topk(beam_size)
                topk_probs = topk_probs.squeeze(0)
                topk_indices = topk_indices.squeeze(0)

                # Update beam with new candidates
                for prob, idx in zip(topk_probs, topk_indices):
                    new_hyp = hyps_pad.clone()  # Copy the current hypothesis
                    new_hyp[0, step+1] = idx  # Update the token at the current step
                    new_score = hyp_score + prob.item()
                    new_beam.append((new_hyp, new_score))

            # Keep top beam_size elements in the beam
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_size]


        # Select the best hypothesis from the final beam
        nbest, nbest_scores = [], []
        for hyp, score in beam:
            nbest.append(hyp[0, 1:].tolist())
            nbest_scores.append(score)
            # nbest_scores.append(0.5*score + 0.5*hyps_score[0])
        tokens, score = max(beam, key=lambda x: x[1])
        # score = 0.5*score + 0.5*hyps_score[0]
        results.append(
            DecodeResult(tokens=tokens,
                         score=score,
                         nbest=nbest,
                         nbest_scores=nbest_scores))

    return results

