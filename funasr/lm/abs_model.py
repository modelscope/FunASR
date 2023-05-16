from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch

from funasr.modules.scorers.scorer_interface import BatchScorerInterface
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from funasr.modules.nets_utils import make_pad_mask
from funasr.torch_utils.device_funcs import force_gatherable
from funasr.models.base_model import FunASRModel

class AbsLM(torch.nn.Module, BatchScorerInterface, ABC):
    """The abstract LM class

    To share the loss calculation way among different models,
    We uses delegate pattern here:
    The instance of this class should be passed to "LanguageModel"

    >>> from funasr.lm.abs_model import AbsLM
    >>> lm = AbsLM()
    >>> model = LanguageESPnetModel(lm=lm)

    This "model" is one of mediator objects for "Task" class.

    """

    @abstractmethod
    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class LanguageModel(FunASRModel):
    def __init__(self, lm: AbsLM, vocab_size: int, ignore_id: int = 0):
        assert check_argument_types()
        super().__init__()
        self.lm = lm
        self.sos = 1
        self.eos = 2

        # ignore_id may be assumed as 0, shared with CTC-blank symbol for ASR.
        self.ignore_id = ignore_id

    def nll(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute negative log likelihood(nll)

        Normally, this function is called in batchify_nll.
        Args:
            text: (Batch, Length)
            text_lengths: (Batch,)
            max_lengths: int
        """
        batch_size = text.size(0)
        # For data parallel
        if max_length is None:
            text = text[:, : text_lengths.max()]
        else:
            text = text[:, :max_length]

        # 1. Create a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # text: (Batch, Length) -> x, y: (Batch, Length + 1)
        x = F.pad(text, [1, 0], "constant", self.sos)
        t = F.pad(text, [0, 1], "constant", self.ignore_id)
        for i, l in enumerate(text_lengths):
            t[i, l] = self.eos
        x_lengths = text_lengths + 1

        # 2. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        y, _ = self.lm(x, None)

        # 3. Calc negative log likelihood
        # nll: (BxL,)
        nll = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1), reduction="none")
        # nll: (BxL,) -> (BxL,)
        if max_length is None:
            nll.masked_fill_(make_pad_mask(x_lengths).to(nll.device).view(-1), 0.0)
        else:
            nll.masked_fill_(
                make_pad_mask(x_lengths, maxlen=max_length + 1).to(nll.device).view(-1),
                0.0,
            )
        # nll: (BxL,) -> (B, L)
        nll = nll.view(batch_size, -1)
        return nll, x_lengths

    def batchify_nll(
        self, text: torch.Tensor, text_lengths: torch.Tensor, batch_size: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute negative log likelihood(nll) from transformer language model

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            text: (Batch, Length)
            text_lengths: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase

        """
        total_num = text.size(0)
        if total_num <= batch_size:
            nll, x_lengths = self.nll(text, text_lengths)
        else:
            nlls = []
            x_lengths = []
            max_length = text_lengths.max()

            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_text = text[start_idx:end_idx, :]
                batch_text_lengths = text_lengths[start_idx:end_idx]
                # batch_nll: [B * T]
                batch_nll, batch_x_lengths = self.nll(
                    batch_text, batch_text_lengths, max_length=max_length
                )
                nlls.append(batch_nll)
                x_lengths.append(batch_x_lengths)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nlls)
            x_lengths = torch.cat(x_lengths)
        assert nll.size(0) == total_num
        assert x_lengths.size(0) == total_num
        return nll, x_lengths

    def forward(
        self, text: torch.Tensor, text_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        nll, y_lengths = self.nll(text, text_lengths)
        ntokens = y_lengths.sum()
        loss = nll.sum() / ntokens
        stats = dict(loss=loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, ntokens), loss.device)
        return loss, stats, weight

    def collect_feats(
        self, text: torch.Tensor, text_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {}
