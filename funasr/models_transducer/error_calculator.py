"""Error Calculator module for Transducer."""

from typing import List, Optional, Tuple

import torch

from funasr.models_transducer.beam_search_transducer import BeamSearchTransducer
from funasr.models_transducer.decoder.abs_decoder import AbsDecoder
from funasr.models_transducer.joint_network import JointNetwork


class ErrorCalculator:
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
