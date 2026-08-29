"""Regression tests for Paraformer timestamp flag precedence."""

import importlib
import unittest
from unittest.mock import MagicMock, patch

import torch


class _Tokenizer:
    def ids2tokens(self, token_ids):
        return ["你" for _ in token_ids]

    def tokens2text(self, tokens):
        return "".join(tokens)


class TestParaformerTimestampContract(unittest.TestCase):
    def _make_paraformer(self):
        from funasr.models.paraformer.model import Paraformer

        model = Paraformer.__new__(Paraformer)
        torch.nn.Module.__init__(model)
        model.beam_search = None
        model.sos = 1
        model.eos = 2
        model.blank_id = 0
        model.encode = MagicMock(
            return_value=(torch.zeros((1, 2, 2)), torch.tensor([2]))
        )
        model.calc_predictor = MagicMock(
            return_value=(
                torch.zeros((1, 1, 2)),
                torch.tensor([1.0]),
                torch.ones((1, 2)),
                torch.ones((1, 2)),
            )
        )
        model.cal_decoder_with_predictor = MagicMock(
            return_value=(
                torch.tensor([[[0.0, 0.0, 0.0, 4.0]]]),
                torch.tensor([1]),
            )
        )
        return model

    @staticmethod
    def _sentence_postprocess(tokens, timestamp=None):
        text = "".join(tokens)
        if timestamp is None:
            return text, None
        return text, [[0, 100]], None

    def test_pred_timestamp_precedence_and_output_timestamp_fallback(self):
        paraformer_module = importlib.import_module("funasr.models.paraformer.model")
        cases = (
            ({"output_timestamp": False}, False),
            ({"output_timestamp": True}, True),
            ({"pred_timestamp": True, "output_timestamp": False}, True),
            ({"pred_timestamp": False, "output_timestamp": True}, False),
        )

        with patch.object(
            paraformer_module,
            "ts_prediction_lfr6_standard",
            return_value=("", [[0, 100]]),
        ), patch.object(
            paraformer_module.postprocess_utils,
            "sentence_postprocess",
            side_effect=self._sentence_postprocess,
        ):
            for timestamp_kwargs, expected_timestamp in cases:
                with self.subTest(timestamp_kwargs=timestamp_kwargs):
                    results, _ = self._make_paraformer().inference(
                        torch.zeros((1, 2, 2)),
                        data_lengths=torch.tensor([[2]]),
                        key=["utt"],
                        tokenizer=_Tokenizer(),
                        frontend=None,
                        device="cpu",
                        data_type="fbank",
                        **timestamp_kwargs,
                    )
                    self.assertEqual("timestamp" in results[0], expected_timestamp)


if __name__ == "__main__":
    unittest.main()
