"""Regression tests for Paraformer timestamp flags and predictor arguments."""

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

    def test_inference_timestamp_boundaries_with_real_helper(self):
        from funasr.models.paraformer.cif_predictor import cif
        from funasr.utils import timestamp_tools

        cases = (
            # Three fires already provide the two tokens' boundary positions.
            ([0.0] + [0.25] * 12, [[150, 390], [390, 780]], False),
            # Two fires require the helper to normalize alphas to three boundaries.
            (
                [0.1, 0.1, 0.4, 0.6] + [0.0] * 4 + [0.1] * 10,
                [[90, 510], [510, 1080]],
                True,
            ),
        )
        for weights, expected, needs_fallback in cases:
            for offset in (0, 1000):
                with self.subTest(needs_fallback=needs_fallback, offset=offset):
                    model = self._make_paraformer()
                    alphas = torch.tensor([weights])
                    width = alphas.shape[1]
                    hidden = torch.zeros(1, width, 2)
                    _, peaks = cif(hidden, alphas, threshold=1.0)
                    model.encode.return_value = (hidden, torch.tensor([width]))
                    model.calc_predictor.return_value = (
                        torch.zeros(1, 2, 2),
                        torch.tensor([2.0]),
                        alphas,
                        peaks,
                    )
                    model.cal_decoder_with_predictor.return_value = (
                        torch.tensor([[[0.0, 0.0, 0.0, 4.0]] * 2]),
                        torch.tensor([2]),
                    )
                    with patch.object(
                        timestamp_tools,
                        "cif_wo_hidden",
                        wraps=timestamp_tools.cif_wo_hidden,
                    ) as fallback:
                        results, _ = model.inference(
                            hidden,
                            data_lengths=torch.tensor([[width]]),
                            key=["utt"],
                            tokenizer=_Tokenizer(),
                            frontend=None,
                            device="cpu",
                            data_type="fbank",
                            pred_timestamp=True,
                            begin_time=offset,
                        )
                    self.assertEqual(results[0]["text"], "你 你")
                    self.assertEqual(
                        results[0]["timestamp"],
                        [[start + offset, end + offset] for start, end in expected],
                    )
                    self.assertEqual(fallback.call_count, int(needs_fallback))


if __name__ == "__main__":
    unittest.main()
