import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import funasr.models.paraformer.model as paraformer_model
from funasr.models.paraformer.model import Paraformer


class _DummyTokenizer:
    def ids2tokens(self, ids):
        return ["x" for _ in ids]

    def tokens2text(self, tokens):
        return "".join(tokens)


class _DummyParaformer(Paraformer):
    def __init__(self, predictor, encoder_lens, predictor_width):
        torch.nn.Module.__init__(self)
        if predictor is not None:
            self.predictor = predictor
        self.encoder_lens = torch.tensor(encoder_lens, dtype=torch.long)
        if isinstance(predictor_width, tuple):
            self.alphas_width, self.pre_peak_width = predictor_width
        else:
            self.alphas_width = self.pre_peak_width = predictor_width
        self.beam_search = None
        self.ctc = None
        self.sos = 1
        self.eos = 2
        self.blank_id = 0

    def encode(self, speech, speech_lengths):
        batch = len(self.encoder_lens)
        max_len = int(self.encoder_lens.max().item())
        return torch.zeros(batch, max_len, 2), self.encoder_lens

    def calc_predictor(self, encoder_out, encoder_out_lens):
        batch = len(encoder_out_lens)
        return (
            torch.zeros(batch, 1, 2),
            torch.ones(batch),
            torch.zeros(batch, self.alphas_width),
            torch.zeros(batch, self.pre_peak_width),
        )

    def cal_decoder_with_predictor(
        self, encoder_out, encoder_out_lens, pre_acoustic_embeds, pre_token_length
    ):
        batch = len(encoder_out_lens)
        decoder_out = torch.zeros(batch, 1, 4)
        decoder_out[:, :, 3] = 1.0
        return decoder_out, pre_token_length


class TestParaformerTimestampPadding(unittest.TestCase):
    def _timestamp_helper_shapes(self, predictor, encoder_lens, predictor_width):
        model = _DummyParaformer(predictor, encoder_lens, predictor_width)
        seen = []

        def fake_timestamp(arg0, arg1, char_list, **kwargs):
            seen.append((arg0.shape[-1], arg1.shape[-1]))
            return "", [[0, 100]]

        with mock.patch.object(
            paraformer_model, "ts_prediction_lfr6_standard", side_effect=fake_timestamp
        ), mock.patch.object(
            paraformer_model.postprocess_utils,
            "sentence_postprocess",
            side_effect=lambda token, timestamp: ("".join(token), timestamp, None),
        ):
            model.inference(
                torch.zeros(len(encoder_lens), max(encoder_lens), 2),
                data_lengths=torch.tensor(encoder_lens, dtype=torch.long),
                key=[f"utt-{i}" for i in range(len(encoder_lens))],
                tokenizer=_DummyTokenizer(),
                frontend=None,
                data_type="fbank",
                device="cpu",
                pred_timestamp=True,
            )
        return seen

    def test_tail_mask_true_uses_per_sample_encoder_extent_plus_tail(self):
        predictor = SimpleNamespace(tail_mask=True, tail_threshold=0.45)
        self.assertEqual(
            self._timestamp_helper_shapes(predictor, [4, 8], 9),
            [(5, 5), (9, 9)],
        )

    def test_tail_mask_false_preserves_full_predictor_extent(self):
        predictor = SimpleNamespace(tail_mask=False, tail_threshold=0.45)
        self.assertEqual(
            self._timestamp_helper_shapes(predictor, [4, 8], 9),
            [(9, 9), (9, 9)],
        )

    def test_predictor_without_tail_mask_preserves_full_extent(self):
        predictor = SimpleNamespace(tail_threshold=0.45)
        self.assertEqual(
            self._timestamp_helper_shapes(predictor, [4, 8], 9),
            [(9, 9), (9, 9)],
        )

    def test_missing_predictor_attribute_preserves_full_extent(self):
        self.assertEqual(
            self._timestamp_helper_shapes(None, [4, 8], 9),
            [(9, 9), (9, 9)],
        )

    def test_zero_tail_threshold_does_not_add_tail_frame(self):
        predictor = SimpleNamespace(tail_mask=True, tail_threshold=0.0)
        self.assertEqual(
            self._timestamp_helper_shapes(predictor, [4, 8], 8),
            [(4, 4), (8, 8)],
        )

    def test_trim_length_is_clamped_to_both_timestamp_tensor_widths(self):
        predictor = SimpleNamespace(tail_mask=True, tail_threshold=0.45)
        self.assertEqual(
            self._timestamp_helper_shapes(predictor, [8], (7, 6)),
            [(6, 6)],
        )


if __name__ == "__main__":
    unittest.main()
