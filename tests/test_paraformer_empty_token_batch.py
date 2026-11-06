import unittest

import numpy as np
import torch

from funasr.auto.auto_model import AutoModel
from funasr.frontends.wav_frontend import WavFrontend
from funasr.models.contextual_paraformer.model import ContextualParaformer
from funasr.models.paraformer.model import Paraformer
from funasr.models.seaco_paraformer.model import SeacoParaformer


class _DummyTokenizer:
    def ids2tokens(self, ids):
        return ["x" for _ in ids]

    def tokens2text(self, tokens):
        return "".join(tokens)


class _ZeroTokenMixin:
    """Encoder/predictor stand-ins where every utterance predicts zero tokens."""

    def _setup(self):
        torch.nn.Module.__init__(self)
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.beam_search = None
        self.ctc = None
        self.sos = 1
        self.eos = 2
        self.blank_id = 0

    def encode(self, speech, speech_lengths):
        return torch.zeros(speech.size(0), 4, 2), torch.full((speech.size(0),), 4)

    def calc_predictor(self, encoder_out, encoder_out_lens):
        batch = encoder_out.size(0)
        return (
            torch.zeros(batch, 1, 2),
            torch.full((batch,), 0.4),
            torch.zeros(batch, 4),
            torch.zeros(batch, 4),
        )

    def generate_hotwords_list(self, *args, **kwargs):
        return None

    def cal_decoder_with_predictor(self, *args, **kwargs):
        raise AssertionError("decoder must not run when no token is predicted")

    _seaco_decode_with_ASF = cal_decoder_with_predictor


class _ZeroTokenParaformer(_ZeroTokenMixin, Paraformer):
    def __init__(self):
        self._setup()


class _ZeroTokenContextualParaformer(_ZeroTokenMixin, ContextualParaformer):
    def __init__(self):
        self._setup()


class _ZeroTokenSeacoParaformer(_ZeroTokenMixin, SeacoParaformer):
    def __init__(self, predictor_name="CifPredictorV3"):
        self._setup()
        self.predictor_name = predictor_name


def _frontend():
    return WavFrontend(cmvn_file=None, dither=0.0)


def _inference(model, keys, **kwargs):
    return model.inference(
        [np.zeros(1600, dtype=np.float32) for _ in keys],
        key=list(keys),
        frontend=_frontend(),
        device="cpu",
        **kwargs,
    )


class TestParaformerEmptyTokenBatch(unittest.TestCase):
    def test_paraformer_returns_one_keyed_result_per_utterance(self):
        results, meta_data = _inference(
            _ZeroTokenParaformer(), ["a", "b"], tokenizer=_DummyTokenizer()
        )
        self.assertEqual(results, [{"key": "a", "text": ""}, {"key": "b", "text": ""}])
        self.assertIn("batch_data_time", meta_data)

    def test_paraformer_keeps_timestamp_field(self):
        results, _ = _inference(
            _ZeroTokenParaformer(), ["a", "b"], tokenizer=_DummyTokenizer(), output_timestamp=True
        )
        self.assertEqual(
            results,
            [{"key": "a", "text": "", "timestamp": []}, {"key": "b", "text": "", "timestamp": []}],
        )

    def test_paraformer_without_tokenizer_returns_empty_token_ids(self):
        results, _ = _inference(_ZeroTokenParaformer(), ["a", "b"])
        self.assertEqual(results, [{"key": "a", "token_int": []}, {"key": "b", "token_int": []}])

    def test_contextual_paraformer_returns_one_keyed_result_per_utterance(self):
        results, _ = _inference(
            _ZeroTokenContextualParaformer(), ["a", "b"], tokenizer=_DummyTokenizer()
        )
        self.assertEqual(results, [{"key": "a", "text": ""}, {"key": "b", "text": ""}])

    def test_seaco_paraformer_returns_one_keyed_result_per_utterance(self):
        results, _ = _inference(
            _ZeroTokenSeacoParaformer(), ["a", "b"], tokenizer=_DummyTokenizer()
        )
        self.assertEqual(
            results,
            [{"key": "a", "text": "", "timestamp": []}, {"key": "b", "text": "", "timestamp": []}],
        )

        results, _ = _inference(
            _ZeroTokenSeacoParaformer("CifPredictorV2"), ["a"], tokenizer=_DummyTokenizer()
        )
        self.assertEqual(results, [{"key": "a", "text": ""}])

    def test_auto_model_keeps_every_utterance_of_the_batch(self):
        am = AutoModel.__new__(AutoModel)
        am.model = _ZeroTokenParaformer()
        am.kwargs = {
            "batch_size": 2,
            "disable_pbar": True,
            "device": "cpu",
            "frontend": _frontend(),
            "tokenizer": _DummyTokenizer(),
        }

        res = AutoModel.inference(
            am,
            [np.zeros(1600, dtype=np.float32), np.zeros(1600, dtype=np.float32)],
            key=["a", "b"],
        )

        self.assertEqual(res, [{"key": "a", "text": ""}, {"key": "b", "text": ""}])


if __name__ == "__main__":
    unittest.main()
