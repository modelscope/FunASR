"""Offline regression for deterministic CTC evaluation losses."""

import importlib.util
from pathlib import Path

import torch


# Load only CTC to avoid importing unrelated optional FunASR dependencies.
MODULE = Path(__file__).resolve().parents[1] / "funasr/models/ctc/ctc.py"
spec = importlib.util.spec_from_file_location("ctc_dropout_eval", MODULE)
ctc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctc)


def test_ctc_eval_loss_is_independent_of_seed():
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(0)
        model = ctc.CTC(odim=3, encoder_output_size=3, dropout_rate=0.5).eval()
        hs_pad = torch.arange(18, dtype=torch.float32).reshape(1, 6, 3) / 10
        hlens = torch.tensor([6])
        ys_pad = torch.tensor([[1, 2]])
        ys_lens = torch.tensor([2])

        torch.manual_seed(1)
        first_loss = model(hs_pad, hlens, ys_pad, ys_lens)
        torch.manual_seed(2)
        second_loss = model(hs_pad, hlens, ys_pad, ys_lens)

        assert torch.isfinite(first_loss)
        torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)
