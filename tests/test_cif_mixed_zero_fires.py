"""CPU CIF regressions using synthetic tensors; no model weights are needed."""

import importlib.util
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from funasr.models.paraformer.cif_predictor import (
    CifPredictorV2,
    CifPredictorV2Export,
    cif_v1,
    cif_v1_export,
)

FUNCTIONS = (cif_v1, cif_v1_export)
# All nonzero totals stay away from integer and rounding boundaries.
# -1 supplies exact zeros; 0 supplies positive mass below the firing threshold.
ALPHAS = {
    -1: [0.0] * 5,
    0: [0.11, 0.13, 0.09, 0.08, 0.12],
    1: [0.23, 0.34, 0.12, 0.33, 0.18],
    2: [0.62, 0.52, 0.43, 0.38, 0.35],
    3: [0.71, 0.67, 0.69, 0.61, 0.52],
}
PATTERNS = (
    (1, 0),
    (0, 1),
    (1, 0, 1),
    (1, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 0),
    (-1, -1),
    (1, 1),
    (2, 0, 3, 0),
    (0, 2, 0, 0, 3, 0, 0),
    (1,),
    (0,),
    (3, 2, 1),
)


def _inputs(pattern):
    alphas = torch.tensor([ALPHAS[n] for n in pattern], dtype=torch.float32)
    hidden = torch.arange(alphas.numel() * 3, dtype=torch.float32)
    hidden = hidden.reshape(len(pattern), 5, 3) / 17 - 2
    return hidden, alphas


def _sequential_frames(hidden, alphas):
    """Integrate one sample at a time, splitting each firing frame at mass 1."""
    samples = []
    for sample, weights in zip(hidden, alphas):
        mass = weights.new_zeros(())
        frame = torch.zeros_like(sample[0])
        emitted = []
        for value, weight in zip(sample, weights):
            if (mass + weight).item() >= 1.0:
                needed = 1.0 - mass
                emitted.append(frame + needed * value)
                mass = weight - needed
                frame = mass * value
            else:
                mass = mass + weight
                frame = frame + weight * value
        samples.append(torch.stack(emitted) if emitted else sample[:0])
    return samples


def _expected(hidden, alphas, function):
    samples = _sequential_frames(hidden, alphas)
    # Preserve the existing round (eager) versus floor (export) padding contract.
    lengths = alphas.sum(-1)
    width = int((lengths.round() if function is cif_v1 else lengths.floor()).max())
    padded = [torch.nn.functional.pad(sample, (0, 0, 0, width - len(sample))) for sample in samples]
    return torch.stack(padded), [len(sample) for sample in samples]


class _ExportCif(torch.nn.Module):
    def forward(self, hidden, alphas):
        return cif_v1_export(hidden, alphas, 1.0)


class TestCifMixedZeroFires(unittest.TestCase):
    def test_minimal_trailing_empty(self):
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        alphas = torch.tensor([[0.6, 0.6], [0.1, 0.1]])
        for function in FUNCTIONS:
            with self.subTest(function=function.__name__):
                frames, fires = function(hidden, alphas, 1.0)
                torch.testing.assert_close(frames, torch.tensor([[[1.8, 2.8]], [[0.0, 0.0]]]))
                self.assertEqual((fires >= 1).sum(1).tolist(), [1, 0])

    def test_predictor_callers(self):
        model = CifPredictorV2(idim=2, l_order=0, r_order=0, tail_threshold=0.0).eval()
        hidden = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
        mask = torch.tensor([[[1.0, 1.0]], [[0.0, 0.0]], [[0.0, 0.0]]])
        with torch.no_grad():
            # Real predictor layers produce alpha=0.6 before the padding mask.
            model.cif_output.weight.zero_()
            model.cif_output.bias.fill_(torch.logit(torch.tensor(0.6)))
            for predictor in (model, CifPredictorV2Export(model).eval()):
                with self.subTest(predictor=type(predictor).__name__):
                    frames, _, _, fires = predictor(hidden, mask=mask)
                    expected = torch.tensor([[[0.8, 1.8]], [[0.0, 0.0]], [[0.0, 0.0]]])
                    torch.testing.assert_close(frames, expected)
                    self.assertEqual((fires >= 1).sum(1).tolist(), [1, 0, 0])

    def test_frames_counts_and_padding(self):
        for pattern, function in itertools.product(PATTERNS, FUNCTIONS):
            with self.subTest(pattern=pattern, function=function.__name__):
                hidden, alphas = _inputs(pattern)
                expected, counts = _expected(hidden, alphas, function)
                self.assertEqual(counts, [max(n, 0) for n in pattern])
                frames, fires = function(hidden, alphas, 1.0)
                torch.testing.assert_close(frames, expected)
                self.assertEqual((fires >= 1).sum(1).tolist(), counts)
                self.assertEqual(frames.dtype, hidden.dtype)
                self.assertEqual(frames.device, hidden.device)

    def test_batch_permutations(self):
        hidden, alphas = _inputs((0, 2, 0, 3))
        for function in FUNCTIONS:
            expected, _ = _expected(hidden, alphas, function)
            _, expected_fires = function(hidden, alphas, 1.0)
            for permutation in itertools.permutations(range(4)):
                with self.subTest(function=function.__name__, permutation=permutation):
                    order = list(permutation)
                    frames, fires = function(hidden[order], alphas[order], 1.0)
                    torch.testing.assert_close(frames, expected[order])
                    torch.testing.assert_close(fires, expected_fires[order])

    def test_gradients_away_from_boundaries(self):
        for pattern, function in itertools.product(((2, 0, 3, 0), (3, 2, 1)), FUNCTIONS):
            with self.subTest(pattern=pattern, function=function.__name__):
                hidden, alphas = _inputs(pattern)
                hidden.requires_grad_()
                alphas.requires_grad_()
                expected, _ = _expected(hidden, alphas, function)
                frames, _ = function(hidden, alphas, 1.0)
                weights = torch.linspace(-0.7, 1.3, frames.numel()).reshape_as(frames)
                gradients = torch.autograd.grad((frames * weights).sum(), (hidden, alphas))
                reference = torch.autograd.grad((expected * weights).sum(), (hidden, alphas))
                for actual, wanted in zip(gradients, reference):
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, wanted, atol=2e-5, rtol=2e-5)

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
        "ONNX validation requires onnx and onnxruntime",
    )
    def test_onnx_dynamic_batches(self):
        import onnx
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cif.onnx"
            # Keep the scripted function's data-dependent empty-batch branch.
            torch.onnx.export(
                _ExportCif(),
                _inputs((2, 1)),
                str(path),
                opset_version=14,
                dynamo=False,
                input_names=["hidden", "alphas"],
                output_names=["frames", "fires"],
                dynamic_axes={
                    "hidden": {0: "batch", 1: "time"},
                    "alphas": {0: "batch", 1: "time"},
                    "frames": {0: "batch", 1: "tokens"},
                    "fires": {0: "batch", 1: "time"},
                },
            )
            onnx.checker.check_model(str(path))
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            for pattern in PATTERNS:
                with self.subTest(pattern=pattern):
                    hidden, alphas = _inputs(pattern)
                    # Also exercise a time dimension different from the export example.
                    hidden = torch.nn.functional.pad(hidden, (0, 0, 0, 2))
                    alphas = torch.nn.functional.pad(alphas, (0, 2))
                    expected, counts = _expected(hidden, alphas, cif_v1_export)
                    frames, fires = session.run(
                        None, {"hidden": hidden.numpy(), "alphas": alphas.numpy()}
                    )
                    self.assertEqual(frames.shape, tuple(expected.shape))
                    self.assertEqual(fires.shape, tuple(alphas.shape))
                    np.testing.assert_allclose(frames, expected.numpy(), atol=2e-5, rtol=2e-5)
                    self.assertEqual((fires >= 1).sum(1).tolist(), counts)
                    _, torch_fires = cif_v1_export(hidden, alphas, 1.0)
                    np.testing.assert_allclose(fires, torch_fires.numpy(), atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
