import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

from funasr.utils import export_utils


class _TinyExportable(torch.nn.Module):
    """Mimics the export hooks of FunASR models, with a data-dependent branch."""

    export_name = "tiny"

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 3)

    def forward(self, feats, feats_len):
        out = self.proj(feats)
        # torch.export cannot guard on this; tracing keeps only the branch taken
        if feats_len.sum() == 0:
            out = out * 0
        return out, feats_len

    def export(self, **kwargs):
        return self

    def export_dummy_inputs(self):
        return torch.randn(2, 5, 4), torch.tensor([5, 3], dtype=torch.int32)

    def export_input_names(self):
        return ["feats", "feats_len"]

    def export_output_names(self):
        return ["out", "out_len"]

    def export_dynamic_axes(self):
        return {
            "feats": {0: "batch", 1: "time"},
            "feats_len": {0: "batch"},
            "out": {0: "batch", 1: "time"},
            "out_len": {0: "batch"},
        }


@unittest.skipUnless(
    importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
    "ONNX export requires onnx and onnxruntime",
)
class OnnxExportTest(unittest.TestCase):
    def test_export_keeps_dynamic_axes(self):
        import onnxruntime as ort

        torch.manual_seed(0)
        model = _TinyExportable().eval()
        with tempfile.TemporaryDirectory() as directory:
            # export() evaluates os.path.dirname(init_param) even when output_dir is given
            export_utils.export(
                model,
                type="onnx",
                output_dir=directory,
                init_param=str(Path(directory) / "model.pt"),
            )
            path = Path(directory) / "tiny.onnx"
            self.assertTrue(path.exists())

            feats = torch.randn(3, 7, 4)
            feats_len = torch.tensor([7, 6, 2], dtype=torch.int32)
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            out, out_len = session.run(
                None, {"feats": feats.numpy(), "feats_len": feats_len.numpy()}
            )

        with torch.no_grad():
            wanted, _ = model(feats, feats_len)
        self.assertEqual(out.shape, (3, 7, 3))
        torch.testing.assert_close(torch.from_numpy(out), wanted, atol=1e-5, rtol=1e-5)
        self.assertEqual(out_len.tolist(), [7, 6, 2])


if __name__ == "__main__":
    unittest.main()
