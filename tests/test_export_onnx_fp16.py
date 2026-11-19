"""onnx_fp16 export must be able to reload and save the fp32 graph; stubs only, no GPU or onnx needed."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from funasr.utils.export_utils import _onnx_opt_for_encdec


class _Decoder(torch.nn.Module):
    def forward(self, x):
        return x


class _Model(torch.nn.Module):
    export_name = "model"

    def __init__(self):
        super().__init__()
        self.decoder = _Decoder()

    def export_dummy_inputs(self):
        return torch.zeros(1, 2)

    def export_input_names(self):
        return ["x"]

    def export_output_names(self):
        return ["y"]

    def export_dynamic_axes(self):
        return {}

    def forward(self, x):
        return self.decoder(x)


class OnnxFp16ExportTest(unittest.TestCase):
    def test_converts_fp32_graph_to_fp16(self):
        saved = []
        fake_onnx = types.ModuleType("onnx")
        fake_onnx.load = lambda path: ("loaded", path)
        fake_onnx.save = lambda model, path: saved.append((model, path))
        fake_float16 = types.ModuleType("onnxconverter_common.float16")
        fake_float16.convert_float_to_float16 = lambda model, keep_io_types: ("fp16", model)
        fake_common = types.ModuleType("onnxconverter_common")
        fake_common.float16 = fake_float16

        def fake_export(model, inputs, path, **kwargs):
            Path(path).write_bytes(b"onnx")

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            sys.modules,
            {"onnx": fake_onnx, "onnxconverter_common": fake_common, "onnxconverter_common.float16": fake_float16},
        ), mock.patch.object(torch.Tensor, "cuda", lambda self: self), mock.patch.object(
            torch.nn.Module, "cuda", lambda self: self
        ), mock.patch.object(torch.onnx, "export", fake_export):
            _onnx_opt_for_encdec(_Model(), tmp, enable_fp16=False)

        self.assertEqual(
            saved,
            [(("fp16", ("loaded", f"{tmp}/model_hook.onnx")), f"{tmp}/model_hook_fp16.onnx")],
        )


if __name__ == "__main__":
    unittest.main()
